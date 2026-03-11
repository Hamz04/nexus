"""CUDA memory pool manager for Nexus distributed inference engine.

Provides pre-allocated GPU memory pools with best-fit allocation,
free-block coalescing, and Prometheus-style metrics tracking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Conditional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.cuda

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """A contiguous region inside a memory pool."""

    offset: int
    size: int
    allocated: bool = False
    stream: Optional[Any] = None  # torch.cuda.Stream or None
    _alloc_time: float = 0.0

    @property
    def end(self) -> int:
        return self.offset + self.size

    def __repr__(self) -> str:
        state = "ALLOC" if self.allocated else "FREE"
        return (
            f"Block({state}, offset=0x{self.offset:X}, "
            f"size={self.size:#_x}, end=0x{self.end:X})"
        )


@dataclass
class MemoryBlock:
    """Handle returned to callers after a successful allocation."""

    offset: int
    size: int
    pool_id: int
    stream: Optional[Any] = None
    _block_ref: Optional[Block] = field(default=None, repr=False)

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class PoolStats:
    """Snapshot of pool utilisation."""

    total_bytes: int = 0
    allocated_bytes: int = 0
    free_bytes: int = 0
    peak_usage_bytes: int = 0
    num_allocations: int = 0
    num_frees: int = 0
    fragmentation_ratio: float = 0.0
    num_free_blocks: int = 0
    largest_free_block: int = 0


# ---------------------------------------------------------------------------
# CUDAMemoryPool
# ---------------------------------------------------------------------------


class CUDAMemoryPool:
    """Pre-allocated contiguous CUDA memory pool with best-fit allocation.

    Parameters
    ----------
    device_id:
        CUDA device ordinal (e.g. 0 for ``cuda:0``).
    pool_size_gb:
        Size of the pre-allocated buffer in gibibytes.
    """

    def __init__(self, device_id: int = 0, pool_size_gb: float = 2.0) -> None:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch with CUDA support is required for CUDAMemoryPool"
            )

        self._device_id = device_id
        self._device = torch.device(f"cuda:{device_id}")
        self._pool_size = int(pool_size_gb * (1 << 30))  # GiB -> bytes
        self._lock = threading.Lock()

        # Pre-allocate one large contiguous buffer
        with torch.cuda.device(self._device):
            self._buffer = torch.cuda.ByteTensor(self._pool_size)
            self._base_ptr: int = self._buffer.data_ptr()

        # Internal bookkeeping -- list kept sorted by offset
        self._blocks: List[Block] = [
            Block(offset=0, size=self._pool_size, allocated=False)
        ]

        # Stats
        self._total_allocated: int = 0
        self._peak_usage: int = 0
        self._num_allocations: int = 0
        self._num_frees: int = 0

        # Prometheus-style metrics dict
        self.metrics: Dict[str, Any] = self._build_metrics()

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "CUDAMemoryPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    # -- allocation ----------------------------------------------------------

    def allocate(self, size_bytes: int, stream: Optional[Any] = None) -> MemoryBlock:
        """Allocate *size_bytes* from the pool using best-fit strategy.

        Parameters
        ----------
        size_bytes:
            Number of bytes requested.  Will be rounded up to 256-byte
            alignment for coalescing efficiency.
        stream:
            Optional ``torch.cuda.Stream`` to tag the block.

        Returns
        -------
        MemoryBlock
            Handle with the offset into the pool buffer.

        Raises
        ------
        MemoryError
            If the pool cannot satisfy the allocation.
        """
        # Round up to 256B alignment
        aligned_size = (size_bytes + 255) & ~255

        with self._lock:
            best_idx: Optional[int] = None
            best_waste = float("inf")

            for idx, blk in enumerate(self._blocks):
                if not blk.allocated and blk.size >= aligned_size:
                    waste = blk.size - aligned_size
                    if waste < best_waste:
                        best_waste = waste
                        best_idx = idx
                        if waste == 0:
                            break  # perfect fit

            if best_idx is None:
                raise MemoryError(
                    f"CUDAMemoryPool: cannot allocate {aligned_size} bytes "
                    f"({self._free_bytes()} bytes free across "
                    f"{self._num_free_blocks()} blocks)"
                )

            blk = self._blocks[best_idx]
            remainder = blk.size - aligned_size

            # Split block if there is leftover space
            if remainder > 0:
                new_free = Block(
                    offset=blk.offset + aligned_size,
                    size=remainder,
                    allocated=False,
                )
                self._blocks.insert(best_idx + 1, new_free)

            blk.size = aligned_size
            blk.allocated = True
            blk.stream = stream
            blk._alloc_time = time.monotonic()

            self._total_allocated += aligned_size
            self._num_allocations += 1
            if self._total_allocated > self._peak_usage:
                self._peak_usage = self._total_allocated

            self._refresh_metrics()

            return MemoryBlock(
                offset=blk.offset,
                size=aligned_size,
                pool_id=self._device_id,
                stream=stream,
                _block_ref=blk,
            )

    # -- free & coalesce -----------------------------------------------------

    def free(self, block: MemoryBlock) -> None:
        """Return *block* to the pool and coalesce adjacent free blocks."""
        with self._lock:
            # Find the internal block by offset
            target: Optional[Block] = block._block_ref
            if target is None:
                for blk in self._blocks:
                    if blk.offset == block.offset and blk.allocated:
                        target = blk
                        break
            if target is None or not target.allocated:
                raise ValueError(f"Block at offset 0x{block.offset:X} is not allocated")

            target.allocated = False
            target.stream = None
            self._total_allocated -= target.size
            self._num_frees += 1

            self._coalesce()
            self._refresh_metrics()

    def _coalesce(self) -> None:
        """Merge adjacent free blocks (caller must hold _lock)."""
        merged: List[Block] = []
        for blk in self._blocks:
            if (
                merged
                and not merged[-1].allocated
                and not blk.allocated
                and merged[-1].end == blk.offset
            ):
                merged[-1].size += blk.size
            else:
                merged.append(blk)
        self._blocks = merged

    # -- stats ---------------------------------------------------------------

    def _free_bytes(self) -> int:
        return sum(b.size for b in self._blocks if not b.allocated)

    def _num_free_blocks(self) -> int:
        return sum(1 for b in self._blocks if not b.allocated)

    @property
    def stats(self) -> PoolStats:
        """Return a snapshot of pool utilisation."""
        with self._lock:
            free_blocks = [b for b in self._blocks if not b.allocated]
            free_bytes = sum(b.size for b in free_blocks)
            largest_free = max((b.size for b in free_blocks), default=0)
            frag = 1.0 - (largest_free / free_bytes) if free_bytes > 0 else 0.0
            return PoolStats(
                total_bytes=self._pool_size,
                allocated_bytes=self._total_allocated,
                free_bytes=free_bytes,
                peak_usage_bytes=self._peak_usage,
                num_allocations=self._num_allocations,
                num_frees=self._num_frees,
                fragmentation_ratio=round(frag, 4),
                num_free_blocks=len(free_blocks),
                largest_free_block=largest_free,
            )

    def _build_metrics(self) -> Dict[str, Any]:
        s = self.stats
        return {
            "gpu_memory_pool_total_bytes": s.total_bytes,
            "gpu_memory_pool_allocated_bytes": s.allocated_bytes,
            "gpu_memory_pool_free_bytes": s.free_bytes,
            "gpu_memory_pool_peak_bytes": s.peak_usage_bytes,
            "gpu_memory_pool_utilization": (
                round(s.allocated_bytes / s.total_bytes, 4) if s.total_bytes else 0.0
            ),
            "gpu_memory_pool_fragmentation": s.fragmentation_ratio,
            "gpu_memory_pool_allocations_total": s.num_allocations,
            "gpu_memory_pool_frees_total": s.num_frees,
            "gpu_memory_pool_free_blocks": s.num_free_blocks,
            "gpu_memory_pool_largest_free_block_bytes": s.largest_free_block,
            "gpu_memory_pool_device": self._device_id,
        }

    def _refresh_metrics(self) -> None:
        self.metrics = self._build_metrics()

    # -- release -------------------------------------------------------------

    def release(self) -> None:
        """Free the entire CUDA buffer."""
        with self._lock:
            if hasattr(self, "_buffer") and self._buffer is not None:
                del self._buffer
                self._buffer = None  # type: ignore[assignment]
            self._blocks.clear()
            self._total_allocated = 0

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"CUDAMemoryPool(device={self._device_id}, "
            f"total={self._pool_size / (1 << 30):.1f}GiB, "
            f"used={s.allocated_bytes / (1 << 20):.1f}MiB, "
            f"frag={s.fragmentation_ratio:.2%})"
        )


# ---------------------------------------------------------------------------
# PinnedMemoryPool -- CPU pinned (page-locked) memory for fast H2D copies
# ---------------------------------------------------------------------------


class PinnedMemoryPool:
    """Pool of CPU page-locked (pinned) memory for fast host-to-device
    transfers.

    Uses the same best-fit + coalescing strategy as :class:`CUDAMemoryPool`
    but allocates via ``torch.cuda.HostAllocator`` (pinned memory).
    """

    def __init__(self, pool_size_mb: float = 512.0) -> None:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch with CUDA support is required for PinnedMemoryPool"
            )

        self._pool_size = int(pool_size_mb * (1 << 20))  # MiB -> bytes
        self._lock = threading.Lock()

        # Pin a large host buffer
        self._buffer = torch.empty(self._pool_size, dtype=torch.uint8, pin_memory=True)
        self._base_ptr: int = self._buffer.data_ptr()

        self._blocks: List[Block] = [
            Block(offset=0, size=self._pool_size, allocated=False)
        ]

        self._total_allocated: int = 0
        self._peak_usage: int = 0
        self._num_allocations: int = 0
        self._num_frees: int = 0

        self.metrics: Dict[str, Any] = self._build_metrics()

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "PinnedMemoryPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    # -- allocation (mirrors CUDAMemoryPool) ---------------------------------

    def allocate(self, size_bytes: int) -> MemoryBlock:
        """Allocate *size_bytes* of pinned host memory (best-fit)."""
        aligned_size = (size_bytes + 63) & ~63  # 64B alignment

        with self._lock:
            best_idx: Optional[int] = None
            best_waste = float("inf")

            for idx, blk in enumerate(self._blocks):
                if not blk.allocated and blk.size >= aligned_size:
                    waste = blk.size - aligned_size
                    if waste < best_waste:
                        best_waste = waste
                        best_idx = idx
                        if waste == 0:
                            break

            if best_idx is None:
                raise MemoryError(
                    f"PinnedMemoryPool: cannot allocate {aligned_size} bytes"
                )

            blk = self._blocks[best_idx]
            remainder = blk.size - aligned_size

            if remainder > 0:
                new_free = Block(
                    offset=blk.offset + aligned_size,
                    size=remainder,
                    allocated=False,
                )
                self._blocks.insert(best_idx + 1, new_free)

            blk.size = aligned_size
            blk.allocated = True
            blk._alloc_time = time.monotonic()

            self._total_allocated += aligned_size
            self._num_allocations += 1
            if self._total_allocated > self._peak_usage:
                self._peak_usage = self._total_allocated

            self._refresh_metrics()

            return MemoryBlock(
                offset=blk.offset,
                size=aligned_size,
                pool_id=-1,  # CPU
                _block_ref=blk,
            )

    def free(self, block: MemoryBlock) -> None:
        """Return *block* to the pool."""
        with self._lock:
            target: Optional[Block] = block._block_ref
            if target is None:
                for blk in self._blocks:
                    if blk.offset == block.offset and blk.allocated:
                        target = blk
                        break
            if target is None or not target.allocated:
                raise ValueError(f"Block at offset 0x{block.offset:X} is not allocated")

            target.allocated = False
            self._total_allocated -= target.size
            self._num_frees += 1

            self._coalesce()
            self._refresh_metrics()

    def _coalesce(self) -> None:
        merged: List[Block] = []
        for blk in self._blocks:
            if (
                merged
                and not merged[-1].allocated
                and not blk.allocated
                and merged[-1].end == blk.offset
            ):
                merged[-1].size += blk.size
            else:
                merged.append(blk)
        self._blocks = merged

    # -- stats / metrics -----------------------------------------------------

    @property
    def stats(self) -> PoolStats:
        with self._lock:
            free_blocks = [b for b in self._blocks if not b.allocated]
            free_bytes = sum(b.size for b in free_blocks)
            largest_free = max((b.size for b in free_blocks), default=0)
            frag = 1.0 - (largest_free / free_bytes) if free_bytes > 0 else 0.0
            return PoolStats(
                total_bytes=self._pool_size,
                allocated_bytes=self._total_allocated,
                free_bytes=free_bytes,
                peak_usage_bytes=self._peak_usage,
                num_allocations=self._num_allocations,
                num_frees=self._num_frees,
                fragmentation_ratio=round(frag, 4),
                num_free_blocks=len(free_blocks),
                largest_free_block=largest_free,
            )

    def _build_metrics(self) -> Dict[str, Any]:
        s = self.stats
        return {
            "pinned_memory_pool_total_bytes": s.total_bytes,
            "pinned_memory_pool_allocated_bytes": s.allocated_bytes,
            "pinned_memory_pool_free_bytes": s.free_bytes,
            "pinned_memory_pool_peak_bytes": s.peak_usage_bytes,
            "pinned_memory_pool_utilization": (
                round(s.allocated_bytes / s.total_bytes, 4) if s.total_bytes else 0.0
            ),
            "pinned_memory_pool_fragmentation": s.fragmentation_ratio,
            "pinned_memory_pool_allocations_total": s.num_allocations,
            "pinned_memory_pool_frees_total": s.num_frees,
        }

    def _refresh_metrics(self) -> None:
        self.metrics = self._build_metrics()

    # -- release -------------------------------------------------------------

    def release(self) -> None:
        with self._lock:
            if hasattr(self, "_buffer") and self._buffer is not None:
                del self._buffer
                self._buffer = None  # type: ignore[assignment]
            self._blocks.clear()
            self._total_allocated = 0

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"PinnedMemoryPool(total={self._pool_size / (1 << 20):.0f}MiB, "
            f"used={s.allocated_bytes / (1 << 20):.1f}MiB, "
            f"frag={s.fragmentation_ratio:.2%})"
        )
