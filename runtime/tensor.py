"""Custom tensor serialization format for Nexus distributed inference engine.

Wire format:
    Header (32B): magic b'NXTS', version uint16, dtype uint16 enum,
                  ndim uint32, data_offset uint32, data_size uint64,
                  crc32c uint32, flags uint32
    Shape (ndim * 8B): int64 per dimension
    Padding to 64B alignment
    Raw contiguous tensor data
"""

from __future__ import annotations

import enum
import mmap
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Conditional imports -- numpy & torch are optional at import time
# ---------------------------------------------------------------------------
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAGIC = b"NXTS"
HEADER_VERSION: int = 1
HEADER_SIZE: int = 32  # bytes
ALIGNMENT: int = 64  # byte boundary for data section
HEADER_STRUCT = struct.Struct("<4sHHIIQII")  # 4+2+2+4+4+8+4+4 = 32


# ---------------------------------------------------------------------------
# DType enum
# ---------------------------------------------------------------------------
class DType(enum.IntEnum):
    """Supported tensor element types."""

    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2
    INT8 = 3
    INT16 = 4
    INT32 = 5
    INT64 = 6
    UINT8 = 7
    BOOL = 8
    FLOAT64 = 9


DTYPE_SIZES: Dict[DType, int] = {
    DType.FLOAT32: 4,
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.INT8: 1,
    DType.INT16: 2,
    DType.INT32: 4,
    DType.INT64: 8,
    DType.UINT8: 1,
    DType.BOOL: 1,
    DType.FLOAT64: 8,
}

# ---------------------------------------------------------------------------
# Numpy / Torch dtype mapping tables
# ---------------------------------------------------------------------------
NUMPY_DTYPE_MAP: Dict[DType, str] = {
    DType.FLOAT32: "float32",
    DType.FLOAT16: "float16",
    DType.BFLOAT16: "float16",  # numpy has no bfloat16; upcast on read
    DType.INT8: "int8",
    DType.INT16: "int16",
    DType.INT32: "int32",
    DType.INT64: "int64",
    DType.UINT8: "uint8",
    DType.BOOL: "bool",
    DType.FLOAT64: "float64",
}

TORCH_DTYPE_MAP: Dict[DType, str] = {
    DType.FLOAT32: "torch.float32",
    DType.FLOAT16: "torch.float16",
    DType.BFLOAT16: "torch.bfloat16",
    DType.INT8: "torch.int8",
    DType.INT16: "torch.int16",
    DType.INT32: "torch.int32",
    DType.INT64: "torch.int64",
    DType.UINT8: "torch.uint8",
    DType.BOOL: "torch.bool",
    DType.FLOAT64: "torch.float64",
}


def _torch_dtype(dtype: DType) -> "torch.dtype":
    """Resolve a *DType* to the real ``torch.dtype`` object."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")
    mapping = {
        DType.FLOAT32: torch.float32,
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
        DType.INT8: torch.int8,
        DType.INT16: torch.int16,
        DType.INT32: torch.int32,
        DType.INT64: torch.int64,
        DType.UINT8: torch.uint8,
        DType.BOOL: torch.bool,
        DType.FLOAT64: torch.float64,
    }
    return mapping[dtype]


# ---------------------------------------------------------------------------
# TensorFlags bitfield
# ---------------------------------------------------------------------------
class TensorFlags(enum.IntFlag):
    """Bitfield flags embedded in the tensor header."""

    NONE = 0
    COMPRESSED = 1 << 0
    QUANTIZED = 1 << 1
    PINNED_MEMORY = 1 << 2
    CONTIGUOUS = 1 << 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crc32c(data: bytes) -> int:
    """Compute a CRC-32C checksum.

    Falls back to zlib.crc32 (CRC-32) when the *crc32c* C-extension is not
    available.  In production the `crc32c` package should be installed for
    hardware-accelerated checksums.
    """
    try:
        import crc32c as _crc32c_mod  # type: ignore[import-untyped]

        return _crc32c_mod.crc32c(data) & 0xFFFFFFFF
    except ImportError:
        return zlib.crc32(data) & 0xFFFFFFFF


def _align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
    """Round *offset* up to the next multiple of *alignment*."""
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_tensor(
    data: bytes,
    shape: Sequence[int],
    dtype: DType,
    flags: TensorFlags = TensorFlags.CONTIGUOUS,
) -> bytes:
    """Serialize raw tensor *data* into the Nexus wire format.

    Parameters
    ----------
    data:
        Raw, contiguous tensor bytes in row-major (C) order.
    shape:
        Tuple / list of dimension sizes.
    dtype:
        Element type.
    flags:
        Optional bitfield flags.

    Returns
    -------
    bytes
        Complete wire-format blob ready to send over the network or write to
        disk.
    """
    ndim = len(shape)
    # Shape section immediately follows the fixed header
    shape_bytes = struct.pack(f"<{ndim}q", *shape)
    shape_section_end = HEADER_SIZE + len(shape_bytes)
    data_offset = _align_offset(shape_section_end, ALIGNMENT)
    padding_size = data_offset - shape_section_end
    data_size = len(data)

    # CRC-32C covers the *payload* data only (not header/shape)
    checksum = _crc32c(data)

    header = HEADER_STRUCT.pack(
        MAGIC,
        HEADER_VERSION,
        int(dtype),
        ndim,
        data_offset,
        data_size,
        checksum,
        int(flags),
    )

    return header + shape_bytes + (b"\x00" * padding_size) + data


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def deserialize_header(buf: Union[bytes, memoryview]) -> Dict[str, Any]:
    """Parse *only* the 32-byte fixed header (useful for routing decisions).

    Returns a dict with keys: magic, version, dtype, ndim, data_offset,
    data_size, checksum, flags.
    """
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"Buffer too small for header: {len(buf)} < {HEADER_SIZE}")
    raw = bytes(buf[:HEADER_SIZE])
    (
        magic,
        version,
        dtype_val,
        ndim,
        data_offset,
        data_size,
        checksum,
        flags_val,
    ) = HEADER_STRUCT.unpack(raw)

    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r} (expected {MAGIC!r})")
    if version != HEADER_VERSION:
        raise ValueError(f"Unsupported version {version} (expected {HEADER_VERSION})")

    return {
        "magic": magic,
        "version": version,
        "dtype": DType(dtype_val),
        "ndim": ndim,
        "data_offset": data_offset,
        "data_size": data_size,
        "checksum": checksum,
        "flags": TensorFlags(flags_val),
    }


def deserialize_tensor(
    buf: Union[bytes, memoryview],
    verify_checksum: bool = True,
) -> Tuple[bytes, Tuple[int, ...], DType]:
    """Deserialize a Nexus wire-format blob.

    Returns
    -------
    (data, shape, dtype)
        Raw bytes payload, shape tuple, and element dtype.
    """
    header = deserialize_header(buf)
    ndim: int = header["ndim"]
    data_offset: int = header["data_offset"]
    data_size: int = header["data_size"]
    dtype: DType = header["dtype"]
    checksum: int = header["checksum"]

    # Parse shape
    shape_start = HEADER_SIZE
    shape_end = shape_start + ndim * 8
    if len(buf) < shape_end:
        raise ValueError("Buffer too small for shape section")
    shape: Tuple[int, ...] = struct.unpack(
        f"<{ndim}q", bytes(buf[shape_start:shape_end])
    )

    # Extract data
    data_end = data_offset + data_size
    if len(buf) < data_end:
        raise ValueError(f"Buffer too small for data: need {data_end}, got {len(buf)}")
    data = bytes(buf[data_offset:data_end])

    if verify_checksum:
        computed = _crc32c(data)
        if computed != checksum:
            raise ValueError(
                f"Checksum mismatch: expected 0x{checksum:08X}, got 0x{computed:08X}"
            )

    return data, shape, dtype


# ---------------------------------------------------------------------------
# Numpy / Torch convenience helpers
# ---------------------------------------------------------------------------


def deserialize_to_numpy(
    buf: Union[bytes, memoryview],
    verify_checksum: bool = True,
) -> "np.ndarray":
    """Deserialize wire format directly into a NumPy ndarray."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy is not installed")
    data, shape, dtype = deserialize_tensor(buf, verify_checksum=verify_checksum)
    np_dtype = NUMPY_DTYPE_MAP[dtype]
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    # bfloat16 was stored as float16 in numpy -- flag a warning
    if dtype == DType.BFLOAT16:
        import warnings

        warnings.warn(
            "bfloat16 tensor was loaded as float16 in NumPy "
            "(numpy lacks native bfloat16 support).",
            stacklevel=2,
        )
    return arr.copy()  # own the memory


def deserialize_to_torch(
    buf: Union[bytes, memoryview],
    device: str = "cpu",
    verify_checksum: bool = True,
) -> "torch.Tensor":
    """Deserialize wire format directly into a PyTorch tensor on *device*."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")
    data, shape, dtype = deserialize_tensor(buf, verify_checksum=verify_checksum)
    td = _torch_dtype(dtype)
    tensor = torch.frombuffer(bytearray(data), dtype=td).reshape(shape)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def numpy_to_nexus(
    arr: "np.ndarray",
    dtype: Optional[DType] = None,
    flags: TensorFlags = TensorFlags.CONTIGUOUS,
) -> bytes:
    """Serialize a NumPy array into the Nexus wire format."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy is not installed")
    # Ensure C-contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if dtype is None:
        # Reverse-lookup
        _np_to_nexus = {v: k for k, v in NUMPY_DTYPE_MAP.items()}
        np_name = str(arr.dtype)
        if np_name not in _np_to_nexus:
            raise ValueError(f"Unsupported numpy dtype: {np_name}")
        dtype = _np_to_nexus[np_name]
    return serialize_tensor(
        data=arr.tobytes(),
        shape=arr.shape,
        dtype=dtype,
        flags=flags,
    )


def torch_to_nexus(
    tensor: "torch.Tensor",
    dtype: Optional[DType] = None,
    flags: TensorFlags = TensorFlags.CONTIGUOUS,
) -> bytes:
    """Serialize a PyTorch tensor into the Nexus wire format."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")
    tensor = tensor.detach().cpu().contiguous()
    if dtype is None:
        _torch_to_nexus = {
            v: k
            for k, v in {
                DType.FLOAT32: torch.float32,
                DType.FLOAT16: torch.float16,
                DType.BFLOAT16: torch.bfloat16,
                DType.INT8: torch.int8,
                DType.INT16: torch.int16,
                DType.INT32: torch.int32,
                DType.INT64: torch.int64,
                DType.UINT8: torch.uint8,
                DType.BOOL: torch.bool,
                DType.FLOAT64: torch.float64,
            }.items()
        }
        if tensor.dtype not in _torch_to_nexus:
            raise ValueError(f"Unsupported torch dtype: {tensor.dtype}")
        dtype = _torch_to_nexus[tensor.dtype]
    raw = tensor.numpy().tobytes()
    return serialize_tensor(
        data=raw,
        shape=tuple(tensor.shape),
        dtype=dtype,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# MMapTensor -- zero-copy memory-mapped access
# ---------------------------------------------------------------------------


class MMapTensor:
    """Zero-copy, memory-mapped access to a Nexus tensor stored on disk.

    Useful for large model weight files: the OS pages data in on demand
    without reading the entire file into RAM.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._header: Optional[Dict[str, Any]] = None
        self._shape: Optional[Tuple[int, ...]] = None
        self._dtype: Optional[DType] = None
        self._data_offset: int = 0
        self._data_size: int = 0
        self._open()

    # -- lifecycle -----------------------------------------------------------

    def _open(self) -> None:
        file_size = self._path.stat().st_size
        self._fd = os.open(str(self._path), os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, file_size, access=mmap.ACCESS_READ)

        self._header = deserialize_header(self._mm)
        ndim = self._header["ndim"]
        shape_start = HEADER_SIZE
        shape_end = shape_start + ndim * 8
        self._shape = struct.unpack(f"<{ndim}q", self._mm[shape_start:shape_end])
        self._dtype = self._header["dtype"]
        self._data_offset = self._header["data_offset"]
        self._data_size = self._header["data_size"]

    def close(self) -> None:
        """Release the memory map and file descriptor."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "MMapTensor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # -- properties ----------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> DType:
        assert self._dtype is not None
        return self._dtype

    @property
    def header(self) -> Dict[str, Any]:
        assert self._header is not None
        return dict(self._header)

    @property
    def data_view(self) -> memoryview:
        """Return a zero-copy *memoryview* over the raw tensor data."""
        assert self._mm is not None
        return memoryview(self._mm)[
            self._data_offset : self._data_offset + self._data_size
        ]

    # -- materialisation helpers ---------------------------------------------

    def to_numpy(self) -> "np.ndarray":
        """Materialise as a NumPy array (still backed by mmap)."""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is not installed")
        np_dtype = NUMPY_DTYPE_MAP[self.dtype]
        arr = np.frombuffer(self.data_view, dtype=np_dtype).reshape(self.shape)
        return arr  # stays mmap-backed for zero-copy reads

    def to_torch(self, device: str = "cpu") -> "torch.Tensor":
        """Materialise as a PyTorch tensor."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed")
        # torch.frombuffer needs a writable buffer or a bytes copy
        raw = bytes(self.data_view)
        td = _torch_dtype(self.dtype)
        tensor = torch.frombuffer(bytearray(raw), dtype=td).reshape(self.shape)
        if device != "cpu":
            tensor = tensor.to(device)
        return tensor

    def verify_checksum(self) -> bool:
        """Verify the stored CRC-32C against the memory-mapped data."""
        assert self._header is not None
        expected = self._header["checksum"]
        computed = _crc32c(bytes(self.data_view))
        return computed == expected

    def __repr__(self) -> str:
        return (
            f"MMapTensor(path={self._path!s}, shape={self.shape}, "
            f"dtype={self.dtype.name}, size={self._data_size}B)"
        )
