"""Python worker that receives batched tensors from the Go router.

Provides shared-memory IPC transport (via /dev/shm), a gRPC-style worker
server, worker pool management, and graceful shutdown handling.
"""

from __future__ import annotations

import mmap
import multiprocessing as mp
import os
import signal
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

import torch

from . import tensor as nxtensor
from .inference import InferenceEngine, QuantizationConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHM_BASE = Path("/dev/shm")
HEARTBEAT_INTERVAL_SEC: float = 2.0
SEGMENT_CLEANUP_AGE_SEC: float = 300.0  # 5 min stale threshold


# ---------------------------------------------------------------------------
# Shared memory transport
# ---------------------------------------------------------------------------


class SharedMemoryTransport:
    """IPC transport backed by POSIX shared memory (``/dev/shm``).

    Each tensor exchange uses a named segment that both the Go router and
    this Python worker can memory-map.
    """

    def __init__(self, prefix: str = "nexus") -> None:
        self._prefix = prefix
        self._segments: Dict[str, mmap.mmap] = {}
        self._fds: Dict[str, int] = {}
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def create_segment(self, name: str, size: int) -> mmap.mmap:
        """Create (or open) a shared memory segment of *size* bytes.

        The segment is stored at ``/dev/shm/<prefix>_<name>``.
        """
        seg_path = SHM_BASE / f"{self._prefix}_{name}"
        with self._lock:
            if name in self._segments:
                return self._segments[name]

            fd = os.open(
                str(seg_path),
                os.O_CREAT | os.O_RDWR,
                0o666,
            )
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)

            self._fds[name] = fd
            self._segments[name] = mm
            logger.info(
                "shm_segment_created",
                name=name,
                size=size,
                path=str(seg_path),
            )
            return mm

    def open_segment(self, name: str) -> mmap.mmap:
        """Open an *existing* shared memory segment for reading."""
        seg_path = SHM_BASE / f"{self._prefix}_{name}"
        with self._lock:
            if name in self._segments:
                return self._segments[name]

            fd = os.open(str(seg_path), os.O_RDONLY)
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)

            self._fds[name] = fd
            self._segments[name] = mm
            return mm

    # -- tensor helpers ------------------------------------------------------

    def read_tensor(
        self, segment_name: str
    ) -> Tuple[bytes, Tuple[int, ...], nxtensor.DType]:
        """Read and deserialize a Nexus tensor from *segment_name*."""
        mm = self.open_segment(segment_name)
        mm.seek(0)
        buf = mm.read(mm.size())
        return nxtensor.deserialize_tensor(buf, verify_checksum=True)

    def read_tensor_torch(
        self, segment_name: str, device: str = "cuda:0"
    ) -> torch.Tensor:
        """Convenience: read tensor and return as a PyTorch tensor on *device*."""
        mm = self.open_segment(segment_name)
        mm.seek(0)
        buf = mm.read(mm.size())
        return nxtensor.deserialize_to_torch(buf, device=device)

    def write_tensor(
        self,
        segment_name: str,
        tensor_data: bytes,
    ) -> None:
        """Write serialized Nexus tensor bytes into *segment_name*.

        The segment is created / resized as needed.
        """
        size = len(tensor_data)
        mm = self.create_segment(segment_name, size)
        mm.seek(0)
        mm.write(tensor_data)
        mm.flush()

    def write_tensor_torch(
        self,
        segment_name: str,
        tensor: torch.Tensor,
    ) -> None:
        """Serialize a PyTorch tensor and write it to shared memory."""
        blob = nxtensor.torch_to_nexus(tensor)
        self.write_tensor(segment_name, blob)

    # -- cleanup -------------------------------------------------------------

    def close_segment(self, name: str) -> None:
        """Close and unlink a single segment."""
        with self._lock:
            mm = self._segments.pop(name, None)
            fd = self._fds.pop(name, None)
            if mm is not None:
                mm.close()
            if fd is not None:
                os.close(fd)
            seg_path = SHM_BASE / f"{self._prefix}_{name}"
            if seg_path.exists():
                seg_path.unlink()

    def cleanup_stale(self, max_age_sec: float = SEGMENT_CLEANUP_AGE_SEC) -> int:
        """Remove stale segments older than *max_age_sec*.

        Returns the number of segments removed.
        """
        now = time.time()
        removed = 0
        prefix = f"{self._prefix}_"
        for entry in SHM_BASE.iterdir():
            if not entry.name.startswith(prefix):
                continue
            try:
                age = now - entry.stat().st_mtime
                if age > max_age_sec:
                    seg_name = entry.name[len(prefix) :]
                    self.close_segment(seg_name)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info("shm_cleanup", removed=removed)
        return removed

    def close_all(self) -> None:
        """Close every tracked segment."""
        with self._lock:
            names = list(self._segments.keys())
        for name in names:
            self.close_segment(name)

    def __del__(self) -> None:
        self.close_all()


# ---------------------------------------------------------------------------
# Model registry entry
# ---------------------------------------------------------------------------


@dataclass
class _ModelEntry:
    model_id: str
    engine: InferenceEngine
    config: Dict[str, Any]
    loaded_at: float = field(default_factory=time.time)
    request_count: int = 0


# ---------------------------------------------------------------------------
# WorkerServer -- serves inference requests from the Go router
# ---------------------------------------------------------------------------


class WorkerServer:
    """Serves inference requests received over shared memory from the Go
    router.  Can also expose a Unix-domain socket or gRPC endpoint.

    Internally manages one :class:`InferenceEngine` per registered model.
    """

    def __init__(
        self,
        worker_id: str = "worker-0",
        shm_prefix: str = "nexus",
        device: str = "cuda:0",
        max_batch_wait_ms: float = 5.0,
    ) -> None:
        self._worker_id = worker_id
        self._device = device
        self._max_batch_wait_ms = max_batch_wait_ms

        self._transport = SharedMemoryTransport(prefix=shm_prefix)
        self._models: Dict[str, _ModelEntry] = {}
        self._lock = threading.Lock()
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._uds_path: Optional[str] = None
        self._uds_socket: Optional[socket.socket] = None

        # Queue depth tracking
        self._queue_depth: int = 0
        self._queue_lock = threading.Lock()

    # -- model management ----------------------------------------------------

    def register_model(
        self,
        model_id: str,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register and load a model for serving."""
        config = config or {}
        dtype = config.get("dtype", "float16")
        quantize_cfg: Optional[QuantizationConfig] = None
        if config.get("quantize"):
            quantize_cfg = QuantizationConfig(
                method=config["quantize"].get("method", "dynamic"),
                calibration_samples=config["quantize"].get("calibration_samples", 256),
            )

        engine = InferenceEngine(
            model_path=model_path,
            dtype=dtype,
            device=self._device,
            quantize=quantize_cfg,
        )
        engine.load_model()

        # Optional warmup
        warmup_shape = config.get("warmup_shape")
        if warmup_shape:
            engine.warmup(tuple(warmup_shape))

        with self._lock:
            self._models[model_id] = _ModelEntry(
                model_id=model_id,
                engine=engine,
                config=config,
            )

        logger.info(
            "model_registered",
            model_id=model_id,
            device=self._device,
            dtype=dtype,
        )

    def unregister_model(self, model_id: str) -> None:
        with self._lock:
            entry = self._models.pop(model_id, None)
        if entry is None:
            logger.warning("model_not_found", model_id=model_id)

    # -- batch handling ------------------------------------------------------

    def handle_batch(
        self,
        model_id: str,
        input_segment: str,
        output_segment: str,
    ) -> Dict[str, Any]:
        """Execute inference for a batch received via shared memory.

        1. Reads the input tensor from *input_segment*.
        2. Runs the model.
        3. Writes the output tensor to *output_segment*.

        Returns metadata dict (latency, output shape, etc.).
        """
        with self._queue_lock:
            self._queue_depth += 1

        t0 = time.monotonic()
        try:
            with self._lock:
                entry = self._models.get(model_id)
            if entry is None:
                raise KeyError(f"Model '{model_id}' is not registered")

            # Read input from shared memory
            input_tensor = self._transport.read_tensor_torch(
                input_segment, device=self._device
            )

            # Run inference
            outputs = entry.engine.infer(input_tensor)
            entry.request_count += 1

            # Write first output to shared memory
            output_key = next(iter(outputs))
            output_tensor = outputs[output_key]
            self._transport.write_tensor_torch(output_segment, output_tensor)

            elapsed_ms = (time.monotonic() - t0) * 1000
            return {
                "status": "ok",
                "model_id": model_id,
                "latency_ms": round(elapsed_ms, 3),
                "output_shape": list(output_tensor.shape),
                "output_dtype": str(output_tensor.dtype),
            }
        except Exception as exc:
            logger.error(
                "handle_batch_error",
                model_id=model_id,
                error=str(exc),
            )
            return {
                "status": "error",
                "model_id": model_id,
                "error": str(exc),
            }
        finally:
            with self._queue_lock:
                self._queue_depth -= 1

    # -- health check --------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return worker health status for the Go router."""
        gpu_util: float = 0.0
        gpu_mem_used: int = 0
        gpu_mem_total: int = 0

        if torch.cuda.is_available():
            dev_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
            gpu_mem_used = torch.cuda.memory_allocated(dev_idx)
            gpu_mem_total = torch.cuda.get_device_properties(dev_idx).total_memory
            gpu_util = (
                round(gpu_mem_used / gpu_mem_total, 4) if gpu_mem_total > 0 else 0.0
            )

        with self._lock:
            loaded_models = {
                mid: {
                    "request_count": entry.request_count,
                    "loaded_at": entry.loaded_at,
                    "dtype": entry.config.get("dtype", "float16"),
                }
                for mid, entry in self._models.items()
            }

        return {
            "worker_id": self._worker_id,
            "status": "healthy" if self._running else "stopped",
            "device": self._device,
            "gpu_utilization": gpu_util,
            "gpu_memory_used_bytes": gpu_mem_used,
            "gpu_memory_total_bytes": gpu_mem_total,
            "loaded_models": loaded_models,
            "num_models": len(loaded_models),
            "queue_depth": self._queue_depth,
            "timestamp": time.time(),
        }

    # -- Unix domain socket server -------------------------------------------

    def start_uds_server(self, path: str = "/tmp/nexus_worker.sock") -> None:
        """Start listening on a Unix domain socket for commands from Go.

        Protocol: length-prefixed JSON messages (4-byte big-endian length
        header followed by UTF-8 JSON body).
        """
        self._uds_path = path
        if os.path.exists(path):
            os.unlink(path)

        self._uds_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._uds_socket.bind(path)
        self._uds_socket.listen(8)
        self._uds_socket.settimeout(1.0)
        self._running = True

        logger.info("uds_server_started", path=path)

        # Start background threads
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="shm-cleanup"
        )
        self._cleanup_thread.start()

        # Accept loop
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="uds-handler")
        try:
            while self._running:
                try:
                    conn, _ = self._uds_socket.accept()
                    executor.submit(self._handle_connection, conn)
                except socket.timeout:
                    continue
        except Exception:
            logger.exception("uds_accept_error")
        finally:
            executor.shutdown(wait=False)
            self._shutdown_uds()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single UDS connection (length-prefixed JSON)."""
        import json

        try:
            conn.settimeout(10.0)
            # Read 4-byte length header
            length_bytes = self._recv_exact(conn, 4)
            if not length_bytes:
                return
            msg_len = struct.unpack(">I", length_bytes)[0]
            payload = self._recv_exact(conn, msg_len)
            if not payload:
                return

            request = json.loads(payload.decode("utf-8"))
            command = request.get("command", "")

            if command == "infer":
                result = self.handle_batch(
                    model_id=request["model_id"],
                    input_segment=request["input_segment"],
                    output_segment=request["output_segment"],
                )
            elif command == "health":
                result = self.health_check()
            elif command == "register":
                self.register_model(
                    model_id=request["model_id"],
                    model_path=request["model_path"],
                    config=request.get("config"),
                )
                result = {"status": "ok", "model_id": request["model_id"]}
            elif command == "unregister":
                self.unregister_model(request["model_id"])
                result = {"status": "ok"}
            elif command == "shutdown":
                result = {"status": "shutting_down"}
                self._running = False
            else:
                result = {"status": "error", "error": f"Unknown command: {command}"}

            # Send response
            resp_bytes = json.dumps(result).encode("utf-8")
            conn.sendall(struct.pack(">I", len(resp_bytes)) + resp_bytes)
        except Exception:
            logger.exception("handle_connection_error")
        finally:
            conn.close()

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly *n* bytes from *sock*."""
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _shutdown_uds(self) -> None:
        if self._uds_socket is not None:
            self._uds_socket.close()
            self._uds_socket = None
        if self._uds_path and os.path.exists(self._uds_path):
            os.unlink(self._uds_path)

    # -- heartbeat -----------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Periodically report health back to the Go router."""
        while self._running:
            try:
                health = self.health_check()
                logger.debug("heartbeat", **health)
            except Exception:
                logger.exception("heartbeat_error")
            time.sleep(HEARTBEAT_INTERVAL_SEC)

    # -- shm cleanup ---------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Periodically clean up stale shared memory segments."""
        while self._running:
            try:
                self._transport.cleanup_stale()
            except Exception:
                logger.exception("cleanup_error")
            time.sleep(60.0)

    # -- graceful shutdown ---------------------------------------------------

    def shutdown(self) -> None:
        """Initiate graceful shutdown."""
        logger.info("shutdown_initiated", worker_id=self._worker_id)
        self._running = False

        # Wait for background threads
        for t in (self._heartbeat_thread, self._cleanup_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)

        # Close shared memory
        self._transport.close_all()

        # Close UDS
        self._shutdown_uds()

        logger.info("shutdown_complete", worker_id=self._worker_id)


# ---------------------------------------------------------------------------
# WorkerPool -- spawn N worker processes for parallel model serving
# ---------------------------------------------------------------------------


def _worker_process_entry(
    worker_id: str,
    device: str,
    uds_path: str,
    shm_prefix: str,
    model_configs: List[Dict[str, Any]],
    ready_event: "mp.Event",  # type: ignore[valid-type]
) -> None:
    """Entry point for a child worker process."""
    # Install signal handlers for graceful shutdown
    server = WorkerServer(
        worker_id=worker_id,
        shm_prefix=shm_prefix,
        device=device,
    )

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info(
            "signal_received",
            signal=signal.Signals(signum).name,
            worker_id=worker_id,
        )
        server.shutdown()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Pre-load models
    for cfg in model_configs:
        server.register_model(
            model_id=cfg["model_id"],
            model_path=cfg["model_path"],
            config=cfg.get("config"),
        )

    ready_event.set()  # type: ignore[attr-defined]
    server.start_uds_server(path=uds_path)


class WorkerPool:
    """Manages a pool of worker processes, each pinned to a CUDA device.

    Example
    -------
    ::

        pool = WorkerPool(num_workers=4)
        pool.start(model_configs=[{"model_id": "llama", "model_path": "/models/llama.pt"}])
        # ... serve requests ...
        pool.shutdown()
    """

    def __init__(
        self,
        num_workers: int = 1,
        devices: Optional[List[str]] = None,
        shm_prefix: str = "nexus",
        base_uds_path: str = "/tmp/nexus_worker",
    ) -> None:
        self._num_workers = num_workers
        self._devices = devices or [
            f"cuda:{i % torch.cuda.device_count()}"
            if torch.cuda.is_available()
            else "cpu"
            for i in range(num_workers)
        ]
        self._shm_prefix = shm_prefix
        self._base_uds_path = base_uds_path
        self._processes: List[mp.process.BaseProcess] = []
        self._ready_events: List["mp.Event"] = []  # type: ignore[valid-type]

    def start(
        self,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 120.0,
    ) -> None:
        """Spawn worker processes and wait until they are ready."""
        model_configs = model_configs or []
        ctx = mp.get_context("spawn")

        for idx in range(self._num_workers):
            worker_id = f"worker-{idx}"
            device = self._devices[idx]
            uds_path = f"{self._base_uds_path}_{idx}.sock"
            ready = ctx.Event()

            proc = ctx.Process(
                target=_worker_process_entry,
                args=(
                    worker_id,
                    device,
                    uds_path,
                    f"{self._shm_prefix}_{idx}",
                    model_configs,
                    ready,
                ),
                name=worker_id,
                daemon=False,
            )
            proc.start()
            self._processes.append(proc)
            self._ready_events.append(ready)
            logger.info(
                "worker_spawned",
                worker_id=worker_id,
                pid=proc.pid,
                device=device,
            )

        # Wait for all workers to signal ready
        deadline = time.monotonic() + timeout
        for idx, evt in enumerate(self._ready_events):
            remaining = max(0.1, deadline - time.monotonic())
            if not evt.wait(timeout=remaining):  # type: ignore[attr-defined]
                logger.error(
                    "worker_start_timeout",
                    worker_id=f"worker-{idx}",
                )

        logger.info(
            "worker_pool_ready",
            num_workers=self._num_workers,
        )

    def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully terminate all worker processes."""
        logger.info("worker_pool_shutdown")
        for proc in self._processes:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGTERM)  # type: ignore[arg-type]

        deadline = time.monotonic() + timeout
        for proc in self._processes:
            remaining = max(0.1, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                logger.warning(
                    "worker_force_kill",
                    pid=proc.pid,
                )
                proc.kill()

        self._processes.clear()
        self._ready_events.clear()
        logger.info("worker_pool_stopped")

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def pids(self) -> List[Optional[int]]:
        return [p.pid for p in self._processes]

    def is_healthy(self) -> bool:
        """Return True if all worker processes are alive."""
        return all(p.is_alive() for p in self._processes)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone entry point: ``python -m nexus.runtime.worker``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Nexus inference worker")
    parser.add_argument("--worker-id", default="worker-0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--uds-path", default="/tmp/nexus_worker.sock")
    parser.add_argument("--shm-prefix", default="nexus")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="JSON file with model configs [{model_id, model_path, config}]",
    )
    args = parser.parse_args()

    server = WorkerServer(
        worker_id=args.worker_id,
        shm_prefix=args.shm_prefix,
        device=args.device,
    )

    # Graceful shutdown on SIGTERM / SIGINT
    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info(
            "signal_received",
            signal=signal.Signals(signum).name,
        )
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Pre-load models from config file
    if args.models:
        with open(args.models) as f:
            model_configs = json.load(f)
        for cfg in model_configs:
            server.register_model(
                model_id=cfg["model_id"],
                model_path=cfg["model_path"],
                config=cfg.get("config"),
            )

    logger.info(
        "worker_starting",
        worker_id=args.worker_id,
        device=args.device,
    )
    server.start_uds_server(path=args.uds_path)


if __name__ == "__main__":
    main()
