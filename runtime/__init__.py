"""Nexus Runtime -- Python inference runtime for the Nexus distributed AI engine.

Submodules
----------
tensor
    Custom binary tensor serialization / deserialization (wire format).
cuda_pool
    Pre-allocated CUDA and pinned-memory pool managers.
inference
    Mixed-precision inference engine with INT8 quantization and tensor
    parallelism (model sharding) across multiple GPUs.
worker
    Worker process that communicates with the Go router over shared memory
    and Unix domain sockets.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# tensor.py exports
# ---------------------------------------------------------------------------
from .tensor import (
    ALIGNMENT,
    DTYPE_SIZES,
    HEADER_SIZE,
    HEADER_VERSION,
    HAS_NUMPY,
    HAS_TORCH,
    MAGIC,
    NUMPY_DTYPE_MAP,
    TORCH_DTYPE_MAP,
    DType,
    MMapTensor,
    TensorFlags,
    deserialize_header,
    deserialize_tensor,
    deserialize_to_numpy,
    deserialize_to_torch,
    numpy_to_nexus,
    serialize_tensor,
    torch_to_nexus,
)

# ---------------------------------------------------------------------------
# cuda_pool.py exports
# ---------------------------------------------------------------------------
from .cuda_pool import (
    Block,
    CUDAMemoryPool,
    MemoryBlock,
    PinnedMemoryPool,
    PoolStats,
)

# ---------------------------------------------------------------------------
# inference.py exports
# ---------------------------------------------------------------------------
from .inference import (
    InferenceEngine,
    ModelShard,
    QuantizationConfig,
)

# ---------------------------------------------------------------------------
# worker.py exports
# ---------------------------------------------------------------------------
from .worker import (
    SharedMemoryTransport,
    WorkerPool,
    WorkerServer,
)

__all__ = [
    # -- tensor --
    "ALIGNMENT",
    "DTYPE_SIZES",
    "DType",
    "HAS_NUMPY",
    "HAS_TORCH",
    "HEADER_SIZE",
    "HEADER_VERSION",
    "MAGIC",
    "MMapTensor",
    "NUMPY_DTYPE_MAP",
    "TORCH_DTYPE_MAP",
    "TensorFlags",
    "deserialize_header",
    "deserialize_tensor",
    "deserialize_to_numpy",
    "deserialize_to_torch",
    "numpy_to_nexus",
    "serialize_tensor",
    "torch_to_nexus",
    # -- cuda_pool --
    "Block",
    "CUDAMemoryPool",
    "MemoryBlock",
    "PinnedMemoryPool",
    "PoolStats",
    # -- inference --
    "InferenceEngine",
    "ModelShard",
    "QuantizationConfig",
    # -- worker --
    "SharedMemoryTransport",
    "WorkerPool",
    "WorkerServer",
]

__version__ = "0.1.0"
