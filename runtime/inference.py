"""Mixed-precision inference engine for Nexus distributed inference.

Provides model loading with FP16/BF16/INT8 support, batched inference,
tensor-parallel model sharding across GPUs, and latency benchmarking.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Configuration for post-training quantization.

    Attributes
    ----------
    method:
        ``'dynamic'`` for ``torch.quantization.quantize_dynamic`` or
        ``'static'`` for calibration-based static quantization.
    dtype:
        Target quantized dtype (``torch.qint8`` by default).
    calibration_samples:
        Number of calibration samples for static quantization.
    layers_to_quantize:
        Module types eligible for quantization.  Defaults to
        ``[nn.Linear, nn.Conv2d]``.
    """

    method: str = "dynamic"  # 'dynamic' | 'static'
    dtype: Any = None  # torch.qint8 default set in __post_init__
    calibration_samples: int = 256
    layers_to_quantize: Optional[List[type]] = None

    def __post_init__(self) -> None:
        if self.dtype is None:
            self.dtype = torch.qint8
        if self.layers_to_quantize is None:
            self.layers_to_quantize = [nn.Linear]


# ---------------------------------------------------------------------------
# ModelShard -- tensor parallelism across multiple GPUs
# ---------------------------------------------------------------------------

class ModelShard:
    """Splits a model across multiple CUDA devices for tensor parallelism.

    Each shard holds a contiguous slice of the model's top-level children
    (layers) and executes them in pipeline order.  Inter-device transfers are
    handled automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        num_shards: int,
        devices: Optional[List[str]] = None,
    ) -> None:
        self._num_shards = num_shards
        self._devices = devices or [
            f"cuda:{i}" for i in range(num_shards)
        ]
        if len(self._devices) < num_shards:
            raise ValueError(
                f"Need at least {num_shards} devices, got {len(self._devices)}"
            )

        self._shards: List[nn.Sequential] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._split_model(model)

    # -- splitting -----------------------------------------------------------

    def _split_model(self, model: nn.Module) -> None:
        """Distribute top-level children across devices."""
        children = list(model.children())
        if not children:
            # Treat the model itself as a single shard
            children = [model]

        n = len(children)
        chunk_size = max(1, (n + self._num_shards - 1) // self._num_shards)

        for shard_idx in range(self._num_shards):
            start = shard_idx * chunk_size
            end = min(start + chunk_size, n)
            layers = children[start:end]
            if not layers:
                break

            device = self._devices[shard_idx]
            shard = nn.Sequential(*layers).to(device)
            shard.eval()
            self._shards.append(shard)

            # Register forward hooks to move activations between devices
            if shard_idx < self._num_shards - 1:
                next_device = self._devices[min(shard_idx + 1, len(self._devices) - 1)]
                hook = shard.register_forward_hook(
                    self._make_transfer_hook(next_device)
                )
                self._hooks.append(hook)

        logger.info(
            "Split model into %d shards across %s",
            len(self._shards),
            [str(d) for d in self._devices[: len(self._shards)]],
        )

    @staticmethod
    def _make_transfer_hook(target_device: str):
        """Create a forward-hook closure that moves output to *target_device*."""

        def _hook(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> Any:
            if isinstance(output, torch.Tensor):
                return output.to(target_device, non_blocking=True)
            if isinstance(output, tuple):
                return tuple(
                    o.to(target_device, non_blocking=True)
                    if isinstance(o, torch.Tensor)
                    else o
                    for o in output
                )
            return output

        return _hook

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pipeline-execute across shards with automatic device transfers."""
        # Move input to the first shard's device
        x = x.to(self._devices[0], non_blocking=True)
        for shard in self._shards:
            x = shard(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # -- cleanup -------------------------------------------------------------

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def devices(self) -> List[str]:
        return self._devices[: len(self._shards)]


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Mixed-precision inference engine with optional INT8 quantization.

    Parameters
    ----------
    model_path:
        Path to a saved PyTorch model (``state_dict`` or ``torch.save``
        full model).
    dtype:
        Compute dtype -- ``'float16'``, ``'bfloat16'``, ``'float32'``.
    device:
        Target device string (e.g. ``'cuda:0'``).
    quantize:
        Optional :class:`QuantizationConfig` for INT8 quantization.
    """

    _DTYPE_MAP = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    def __init__(
        self,
        model_path: Union[str, Path],
        dtype: str = "float16",
        device: str = "cuda:0",
        quantize: Optional[QuantizationConfig] = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._dtype_str = dtype
        self._torch_dtype = self._DTYPE_MAP.get(dtype)
        if self._torch_dtype is None:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. "
                f"Choose from: {list(self._DTYPE_MAP.keys())}"
            )
        self._device = torch.device(device)
        self._quantize_config = quantize
        self._model: Optional[nn.Module] = None
        self._model_shard: Optional[ModelShard] = None
        self._is_loaded = False
        self._warmup_done = False

    # -- model loading -------------------------------------------------------

    def load_model(self) -> None:
        """Load the model from disk, apply dtype conversion, and optionally
        quantize to INT8."""
        logger.info(
            "Loading model from %s (dtype=%s, device=%s)",
            self._model_path,
            self._dtype_str,
            self._device,
        )

        checkpoint = torch.load(
            str(self._model_path),
            map_location="cpu",
            weights_only=False,
        )

        # Support both full-model saves and state_dict saves
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        else:
            raise ValueError(
                "Checkpoint must be a nn.Module or a dict with a 'model' key. "
                f"Got {type(checkpoint).__name__} with keys: "
                f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
            )

        # Cast to target dtype
        if self._torch_dtype != torch.float32:
            model = model.to(self._torch_dtype)
            logger.info("Converted model to %s", self._torch_dtype)

        # Optional INT8 quantization
        if self._quantize_config is not None:
            model = self._apply_quantization(model)

        # Move to device and set eval mode
        model = model.to(self._device)
        model.eval()

        self._model = model
        self._is_loaded = True
        logger.info("Model loaded successfully on %s", self._device)

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization per the engine's QuantizationConfig."""
        cfg = self._quantize_config
        assert cfg is not None

        if cfg.method == "dynamic":
            logger.info(
                "Applying dynamic quantization (dtype=%s) to %s",
                cfg.dtype,
                [t.__name__ for t in (cfg.layers_to_quantize or [])],
            )
            model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec=set(cfg.layers_to_quantize or []),
                dtype=cfg.dtype,
            )
        elif cfg.method == "static":
            logger.info(
                "Applying static quantization with %d calibration samples",
                cfg.calibration_samples,
            )
            model.qconfig = torch.quantization.get_default_qconfig("x86")  # type: ignore[assignment]
            torch.quantization.prepare(model, inplace=True)
            # NOTE: In production the caller should run calibration data
            # through the model here.  We prepare but skip actual calibration
            # when no data is provided.
            torch.quantization.convert(model, inplace=True)
        else:
            raise ValueError(f"Unknown quantization method: {cfg.method}")

        return model

    # -- warmup --------------------------------------------------------------

    def warmup(self, input_shape: Tuple[int, ...], num_runs: int = 5) -> None:
        """Pre-fill CUDA caches by running dummy inputs."""
        self._ensure_loaded()
        assert self._model is not None

        logger.info("Warming up with %d dummy forward passes", num_runs)
        dummy = torch.randn(input_shape, dtype=self._torch_dtype, device=self._device)

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=(self._torch_dtype == torch.float16),
        ):
            for _ in range(num_runs):
                self._model(dummy)
        torch.cuda.synchronize(self._device)
        self._warmup_done = True
        logger.info("Warmup complete")

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        input_tensor: torch.Tensor,
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a single forward pass with mixed-precision autocast.

        Parameters
        ----------
        input_tensor:
            Input tensor (will be moved to the engine device if needed).
        output_names:
            Optional names for output tensors.  If the model returns a dict
            these keys are used to filter; if it returns a single tensor the
            first name is used.

        Returns
        -------
        dict
            Mapping of name -> output tensor.
        """
        self._ensure_loaded()
        assert self._model is not None

        if input_tensor.device != self._device:
            input_tensor = input_tensor.to(self._device, non_blocking=True)

        use_autocast = self._torch_dtype in (torch.float16, torch.bfloat16)

        with torch.cuda.amp.autocast(
            enabled=use_autocast,
            dtype=self._torch_dtype if use_autocast else None,
        ):
            raw_output = self._model(input_tensor)

        return self._format_output(raw_output, output_names)

    @torch.no_grad()
    def infer_batch(
        self,
        batch_inputs: List[torch.Tensor],
        output_names: Optional[List[str]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run batched inference -- stacks inputs, runs one forward pass,
        then unbatches the results."""
        self._ensure_loaded()
        assert self._model is not None

        # Stack individual inputs into one batch
        batched = torch.stack(batch_inputs, dim=0).to(
            self._device, non_blocking=True
        )
        batch_size = batched.shape[0]

        use_autocast = self._torch_dtype in (torch.float16, torch.bfloat16)

        with torch.cuda.amp.autocast(
            enabled=use_autocast,
            dtype=self._torch_dtype if use_autocast else None,
        ):
            raw_output = self._model(batched)

        # Unbatch
        results: List[Dict[str, torch.Tensor]] = []
        if isinstance(raw_output, torch.Tensor):
            for i in range(batch_size):
                name = (
                    output_names[0]
                    if output_names
                    else "output"
                )
                results.append({name: raw_output[i]})
        elif isinstance(raw_output, dict):
            for i in range(batch_size):
                entry: Dict[str, torch.Tensor] = {}
                for k, v in raw_output.items():
                    if output_names is None or k in output_names:
                        entry[k] = v[i] if isinstance(v, torch.Tensor) else v
                results.append(entry)
        elif isinstance(raw_output, (tuple, list)):
            for i in range(batch_size):
                entry = {}
                for j, v in enumerate(raw_output):
                    name = (
                        output_names[j]
                        if output_names and j < len(output_names)
                        else f"output_{j}"
                    )
                    entry[name] = v[i] if isinstance(v, torch.Tensor) else v
                results.append(entry)
        else:
            raise TypeError(
                f"Unexpected model output type: {type(raw_output).__name__}"
            )

        return results

    # -- tensor parallelism --------------------------------------------------

    def split_model(
        self,
        num_shards: int,
        devices: Optional[List[str]] = None,
    ) -> ModelShard:
        """Split the loaded model across multiple GPUs.

        The resulting :class:`ModelShard` replaces the internal model
        reference so that subsequent :meth:`infer` calls use the sharded
        pipeline.
        """
        self._ensure_loaded()
        assert self._model is not None

        shard = ModelShard(
            model=self._model,
            num_shards=num_shards,
            devices=devices,
        )
        self._model_shard = shard
        # Replace model callable so infer() routes through the shard
        self._model = _ShardWrapper(shard)
        logger.info("Model sharded across %d devices", shard.num_shards)
        return shard

    # -- benchmarking --------------------------------------------------------

    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """Run *num_iterations* forward passes and report latency statistics.

        Returns
        -------
        dict
            Keys: ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
            ``min_ms``, ``max_ms``, ``throughput_samples_per_sec``.
        """
        self._ensure_loaded()
        assert self._model is not None

        dummy = torch.randn(
            input_shape, dtype=self._torch_dtype, device=self._device
        )

        use_autocast = self._torch_dtype in (torch.float16, torch.bfloat16)

        # Warmup
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=use_autocast,
            dtype=self._torch_dtype if use_autocast else None,
        ):
            for _ in range(warmup_iterations):
                self._model(dummy)
        torch.cuda.synchronize(self._device)

        # Timed runs
        latencies: List[float] = []
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=use_autocast,
            dtype=self._torch_dtype if use_autocast else None,
        ):
            for _ in range(num_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                self._model(dummy)
                end_event.record()

                torch.cuda.synchronize(self._device)
                latencies.append(start_event.elapsed_time(end_event))  # ms

        latencies.sort()
        n = len(latencies)
        mean_ms = statistics.mean(latencies)
        batch_size = input_shape[0] if len(input_shape) > 1 else 1

        return {
            "mean_ms": round(mean_ms, 3),
            "p50_ms": round(latencies[n // 2], 3),
            "p95_ms": round(latencies[int(n * 0.95)], 3),
            "p99_ms": round(latencies[int(n * 0.99)], 3),
            "min_ms": round(latencies[0], 3),
            "max_ms": round(latencies[-1], 3),
            "throughput_samples_per_sec": round(
                (batch_size * 1000.0) / mean_ms, 2
            ),
            "num_iterations": num_iterations,
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _format_output(
        raw: Any,
        names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(raw, dict):
            if names:
                return {k: raw[k] for k in names if k in raw}
            return raw
        if isinstance(raw, torch.Tensor):
            name = names[0] if names else "output"
            return {name: raw}
        if isinstance(raw, (tuple, list)):
            result: Dict[str, torch.Tensor] = {}
            for i, v in enumerate(raw):
                name = names[i] if names and i < len(names) else f"output_{i}"
                result[name] = v
            return result
        raise TypeError(f"Unexpected model output type: {type(raw).__name__}")

    def _ensure_loaded(self) -> None:
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                "Model not loaded -- call load_model() first"
            )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._torch_dtype  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"InferenceEngine(model={self._model_path.name}, "
            f"dtype={self._dtype_str}, device={self._device}, "
            f"loaded={self._is_loaded})"
        )


# ---------------------------------------------------------------------------
# Internal helper to wrap ModelShard as an nn.Module-like callable
# ---------------------------------------------------------------------------

class _ShardWrapper(nn.Module):
    """Thin wrapper so that :class:`ModelShard` can be called like a
    regular ``nn.Module`` inside :class:`InferenceEngine`."""

    def __init__(self, shard: ModelShard) -> None:
        super().__init__()
        self._shard = shard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shard.forward(x)
