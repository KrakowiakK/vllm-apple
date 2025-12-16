# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Model Runner for vLLM V1 engine.

This model runner handles model inference on Apple Silicon (MPS).
It implements the model runner interface WITHOUT inheriting from
GPUModelRunner to avoid Triton dependencies.

Key design decisions:
1. No inheritance from GPUModelRunner - avoids all Triton/CUDA dependencies
2. Uses PyTorch SDPA for attention instead of Flash Attention
3. Implements paged KV cache management using native PyTorch operations
4. Supports both prefill and decode phases
5. Based on patterns from HPUModelRunner and TPUModelRunner

Reference: vllm/v1/worker/gpu_model_runner.py (interface)
"""

import gc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sampling_params import SamplingType
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.utils import bind_kv_cache

# Engine mode imports (v2.0 Metal engine)
from vllm_apple.engine import (
    is_engine_mode_enabled,
    is_engine_prefill_enabled,
    is_strict_mode,
)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class AppleModelInput:
    """Input data for a single model forward pass on Apple Silicon."""

    input_ids: torch.Tensor  # [num_tokens]
    positions: torch.Tensor  # [num_tokens]
    attn_metadata: Any  # AppleAttentionMetadata
    num_scheduled_tokens: int
    num_reqs: int
    req_ids: list[str]
    seq_lens: list[int]
    sampling_metadata: Optional[SamplingMetadata] = None


@dataclass
class CachedRequestState:
    """Cached state for a request being processed."""

    req_id: str
    prompt_token_ids: list[int] | None
    num_computed_tokens: int
    output_token_ids: list[int]
    sampling_params: Any  # SamplingParams
    pooling_params: Any  # PoolingParams
    block_ids: tuple[list[int], ...]
    generator: Optional[torch.Generator] = None


class AppleModelRunner:
    """Model runner for Apple Silicon (MPS).

    This runner manages model inference without inheriting from GPUModelRunner
    to avoid Triton dependencies. It implements:
    - Model loading with get_model()
    - KV cache management using paged attention
    - Forward pass execution with attention metadata
    - Token sampling using the Sampler

    Architecture:
    - Uses PyTorch MPS backend for acceleration
    - Implements paged KV cache for memory efficiency
    - Supports both prefill (prompt) and decode phases
    - Handles batch processing with variable sequence lengths

    Based on patterns from HPUModelRunner and TPUModelRunner.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_driver_worker: bool = False,
    ):
        """Initialize the Apple model runner.

        Args:
            vllm_config: Complete vLLM configuration
            device: Target device (should be MPS)
            is_driver_worker: Whether this is the driver worker
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.lora_config = vllm_config.lora_config
        self.speculative_config = vllm_config.speculative_config

        self.device = device
        self.is_driver_worker = is_driver_worker

        # Model will be loaded in load_model()
        self.model: Optional[nn.Module] = None

        # Engine mode (v2.0 Metal engine)
        self._use_engine_mode = is_engine_mode_enabled()
        self._engine_runner = None  # Initialized in load_model if engine mode
        self._engine_kv_cache = None  # Engine KV cache for prefill→decode sync
        if self._use_engine_mode:
            logger.info("Engine mode enabled (VLLM_APPLE_USE_ENGINE=1)")
            import vllm_apple.engine.runner
            logger.info(f"DEBUG: vllm_apple.engine.runner imported from: {vllm_apple.engine.runner.__file__}")

        # KV cache configuration
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.kv_cache_config: Optional[KVCacheConfig] = None
        self.block_size = self.cache_config.block_size

        # Attention backend - will be set during model loading
        self.attn_backend = None

        # Sampler - initialized after model loading
        self.sampler: Optional[Sampler] = None

        # Configuration derived values
        self.dtype = self.model_config.dtype
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        self.vocab_size = self.model_config.get_vocab_size()

        # Request state tracking
        # Maps req_id -> CachedRequestState
        self.requests: dict[str, CachedRequestState] = {}

        # Pooling model support
        self.is_pooling_model = self.model_config.runner_type == "pooling"

        # Multimodal support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.is_multimodal = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )

        # Pipeline parallelism
        self.is_last_pp_rank = get_pp_group().is_last_rank

        logger.info(
            "AppleModelRunner initialized: dtype=%s, max_model_len=%d, "
            "max_num_reqs=%d, block_size=%d, vocab_size=%d",
            self.dtype,
            self.max_model_len,
            self.max_num_reqs,
            self.block_size,
            self.vocab_size,
        )

    def _get_input_device(self, is_decode_only: bool) -> torch.device:
        """Get device for input tensor preparation based on step type.

        This implements step-aware device selection per METAL_PLAN.md:
        - Prefill steps: MPS (PyTorch model needs MPS inputs)
        - Decode-only steps in engine mode: CPU (engine needs CPU inputs)
        - Prefill steps in engine mode (opt-in): CPU (engine needs CPU inputs)
        - Decode steps without engine mode: MPS (PyTorch path)

        By preparing inputs on the correct device from the start, we avoid
        costly MPS→CPU conversions at the engine boundary during decode.

        Args:
            is_decode_only: True if this step has no prefill requests

        Returns:
            Device for input tensor preparation
        """
        if self._use_engine_mode and self._engine_runner is not None:
            if is_decode_only or is_engine_prefill_enabled():
                # Engine decode (and optionally engine prefill): prepare inputs on CPU directly.
                # This avoids forbidden MPS→CPU conversions at the engine boundary.
                return torch.device("cpu")

        # Prefill or non-engine path: PyTorch model needs MPS inputs.
        return self.device

    def _is_decode_only_step(self, scheduler_output: "SchedulerOutput") -> bool:
        """Check if this step contains only decode requests (no prefill).

        A decode-only step has:
        - No new requests (scheduled_new_reqs is empty)
        - All requests are continuing/cached (have prior context)

        Args:
            scheduler_output: Scheduler output to check

        Returns:
            True if all requests are decode, False if any prefill
        """
        # If there are any new requests, this is not decode-only
        if scheduler_output.scheduled_new_reqs:
            return False
        # No new requests = all decode
        return True

    @property
    def _input_device(self) -> torch.device:
        """Legacy property for backwards compatibility.

        DEPRECATED: Use _get_input_device(is_decode_only) instead.
        This property always returns MPS for safety (assumes prefill).
        """
        # Default to MPS (safe for prefill)
        return self.device

    def load_model(self) -> None:
        """Load the model onto MPS device.

        This method:
        1. Loads the model using vLLM's model loader
        2. Initializes the sampler
        3. Sets up the attention backend
        """
        logger.info("Loading model %s...", self.model_config.model)

        # Load model using vLLM's standard model loader
        # This handles HuggingFace model loading, quantization, etc.
        self.model = get_model(vllm_config=self.vllm_config)

        # Move model to MPS device
        self.model = self.model.to(self.device)

        # Initialize sampler for token generation
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        # Set attention backend from platform
        from vllm_apple.platform import ApplePlatform

        attn_backend_cls = ApplePlatform.get_attn_backend_cls(
            selected_backend=None,
            head_size=self.model_config.get_head_size(),
            dtype=self.dtype,
            kv_cache_dtype=self.cache_config.cache_dtype if self.cache_config else "auto",
            block_size=self.block_size,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            is_attention_free=False,
        )

        # Import the backend class
        if attn_backend_cls:
            module_name, class_name = attn_backend_cls.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            self.attn_backend = getattr(module, class_name)
        else:
            from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionBackend
            self.attn_backend = MetalAttentionBackend

        # Determine KV cache device from backend
        if hasattr(self.attn_backend, 'get_kv_cache_device'):
            self.kv_cache_device = self.attn_backend.get_kv_cache_device()
        else:
            self.kv_cache_device = str(self.device)

        logger.info("Model loaded successfully on %s (KV cache on %s)", self.device, self.kv_cache_device)

        # Initialize engine runner if engine mode is enabled
        if self._use_engine_mode:
            self._try_init_engine_runner()

    def _try_init_engine_runner(self) -> None:
        """Try to initialize engine runner with graceful fallback.

        If engine initialization fails:
        - In strict mode: re-raise the error (no silent fallback)
        - In non-strict mode: log warning and disable engine mode
        """
        try:
            self._init_engine_runner()
        except Exception as e:
            if is_strict_mode():
                # Strict mode: fail fast, no silent fallback
                logger.error(
                    "Engine initialization failed in strict mode: %s\n"
                    "Strict mode (VLLM_METAL_STRICT_NO_MPS=1) does not allow "
                    "silent fallback to PyTorch/MPS.", e
                )
                raise
            else:
                # Non-strict mode: graceful fallback to PyTorch
                logger.warning(
                    "Engine initialization failed, falling back to PyTorch/MPS: %s\n"
                    "To require engine mode, set VLLM_METAL_STRICT_NO_MPS=1", e
                )
                self._use_engine_mode = False
                self._engine_runner = None
                self._engine_kv_cache = None

    def _init_engine_runner(self) -> None:
        """Initialize the engine runner for v2.0 Metal engine mode.

        This sets up:
        1. MetalEngineContext with device and pipelines
        2. EngineKVCache wrapping the allocated KV cache
        3. EngineRunner with model weights

        Must be called after load_model() and initialize_kv_cache().
        """
        from vllm_apple.engine import (
            create_engine_runner,
            MetalEngineContext,
            EngineKVCache,
            ModelDescriptor,
            KVCacheDescriptor,
        )

        logger.info("Initializing engine runner...")

        # Create engine context
        context = MetalEngineContext()

        # Create model descriptor from config
        # Note: Some methods need parallel_config for tensor parallelism
        parallel_config = self.vllm_config.parallel_config
        hf_config = self.model_config.hf_config
        hidden_size = self.model_config.get_hidden_size()

        # Get intermediate_size from hf_config (different names per architecture)
        intermediate_size = getattr(hf_config, 'intermediate_size', None)
        if intermediate_size is None:
            # GPT-2 uses n_inner (or 4 * n_embd if not set)
            intermediate_size = getattr(hf_config, 'n_inner', None)
            if intermediate_size is None:
                intermediate_size = 4 * hidden_size

        # Detect architecture from hf_config for engine validation
        architecture = ModelDescriptor._detect_architecture(hf_config)
        logger.info(f"Detected model architecture: {architecture}")

        model_desc = ModelDescriptor(
            hidden_size=hidden_size,
            num_attention_heads=self.model_config.get_num_attention_heads(
                parallel_config
            ),
            num_kv_heads=self.model_config.get_num_kv_heads(parallel_config),
            num_layers=self.model_config.get_num_layers(parallel_config),
            head_size=self.model_config.get_head_size(),
            intermediate_size=intermediate_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_model_len,
            rope_theta=getattr(hf_config, 'rope_theta', 10000.0),
            architecture=architecture,  # Will raise ValueError if unsupported
        )

        # Engine KV cache will be initialized in initialize_kv_cache
        # For now, just store the context and model_desc
        self._engine_context = context
        self._engine_model_desc = model_desc

        logger.info(
            "Engine runner initialized: %d layers, hidden=%d, heads=%d/%d",
            model_desc.num_layers,
            model_desc.hidden_size,
            model_desc.num_attention_heads,
            model_desc.num_kv_heads,
        )

    def get_model(self) -> nn.Module:
        """Return the loaded model.

        Returns:
            The loaded PyTorch model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Return supported tasks for this model.

        Returns:
            Tuple of supported tasks (e.g., 'generate', 'embed')
        """
        if self.is_pooling_model:
            return (PoolingTask,)
        return (GenerationTask,)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get KV cache specifications for each attention layer.

        This method inspects the model's attention layers and generates
        a KVCacheSpec for each layer describing its cache requirements.

        Returns:
            Dict mapping layer names to their KV cache specifications
        """
        kv_cache_spec: dict[str, KVCacheSpec] = {}

        # Get all attention layers from the model
        from typing import Any, cast

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)

        for layer_name, attn_module in attn_layers.items():
            # Skip layers that share KV cache with another layer
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                continue

            # Get KV cache spec from the attention module
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        logger.info(
            "Generated KV cache specs for %d attention layers", len(kv_cache_spec)
        )

        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate KV cache tensors on MPS device.

        This method:
        1. Gets KV cache specifications from attention layers
        2. Allocates key and value cache tensors
        3. Binds the caches to the model's attention layers

        Args:
            kv_cache_config: Configuration specifying cache size and layout
        """
        self.kv_cache_config = kv_cache_config
        num_blocks = kv_cache_config.num_blocks
        kv_cache_spec = self.get_kv_cache_spec()

        self.kv_caches = []

        logger.info("Allocating KV cache: %d blocks for %d layers", num_blocks, len(kv_cache_spec))

        # In engine mode with engine prefill enabled, the engine-owned MTLBuffer KV cache
        # is the single source of truth. vLLM KV tensors are stubs only (no full duplication).
        use_stub_kv_cache = self._use_engine_mode and is_engine_prefill_enabled()

        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                # Allocate combined key and value cache as single tensor
                # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
                # This allows kv_cache.unbind(0) -> (key_cache, value_cache)
                cache_shape = (
                    2,  # K and V stacked
                    0 if use_stub_kv_cache else num_blocks,
                    layer_spec.block_size,
                    layer_spec.num_kv_heads,
                    layer_spec.head_size,
                )

                # Calculate cache size in GB for logging
                dtype_size = get_dtype_size(layer_spec.dtype)
                cache_size_bytes = np.prod(cache_shape) * dtype_size
                cache_size_gb = cache_size_bytes / (1024**3)

                logger.debug(
                    "Layer %s: shape=%s, size=%.2f GB",
                    layer_name,
                    cache_shape,
                    cache_size_gb,
                )

                # Create single stacked tensor for K and V. In engine mode, allocate
                # stubs on CPU; in non-engine mode use the backend-selected device.
                kv_cache = torch.zeros(
                    cache_shape,
                    dtype=layer_spec.dtype,
                    device="cpu" if use_stub_kv_cache else self.kv_cache_device,
                )

                self.kv_caches.append(kv_cache)
            else:
                # Handle other attention types (sliding window, etc.)
                # For now, use full attention spec
                logger.warning(
                    "Layer %s uses non-standard attention spec: %s",
                    layer_name,
                    type(layer_spec),
                )
                # Allocate with default parameters
                cache_shape = (
                    2,  # K and V stacked
                    0 if use_stub_kv_cache else num_blocks,
                    self.block_size,
                    layer_spec.num_kv_heads if hasattr(layer_spec, 'num_kv_heads') else 8,
                    layer_spec.head_size if hasattr(layer_spec, 'head_size') else 128,
                )
                kv_cache = torch.zeros(
                    cache_shape,
                    dtype=self.dtype,
                    device="cpu" if use_stub_kv_cache else self.kv_cache_device,
                )
                self.kv_caches.append(kv_cache)

        if not use_stub_kv_cache:
            # Bind KV caches to model's attention layers (PyTorch execution path).
            kv_caches_dict = {}
            for idx, (layer_name, _) in enumerate(kv_cache_spec.items()):
                kv_caches_dict[layer_name] = self.kv_caches[idx]

            # Use the bind_kv_cache utility to connect caches to the forward context
            runner_kv_caches: list[torch.Tensor] = []
            bind_kv_cache(
                kv_caches_dict,
                self.vllm_config.compilation_config.static_forward_context,
                runner_kv_caches,
            )

        # Calculate total cache size (for logging).
        if use_stub_kv_cache:
            # Engine KV layout: [num_blocks, num_kv_heads, block_size, head_size] for K and V.
            first_spec = next(iter(kv_cache_spec.values()))
            num_kv_heads = first_spec.num_kv_heads if hasattr(first_spec, "num_kv_heads") else 8
            head_size = first_spec.head_size if hasattr(first_spec, "head_size") else 128
            dtype_size = get_dtype_size(first_spec.dtype) if hasattr(first_spec, "dtype") else get_dtype_size(self.dtype)
            elements_per_layer = num_blocks * self.block_size * num_kv_heads * head_size
            total_cache_bytes = elements_per_layer * dtype_size * 2 * len(kv_cache_spec)
            total_cache_gb = total_cache_bytes / (1024**3)
        else:
            # kv_caches is now list of stacked tensors [2, num_blocks, block_size, num_kv_heads, head_size]
            total_cache_gb = sum(
                kv_cache.numel() * get_dtype_size(kv_cache.dtype) / (1024**3)
                for kv_cache in self.kv_caches
            )

        logger.info(
            "KV cache allocated: %d blocks, %d layers, %.2f GB total",
            num_blocks,
            len(self.kv_caches),
            total_cache_gb,
        )

        # Complete engine runner initialization if engine mode enabled
        if self._use_engine_mode and hasattr(self, '_engine_context'):
            try:
                self._complete_engine_runner_init(num_blocks, kv_cache_spec)
            except Exception as e:
                # Engine KV cache is the single source of truth only if the engine
                # runner is fully initialized. If init fails:
                # - strict mode: fail fast (no silent fallback)
                # - non-strict: fall back to the PyTorch path (allocate real KV caches)
                if is_strict_mode():
                    logger.error(
                        "Engine runner initialization failed in strict mode: %s\n"
                        "Strict mode (VLLM_METAL_STRICT_NO_MPS=1) does not allow "
                        "silent fallback to PyTorch/MPS.", e
                    )
                    raise

                logger.warning(
                    "Engine runner initialization failed, falling back to PyTorch/MPS: %s\n"
                    "To require engine mode, set VLLM_METAL_STRICT_NO_MPS=1", e
                )
                self._use_engine_mode = False
                self._engine_runner = None
                self._engine_kv_cache = None

                # If we allocated stub KV caches for engine mode, replace them with
                # real PyTorch KV caches and bind them for the fallback path.
                if use_stub_kv_cache:
                    self._allocate_full_torch_kv_cache(num_blocks, kv_cache_spec)

    def _complete_engine_runner_init(
        self,
        num_blocks: int,
        kv_cache_spec: dict,
    ) -> None:
        """Complete engine runner initialization after KV cache is allocated.

        This creates:
        1. EngineKVCache wrapping the allocated KV caches
        2. Loads model weights to MTLBuffer
        3. Creates the final EngineRunner

        Args:
            num_blocks: Number of KV cache blocks
            kv_cache_spec: KV cache specifications per layer
        """
        from vllm_apple.engine import (
            EngineKVCache,
            EngineRunner,
            EngineWeightLoader,
            KVCacheDescriptor,
        )

        logger.info("Completing engine runner initialization...")

        # Get first layer spec for cache configuration
        first_spec = next(iter(kv_cache_spec.values()))
        num_kv_heads = first_spec.num_kv_heads if hasattr(first_spec, 'num_kv_heads') else 8
        head_size = first_spec.head_size if hasattr(first_spec, 'head_size') else 128

        # Create KV cache descriptor
        kv_desc = KVCacheDescriptor(
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            num_layers=len(kv_cache_spec),
        )

        # Create engine KV cache
        # The engine will use MTLBuffer for KV storage
        # Store as instance variable for KV sync after prefill
        self._engine_kv_cache = EngineKVCache(
            engine_context=self._engine_context,
            descriptor=kv_desc,
        )

        # Load weights from PyTorch model to MTLBuffer
        weight_loader = EngineWeightLoader(
            context=self._engine_context,
            model_config=self.model_config,
        )

        # Determine architecture from model type
        model_type = getattr(self.model_config.hf_config, 'model_type', 'llama')
        model_type_lower = model_type.lower()
        if 'qwen' in model_type_lower:
            arch = 'qwen2'
        elif 'gpt2' in model_type_lower:
            arch = 'gpt2'
        else:
            arch = 'llama'

        weights = weight_loader.load_from_hf_model(self.model, arch=arch)

        # Create the engine runner
        self._engine_runner = EngineRunner(
            context=self._engine_context,
            model_desc=self._engine_model_desc,
            weights=weights,
            kv_cache=self._engine_kv_cache,
        )

        logger.info(
            "Engine runner ready: %d layers, KV cache: %d blocks x %d",
            self._engine_model_desc.num_layers,
            num_blocks,
            self.block_size,
        )

    def _allocate_full_torch_kv_cache(
        self,
        num_blocks: int,
        kv_cache_spec: dict,
    ) -> None:
        """Allocate and bind full torch KV caches for the PyTorch path.

        This is used as a fallback when engine mode is enabled but the engine
        runner fails to initialize. It reverts to the standard vLLM behavior
        where attention layers read/write torch KV tensors.
        """
        self.kv_caches = []

        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                cache_shape = (
                    2,  # K and V stacked
                    num_blocks,
                    layer_spec.block_size,
                    layer_spec.num_kv_heads,
                    layer_spec.head_size,
                )
                kv_cache = torch.zeros(
                    cache_shape,
                    dtype=layer_spec.dtype,
                    device=self.kv_cache_device,
                )
                self.kv_caches.append(kv_cache)
            else:
                cache_shape = (
                    2,  # K and V stacked
                    num_blocks,
                    self.block_size,
                    layer_spec.num_kv_heads if hasattr(layer_spec, 'num_kv_heads') else 8,
                    layer_spec.head_size if hasattr(layer_spec, 'head_size') else 128,
                )
                kv_cache = torch.zeros(
                    cache_shape,
                    dtype=self.dtype,
                    device=self.kv_cache_device,
                )
                self.kv_caches.append(kv_cache)

        # Bind to the forward context used by attention layers.
        kv_caches_dict = {}
        for idx, (layer_name, _) in enumerate(kv_cache_spec.items()):
            kv_caches_dict[layer_name] = self.kv_caches[idx]

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches_dict,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches,
        )

        total_cache_gb = sum(
            kv_cache.numel() * get_dtype_size(kv_cache.dtype) / (1024**3)
            for kv_cache in self.kv_caches
        )
        logger.info(
            "Fallback torch KV cache allocated: %d blocks, %d layers, %.2f GB total",
            num_blocks,
            len(self.kv_caches),
            total_cache_gb,
        )

    def profile_run(self) -> None:
        """Run a profiling pass to measure memory usage.

        This creates dummy inputs and runs a forward pass to measure
        the model's memory footprint for capacity planning.
        """
        logger.info("Running profile pass...")

        # Create dummy inputs sized for profiling
        batch_size = min(16, self.max_num_reqs)
        seq_len = min(128, self.max_model_len)
        num_tokens = batch_size * seq_len

        dummy_input_ids = torch.zeros(
            (num_tokens,),
            dtype=torch.long,
            device=self.device,
        )
        dummy_positions = torch.arange(num_tokens, dtype=torch.long, device=self.device) % seq_len

        # Run forward pass without KV cache (profiling only)
        with torch.inference_mode():
            try:
                with set_forward_context(
                    None, self.vllm_config, num_tokens=num_tokens
                ):
                    self.model(
                        input_ids=dummy_input_ids,
                        positions=dummy_positions,
                    )
            except Exception as e:
                logger.warning("Profile run failed (expected for some models): %s", e)

        # Synchronize MPS (only for PyTorch path, engine mode handles its own sync)
        if not self._use_engine_mode and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

        logger.info("Profile pass complete")

    def warmup_model(self) -> None:
        """Warm up the model with dummy inputs.

        Runs several forward passes to ensure the model and MPS backend
        are fully initialized and optimized.
        """
        logger.info("Warming up model...")

        # Run multiple warmup passes
        for i in range(3):
            logger.debug("Warmup pass %d/3", i + 1)
            self.profile_run()
            time.sleep(0.1)

        # Clean up
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        logger.info("Warmup complete")

    # Timing stats for performance analysis
    _timing_enabled = True  # Enable for debugging
    _timing_stats: dict = {}
    _timing_count = 0

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        """Execute a single model forward pass.

        This is the main entry point for inference. It:
        1. Updates request states from scheduler output
        2. Prepares model inputs (token IDs, positions, attention metadata)
        3. Executes the forward pass
        4. Samples tokens from the output
        5. Returns the sampled tokens and metadata

        Args:
            scheduler_output: Output from scheduler with requests to process

        Returns:
            ModelRunnerOutput with sampled tokens, or None if no tokens to process
        """
        if self._timing_enabled:
            t0 = time.time()

        # Update cached states with scheduler output
        self._update_states(scheduler_output)

        if self._timing_enabled:
            t1 = time.time()

        # Build model inputs from scheduler output
        model_input = self._prepare_model_input(scheduler_output)

        if self._timing_enabled:
            t2 = time.time()

        if model_input.num_scheduled_tokens == 0:
            # No tokens to process - return empty output
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        # Execute forward pass
        # Returns tuple (output, is_logits) where:
        # - PyTorch path: (hidden_states, False)
        # - Engine path: (logits, True)
        forward_output, is_logits = self._execute_forward(model_input, scheduler_output)

        if self._timing_enabled:
            # In engine mode, engine handles its own synchronization via waitUntilCompleted
            # Only synchronize MPS for PyTorch path timing accuracy
            if not self._use_engine_mode and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            t3 = time.time()

        # Sample tokens from the model output
        output = self._sample_tokens(forward_output, model_input, scheduler_output, is_logits=is_logits)

        if self._timing_enabled:
            t4 = time.time()
            self._timing_count += 1
            if self._timing_count <= 5 or self._timing_count % 10 == 0:
                logger.info(
                    f"[Timing #{self._timing_count}] update={1000*(t1-t0):.1f}ms, "
                    f"prepare={1000*(t2-t1):.1f}ms, forward={1000*(t3-t2):.1f}ms, "
                    f"sample={1000*(t4-t3):.1f}ms, total={1000*(t4-t0):.1f}ms"
                )
            # Log Metal attention profile every 50 steps
            if self._timing_count % 50 == 0:
                try:
                    from vllm_apple.v1.attention.backends.metal_attn import get_metal_profile, reset_metal_profile
                    profile = get_metal_profile()
                    if profile['call_count'] > 0:
                        calls = profile['call_count']
                        total_attn = profile['total_forward_ms']
                        kv_update = profile['kv_update_ms']
                        kv_sync = profile.get('kv_sync_ms', 0)
                        kv_to_cpu = profile.get('kv_to_cpu_ms', 0)
                        kv_compute = profile.get('kv_compute_ms', 0)
                        kv_native = profile.get('kv_native_ms', 0)
                        metal_compute = profile['metal_compute_ms']
                        sdpa = profile['sdpa_compute_ms']
                        non_attn = 1000 * (t3 - t2) * (50 / max(calls // 22, 1)) - total_attn
                        logger.info(
                            f"[Metal Profile] calls={calls}, "
                            f"attn_total={total_attn:.1f}ms, "
                            f"kv_update={kv_update:.1f}ms ({100*kv_update/total_attn:.0f}%), "
                            f"metal_kernel={metal_compute:.1f}ms ({100*metal_compute/total_attn:.0f}%), "
                            f"sdpa={sdpa:.1f}ms"
                        )
                        logger.info(
                            f"  KV breakdown: sync={kv_sync:.1f}ms, to_cpu={kv_to_cpu:.1f}ms, "
                            f"compute={kv_compute:.1f}ms, native={kv_native:.1f}ms"
                        )
                        reset_metal_profile()
                except ImportError:
                    pass

        return output

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        """Sample tokens when called separately from execute_model.

        For Apple backend, we always sample in execute_model, so this
        should not be called.

        Args:
            grammar_output: Grammar-constrained output (not used)

        Raises:
            NotImplementedError: This method should not be called
        """
        raise NotImplementedError(
            "sample_tokens should not be called for AppleModelRunner. "
            "Sampling is done in execute_model()."
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update cached request states with scheduler output.

        This method:
        1. Removes finished requests
        2. Adds new requests
        3. Updates running requests

        Args:
            scheduler_output: Output from scheduler
        """
        # Remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Add new requests
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            # Create generator for random sampling if needed
            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                block_ids=new_req_data.block_ids,
                generator=generator,
            )
            self.requests[req_id] = req_state

        # Update running requests
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            if req_id not in self.requests:
                continue

            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]

            # Update computed tokens
            req_state.num_computed_tokens = num_computed_tokens

            # Update block IDs
            if new_block_ids is not None:
                # Check if resumed from preemption
                if req_id in req_data.resumed_req_ids:
                    # Replace block IDs
                    req_state.block_ids = new_block_ids
                else:
                    # Append new blocks
                    for block_ids, new_ids in zip(
                        req_state.block_ids, new_block_ids
                    ):
                        block_ids.extend(new_ids)

            # Update output tokens (for non-last PP ranks)
            if not self.is_last_pp_rank:
                new_token_ids = req_data.new_token_ids[i]
                req_state.output_token_ids.extend(new_token_ids)

    def _prepare_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> AppleModelInput:
        """Prepare inputs for model forward pass.

        This method builds the input tensors from scheduler output:
        - input_ids: Token IDs to process
        - positions: Position IDs for each token
        - attn_metadata: Attention metadata (sequence lengths, masks, etc.)
        - sampling_metadata: Sampling parameters for each request

        STEP-AWARE DEVICE SELECTION:
        - Prefill steps (has scheduled_new_reqs): inputs on MPS for PyTorch
        - Decode-only steps with engine mode: inputs on CPU for engine
        This avoids MPS→CPU conversion overhead at the engine boundary.

        Args:
            scheduler_output: Output from scheduler

        Returns:
            AppleModelInput with all necessary tensors
        """
        # Detect step type early for device selection
        is_decode_only = self._is_decode_only_step(scheduler_output)

        input_ids_list = []
        positions_list = []
        req_ids = []
        seq_lens = []

        # Process new requests (prefill phase)
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_scheduled == 0:
                continue

            req_state = self.requests.get(req_id)
            if req_state is None:
                continue

            # Get prompt tokens to process
            prompt_token_ids = req_state.prompt_token_ids or []
            num_computed = req_state.num_computed_tokens

            # Tokens to process in this step
            end_idx = min(num_computed + num_scheduled, len(prompt_token_ids))
            tokens = prompt_token_ids[num_computed:end_idx]
            positions = list(range(num_computed, num_computed + len(tokens)))

            if tokens:
                input_ids_list.extend(tokens)
                positions_list.extend(positions)
                req_ids.append(req_id)
                seq_lens.append(len(tokens))

        # Process cached requests (decode phase)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self.requests:
                continue

            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_scheduled == 0:
                continue

            req_state = self.requests[req_id]

            # Get new token IDs
            if idx < len(cached_reqs.new_token_ids):
                new_token_ids = cached_reqs.new_token_ids[idx]
            elif cached_reqs.all_token_ids and req_id in cached_reqs.all_token_ids:
                # Use last token from all_token_ids
                all_tokens = cached_reqs.all_token_ids[req_id]
                new_token_ids = [all_tokens[-1]] if all_tokens else []
            elif req_state.output_token_ids:
                # Use last output token
                new_token_ids = [req_state.output_token_ids[-1]]
            else:
                continue

            if not new_token_ids:
                continue

            # Position is the total number of computed tokens
            num_computed = req_state.num_computed_tokens
            positions = list(range(num_computed, num_computed + len(new_token_ids)))

            input_ids_list.extend(new_token_ids)
            positions_list.extend(positions)
            req_ids.append(req_id)
            seq_lens.append(len(new_token_ids))

        # Convert to tensors using step-aware device selection
        # - Decode-only + engine mode: CPU (engine expects CPU inputs)
        # - Prefill or non-engine: MPS (PyTorch model expects MPS inputs)
        input_device = self._get_input_device(is_decode_only)
        if input_ids_list:
            input_ids = torch.tensor(
                input_ids_list, dtype=torch.long, device=input_device
            )
            positions = torch.tensor(
                positions_list, dtype=torch.long, device=input_device
            )
        else:
            input_ids = torch.tensor([], dtype=torch.long, device=input_device)
            positions = torch.tensor([], dtype=torch.long, device=input_device)

        # Build attention metadata
        # Pass positions_list (Python list) to avoid MPS→CPU sync from .tolist()
        # Pass is_decode_only for consistent device selection
        attn_metadata = self._build_attn_metadata(req_ids, seq_lens, positions_list, is_decode_only)

        # Build sampling metadata
        sampling_metadata = self._build_sampling_metadata(req_ids, seq_lens)

        return AppleModelInput(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            num_scheduled_tokens=len(input_ids_list),
            num_reqs=len(req_ids),
            req_ids=req_ids,
            seq_lens=seq_lens,
            sampling_metadata=sampling_metadata,
        )

    def _build_attn_metadata(
        self,
        req_ids: list[str],
        seq_lens: list[int],
        positions_list: list[int],
        is_decode_only: bool = False,
    ) -> Any:
        """Build attention metadata for the forward pass.

        Creates the AppleAttentionMetadata that contains:
        - Sequence lengths (FULL sequence length including past tokens)
        - Block mappings for paged attention
        - Query start locations
        - Slot mappings for KV cache

        Args:
            req_ids: Request IDs
            seq_lens: Number of NEW tokens for each request (query length)
            positions_list: Position list (Python list to avoid MPS sync)
            is_decode_only: True if this is a decode-only step (no prefill)

        Returns:
            AppleAttentionMetadata
        """
        from vllm_apple.v1.attention.backends.metal_attn import (
            MetalAttentionMetadata,
        )

        num_tokens = sum(seq_lens)
        num_reqs = len(req_ids)

        # Determine if this is prefill or decode
        # Prefill: at least one sequence has more than 1 token
        # Decode: all sequences have exactly 1 token
        is_prefill = any(s > 1 for s in seq_lens)
        num_decode_tokens = sum(1 for s in seq_lens if s == 1) if not is_prefill else 0

        # Use step-aware device selection for consistent tensor placement
        # - Decode-only + engine mode: CPU (engine expects CPU inputs)
        # - Prefill or non-engine: MPS (PyTorch model expects MPS inputs)
        input_device = self._get_input_device(is_decode_only)

        # Build query start locations (cumulative sum of query lengths with leading 0)
        query_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=input_device)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=input_device)
        query_start_loc[1:] = torch.cumsum(query_lens_tensor, dim=0)

        # Build FULL sequence lengths (computed_tokens + new_tokens)
        # This is critical for attention to know how many KV tokens to attend to
        full_seq_lens = []
        for req_idx, req_id in enumerate(req_ids):
            if req_id in self.requests:
                req_state = self.requests[req_id]
                # Full sequence length = already computed + new tokens being processed
                full_len = req_state.num_computed_tokens + seq_lens[req_idx]
                full_seq_lens.append(full_len)
            else:
                full_seq_lens.append(seq_lens[req_idx])

        seq_lens_tensor = torch.tensor(full_seq_lens, dtype=torch.int32, device=input_device)

        # Build slot mapping for KV cache using actual positions
        # slot_mapping tells where to store each token's KV in the cache
        # Optimized: avoid .item() calls by computing on GPU where possible

        # For decode (most common case): single token per request
        if num_tokens == num_reqs and num_reqs > 0:
            # Fast path: all requests have exactly 1 token (decode phase)
            # Use positions_list directly (Python list) - no MPS sync needed
            slot_mapping_list = []
            for req_idx, req_id in enumerate(req_ids):
                token_pos = positions_list[req_idx]
                if req_id in self.requests:
                    req_state = self.requests[req_id]
                    block_ids = req_state.block_ids[0] if req_state.block_ids else []
                    block_idx = token_pos // self.block_size
                    block_offset = token_pos % self.block_size
                    if block_idx < len(block_ids):
                        slot = block_ids[block_idx] * self.block_size + block_offset
                    else:
                        slot = token_pos
                else:
                    slot = token_pos
                slot_mapping_list.append(slot)
            slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=input_device)
        else:
            # General path: prefill or mixed batch
            # Use positions_list directly (Python list) - no MPS sync needed
            slot_mapping_list = []
            pos_idx = 0
            for req_idx, req_id in enumerate(req_ids):
                num_new_tokens = seq_lens[req_idx]
                if req_id in self.requests:
                    req_state = self.requests[req_id]
                    block_ids = req_state.block_ids[0] if req_state.block_ids else []
                    for token_idx in range(num_new_tokens):
                        token_pos = positions_list[pos_idx + token_idx]
                        block_idx = token_pos // self.block_size
                        block_offset = token_pos % self.block_size
                        if block_idx < len(block_ids):
                            slot = block_ids[block_idx] * self.block_size + block_offset
                        else:
                            slot = token_pos
                        slot_mapping_list.append(slot)
                else:
                    for token_idx in range(num_new_tokens):
                        slot_mapping_list.append(positions_list[pos_idx + token_idx])
                pos_idx += num_new_tokens
            slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=input_device)

        # Build block tables for paged attention
        # Optimized: collect all block IDs first, then create tensor once
        block_tables = []
        max_blocks = 0
        for req_id in req_ids:
            if req_id in self.requests:
                req_state = self.requests[req_id]
                bt = req_state.block_ids[0] if req_state.block_ids else []
            else:
                bt = []
            block_tables.append(bt)
            if len(bt) > max_blocks:
                max_blocks = len(bt)

        # Convert block tables to tensor - single allocation
        if max_blocks > 0:
            # Pre-allocate with -1 fill
            block_table_data = [[-1] * max_blocks for _ in range(num_reqs)]
            for i, bt in enumerate(block_tables):
                for j, block_id in enumerate(bt):
                    block_table_data[i][j] = block_id
            block_table = torch.tensor(
                block_table_data, dtype=torch.int32, device=input_device
            )
        else:
            block_table = torch.zeros(
                (num_reqs, 1), dtype=torch.int32, device=input_device
            )

        # Calculate max lengths
        # max_query_len: max NEW tokens being processed
        # max_seq_len: max FULL sequence length (for KV cache access)
        max_query_len = max(seq_lens) if seq_lens else 0
        max_seq_len = max(full_seq_lens) if full_seq_lens else 0

        return MetalAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens_tensor,
            block_table=block_table,
            slot_mapping=slot_mapping,
            scheduler_metadata=None,
            causal=True,
            num_decode_tokens=num_decode_tokens,
        )

    def _build_sampling_metadata(
        self,
        req_ids: list[str],
        seq_lens: list[int],
    ) -> SamplingMetadata:
        """Build sampling metadata for token generation.

        Args:
            req_ids: Request IDs
            seq_lens: Sequence lengths

        Returns:
            SamplingMetadata for the sampler (vLLM 0.12.0 format)
        """
        from vllm.v1.sample.logits_processor.state import LogitsProcessors

        num_reqs = len(req_ids)

        # Collect sampling parameters and generators
        temperatures = []
        top_ps = []
        top_ks = []
        frequency_penalties = []
        presence_penalties = []
        repetition_penalties = []
        generators_dict: dict[int, torch.Generator] = {}
        output_token_ids: list[list[int]] = []

        for idx, req_id in enumerate(req_ids):
            if req_id in self.requests:
                req_state = self.requests[req_id]
                sp = req_state.sampling_params
                if sp is not None:
                    temperatures.append(sp.temperature)
                    top_ps.append(sp.top_p if sp.top_p is not None else 1.0)
                    top_ks.append(sp.top_k if sp.top_k is not None else 0)
                    frequency_penalties.append(sp.frequency_penalty if sp.frequency_penalty is not None else 0.0)
                    presence_penalties.append(sp.presence_penalty if sp.presence_penalty is not None else 0.0)
                    repetition_penalties.append(sp.repetition_penalty if sp.repetition_penalty is not None else 1.0)
                else:
                    temperatures.append(1.0)
                    top_ps.append(1.0)
                    top_ks.append(0)
                    frequency_penalties.append(0.0)
                    presence_penalties.append(0.0)
                    repetition_penalties.append(1.0)

                if req_state.generator is not None:
                    generators_dict[idx] = req_state.generator

                # Get output tokens generated so far
                output_token_ids.append(req_state.output_token_ids if hasattr(req_state, 'output_token_ids') else [])
            else:
                temperatures.append(1.0)
                top_ps.append(1.0)
                top_ks.append(0)
                frequency_penalties.append(0.0)
                presence_penalties.append(0.0)
                repetition_penalties.append(1.0)
                output_token_ids.append([])

        # Build tensors
        temperature_tensor = torch.tensor(temperatures, dtype=torch.float32, device=self.device) if temperatures else None
        top_p_tensor = torch.tensor(top_ps, dtype=torch.float32, device=self.device) if top_ps else None
        top_k_tensor = torch.tensor(top_ks, dtype=torch.int32, device=self.device) if top_ks else None
        frequency_penalties_tensor = torch.tensor(frequency_penalties, dtype=torch.float32, device=self.device)
        presence_penalties_tensor = torch.tensor(presence_penalties, dtype=torch.float32, device=self.device)
        repetition_penalties_tensor = torch.tensor(repetition_penalties, dtype=torch.float32, device=self.device)

        # Check if all greedy or all random
        all_greedy = all(t == 0.0 for t in temperatures) if temperatures else True
        all_random = all(t > 0.0 for t in temperatures) if temperatures else False

        # Check if no penalties
        no_penalties = (
            all(fp == 0.0 for fp in frequency_penalties) and
            all(pp == 0.0 for pp in presence_penalties) and
            all(rp == 1.0 for rp in repetition_penalties)
        )

        return SamplingMetadata(
            temperature=temperature_tensor,
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=top_p_tensor,
            top_k=top_k_tensor,
            generators=generators_dict,
            max_num_logprobs=None,
            no_penalties=no_penalties,
            prompt_token_ids=None,
            frequency_penalties=frequency_penalties_tensor,
            presence_penalties=presence_penalties_tensor,
            repetition_penalties=repetition_penalties_tensor,
            output_token_ids=output_token_ids,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

    def _execute_forward(
        self,
        model_input: AppleModelInput,
        scheduler_output: "SchedulerOutput" = None,
    ) -> tuple[torch.Tensor, bool]:
        """Execute the model forward pass.

        The forward context provides access to KV caches, so we don't
        pass them directly to the model.

        Args:
            model_input: Prepared model input
            scheduler_output: Scheduler output (used by execute_model)

        Returns:
            Tuple of (output_tensor, is_logits):
            - PyTorch path: (hidden_states [num_tokens, hidden_size], False)
            - Engine path: (logits [num_reqs, vocab_size], True)
        """
        # Engine mode path (v2.0 Metal engine)
        if self._use_engine_mode and self._engine_runner is not None:
            # Detect prefill from attn_metadata: num_decode_tokens < num_actual_tokens
            attn_metadata = model_input.attn_metadata
            has_prefill = attn_metadata.num_decode_tokens < attn_metadata.num_actual_tokens

            if has_prefill and not is_engine_prefill_enabled():
                if is_strict_mode():
                    raise RuntimeError(
                        "Engine mode is enabled but engine prefill is disabled "
                        "(VLLM_APPLE_ENGINE_PREFILL=0). Strict mode "
                        "(VLLM_METAL_STRICT_NO_MPS=1) does not allow routing "
                        "prefill through PyTorch/MPS with KV sync. Enable engine "
                        "prefill (VLLM_APPLE_ENGINE_PREFILL=1) or disable strict mode."
                    )
                # Default behavior: route prefill through PyTorch and sync KV for decode.
                logger.debug(
                    f"Prefill step (num_tokens={model_input.num_scheduled_tokens}, "
                    f"num_decode={attn_metadata.num_decode_tokens}) - using PyTorch path"
                )
                result = self._execute_prefill_and_sync_kv(model_input)
                return result, False  # Returns hidden states, not logits

            # Engine path (decode-only or prefill enabled): return logits.
            return self._execute_forward_engine(model_input), True
        elif self._use_engine_mode:
            # Engine mode enabled but runner not initialized
            if is_strict_mode():
                raise RuntimeError(
                    "Engine mode enabled (VLLM_APPLE_USE_ENGINE=1) but engine runner "
                    "not initialized. In strict mode (VLLM_METAL_STRICT_NO_MPS=1), "
                    "silent fallback to PyTorch/MPS is not allowed. "
                    "Ensure model weights are loaded to engine before inference."
                )
            logger.warning(
                "Engine mode enabled but runner not initialized - "
                "falling back to PyTorch/MPS path"
            )

        # Standard PyTorch path
        # Set forward context - attention layers access KV caches through this
        with set_forward_context(
            model_input.attn_metadata,
            self.vllm_config,
            num_tokens=model_input.num_scheduled_tokens,
        ):
            # Run model forward pass
            hidden_states = self.model(
                input_ids=model_input.input_ids,
                positions=model_input.positions,
            )

        return hidden_states, False

    def _execute_prefill_and_sync_kv(
        self,
        model_input: AppleModelInput,
    ) -> torch.Tensor:
        """Execute prefill via PyTorch and sync KV cache to engine.

        This method:
        1. Runs prefill through PyTorch path (attention writes to torch KV cache)
        2. Syncs the KV data from torch cache to engine MTLBuffers
        3. Returns hidden states for LM head computation

        This ensures that after prefill, the engine's KV cache contains the
        correct data for subsequent decode steps.

        IMPORTANT: This sync is a temporary solution until prefill is moved
        to the engine. It adds overhead but ensures correctness.

        Args:
            model_input: Prepared model input

        Returns:
            Hidden states tensor [num_tokens, hidden_size]
        """
        # Execute prefill via PyTorch path
        with set_forward_context(
            model_input.attn_metadata,
            self.vllm_config,
            num_tokens=model_input.num_scheduled_tokens,
        ):
            hidden_states = self.model(
                input_ids=model_input.input_ids,
                positions=model_input.positions,
            )

        # Sync KV cache from torch to engine
        # This ensures engine's MTLBuffers contain the prefill KV data
        if hasattr(self, '_engine_kv_cache') and self._engine_kv_cache is not None:
            attn_metadata = model_input.attn_metadata

            # Get seq_lens from attn_metadata (full sequence lengths)
            seq_lens = attn_metadata.seq_lens

            # Get block_table
            block_table = attn_metadata.block_table if attn_metadata.block_table is not None else torch.empty(0, 0, dtype=torch.int32)

            if block_table.numel() > 0:
                logger.debug(
                    f"Syncing KV cache after prefill: "
                    f"{len(self.kv_caches)} layers, "
                    f"block_table shape {block_table.shape}"
                )
                self._engine_kv_cache.sync_from_torch_cache(
                    torch_caches=self.kv_caches,
                    block_table=block_table,
                    seq_lens=seq_lens,
                )

        return hidden_states

    def _execute_forward_engine(
        self,
        model_input: AppleModelInput,
    ) -> torch.Tensor:
        """Execute forward pass using v2.0 Metal engine.

        This path uses the EngineRunner for pure Metal execution with
        step-boundary-only synchronization.

        Args:
            model_input: Prepared model input

        Returns:
            Logits from the model [num_tokens, vocab_size]
            (Engine computes LM head internally)
        """
        from vllm_apple.engine import StepDescriptor, EngineInputs

        # Convert AppleModelInput to EngineInputs
        attn_metadata = model_input.attn_metadata

        # Derive step_kind from attn_metadata using available attributes
        # MetalAttentionMetadata has: num_actual_tokens, num_decode_tokens
        # Pure decode: num_decode_tokens == num_actual_tokens
        # Has prefill: num_decode_tokens < num_actual_tokens
        has_prefill = attn_metadata.num_decode_tokens < attn_metadata.num_actual_tokens
        step_kind = "prefill" if has_prefill else "decode"

        # Derive num_seqs from model_input (reliable, always available)
        num_seqs_active = model_input.num_reqs

        # Derive max_num_blocks_per_seq from block_table shape
        # block_table shape is [num_seqs, max_blocks_per_seq]
        if attn_metadata.block_table is not None and attn_metadata.block_table.numel() > 0:
            max_num_blocks_per_seq = attn_metadata.block_table.shape[1]
        else:
            max_num_blocks_per_seq = 0

        step_desc = StepDescriptor(
            step_id=0,  # Will be tracked by runner
            step_kind=step_kind,
            num_scheduled_tokens=model_input.num_scheduled_tokens,
            num_seqs_active=num_seqs_active,
            max_num_blocks_per_seq=max_num_blocks_per_seq,
        )

        # CRITICAL: Use attn_metadata.seq_lens which contains FULL sequence lengths
        # (computed_tokens + new_tokens), NOT model_input.seq_lens which contains
        # query lengths (number of new tokens per request).
        # Engine/kernels need full context lengths for:
        # - max_seq_len calculation for attention
        # - decode KV-write position (seq_len - 1)
        # - paged attention context length
        #
        # STEP-AWARE BOUNDARY: For decode-only steps, inputs were prepared on CPU
        # directly via _get_input_device(is_decode_only=True). This avoids
        # MPS→CPU sync overhead. The ensure_cpu_tensor call validates the invariant
        # that engine inputs must be on CPU - no conversion should be needed.
        from vllm_apple.engine.boundary import ensure_cpu_tensor

        engine_inputs = EngineInputs(
            token_ids=ensure_cpu_tensor(model_input.input_ids, "token_ids"),
            positions=ensure_cpu_tensor(model_input.positions, "positions"),
            block_table=ensure_cpu_tensor(attn_metadata.block_table, "block_table") if attn_metadata.block_table is not None else torch.empty(0, 0, dtype=torch.int32),
            slot_mapping=ensure_cpu_tensor(attn_metadata.slot_mapping, "slot_mapping") if attn_metadata.slot_mapping is not None else torch.empty(0, dtype=torch.int32),
            seq_lens=ensure_cpu_tensor(attn_metadata.seq_lens.to(dtype=torch.int32), "seq_lens"),
            query_start_locs=ensure_cpu_tensor(attn_metadata.query_start_loc, "query_start_locs"),
        )

        # Execute via engine runner
        outputs = self._engine_runner.execute_step(step_desc, engine_inputs)

        # Return logits directly (engine already computed LM head)
        # The caller will skip LM head when is_logits=True
        #
        # NOTE: Transferring logits to self.device (MPS) is a temporary tradeoff.
        # This reintroduces MPS into the post-engine path, which conflicts with
        # "pure Metal end-to-end" goal. Future options:
        # - Keep logits on CPU and do CPU-based sampling
        # - Have engine return top-k candidates directly (GPU selection kernel)
        # For now, sampling uses PyTorch ops that expect device tensors.
        return outputs.logits.to(self.device)

    def _sample_tokens(
        self,
        hidden_states: torch.Tensor,
        model_input: AppleModelInput,
        scheduler_output: "SchedulerOutput",
        is_logits: bool = False,
    ) -> ModelRunnerOutput:
        """Sample tokens from model output.

        This method:
        1. Extracts logits from hidden states (or uses logits directly if is_logits=True)
        2. Applies the language model head (skipped if is_logits=True)
        3. Uses the sampler to generate tokens
        4. Updates request states
        5. Returns ModelRunnerOutput

        Args:
            hidden_states: Model output hidden states, or logits if is_logits=True
            model_input: Input that was used
            scheduler_output: Scheduler output
            is_logits: If True, hidden_states are already logits (skip LM head)

        Returns:
            ModelRunnerOutput with sampled tokens
        """
        if model_input.num_reqs == 0:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        # If is_logits=True, hidden_states are already logits from engine mode
        # Engine returns [num_tokens, vocab_size], need to extract last token per seq for prefill
        if is_logits:
            # Check if we need to extract last token positions (prefill case)
            num_reqs = model_input.num_reqs
            if hidden_states.shape[0] != num_reqs:
                # Prefill: engine returns [num_tokens, vocab_size], extract last token per seq
                # Invariant check: sum(seq_lens) must equal num_tokens for correct indexing
                expected_tokens = sum(model_input.seq_lens)
                actual_tokens = hidden_states.shape[0]
                if expected_tokens != actual_tokens:
                    raise ValueError(
                        f"Engine logits shape mismatch: sum(seq_lens)={expected_tokens} "
                        f"but got {actual_tokens} tokens. This indicates token packing "
                        f"inconsistency between engine and model_input."
                    )
                last_token_positions = []
                current_pos = 0
                for seq_len in model_input.seq_lens:
                    last_token_positions.append(current_pos + seq_len - 1)
                    current_pos += seq_len
                logits = hidden_states[last_token_positions]
            else:
                # Decode: engine returns [num_reqs, vocab_size], use directly
                logits = hidden_states
        else:
            # Calculate positions of last tokens
            last_token_positions = []
            current_pos = 0
            for seq_len in model_input.seq_lens:
                last_token_positions.append(current_pos + seq_len - 1)
                current_pos += seq_len

            # Extract hidden states at last token positions
            if hidden_states.dim() == 2:
                # [num_tokens, hidden_size]
                logits_hidden = hidden_states[last_token_positions]
            else:
                # Handle other shapes
                hidden_states = hidden_states.view(-1, hidden_states.size(-1))
                logits_hidden = hidden_states[last_token_positions]

            # Apply language model head to get logits
            # Use compute_logits method if available (vLLM models use this)
            if hasattr(self.model, "compute_logits"):
                logits = self.model.compute_logits(logits_hidden)
            elif hasattr(self.model, "logits_processor") and hasattr(self.model, "lm_head"):
                # Use logits_processor if available
                logits = self.model.logits_processor(self.model.lm_head, logits_hidden)
            elif hasattr(self.model, "lm_head"):
                # Fallback: directly access lm_head weight for linear projection
                lm_head_weight = self.model.lm_head.weight
                logits = torch.nn.functional.linear(logits_hidden, lm_head_weight)
            elif hasattr(self.model, "output"):
                logits = self.model.output(logits_hidden)
            else:
                # Assume hidden states are already logits
                logits = logits_hidden

        # Use the sampler to generate tokens
        # For simplicity, we'll do manual sampling here
        # In production, would use self.sampler with proper metadata
        sampled_token_ids = self._manual_sample(
            logits, model_input.req_ids, model_input.sampling_metadata
        )

        # Update request states with sampled tokens
        for req_idx, req_id in enumerate(model_input.req_ids):
            if req_id in self.requests:
                req_state = self.requests[req_id]
                token_id = sampled_token_ids[req_idx]
                if isinstance(token_id, (list, np.ndarray)):
                    token_id = int(token_id[0]) if len(token_id) > 0 else 0
                req_state.output_token_ids.append(token_id)
                # Increment by the number of tokens processed in this step
                # For prefill: seq_lens[req_idx] = number of prefill tokens
                # For decode: seq_lens[req_idx] = 1
                tokens_processed = model_input.seq_lens[req_idx] if model_input.seq_lens else 1
                req_state.num_computed_tokens += tokens_processed

        # Build output
        req_id_to_index = {
            req_id: idx for idx, req_id in enumerate(model_input.req_ids)
        }

        # Convert sampled_token_ids to list[list[int]] format expected by vLLM 0.12.0
        # Each element should be a list of sampled tokens (typically one token per request)
        sampled_token_ids_formatted = []
        for token_ids in sampled_token_ids:
            if isinstance(token_ids, np.ndarray):
                sampled_token_ids_formatted.append(token_ids.tolist())
            elif isinstance(token_ids, (list, tuple)):
                sampled_token_ids_formatted.append(list(token_ids))
            else:
                sampled_token_ids_formatted.append([int(token_ids)])

        # Create pooler output (None for each request - not a pooling model)
        pooler_output = [None for _ in model_input.req_ids]

        return ModelRunnerOutput(
            req_ids=model_input.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids_formatted,
            logprobs=None,  # Could compute if needed
            prompt_logprobs_dict={},  # Could compute if needed
            pooler_output=pooler_output,
        )

    def _manual_sample(
        self,
        logits: torch.Tensor,
        req_ids: list[str],
        sampling_metadata: Optional[SamplingMetadata],
    ) -> list:
        """Manual token sampling implementation.

        This is a simplified sampling implementation. In production,
        would use vLLM's Sampler with full metadata.

        Args:
            logits: Logits tensor [num_reqs, vocab_size]
            req_ids: Request IDs
            sampling_metadata: Sampling parameters

        Returns:
            List of sampled token IDs (as numpy arrays for compatibility)
        """
        sampled_token_ids = []

        for req_idx, req_id in enumerate(req_ids):
            if req_id not in self.requests:
                sampled_token_ids.append(np.array([0], dtype=np.int32))
                continue

            req_state = self.requests[req_id]
            sampling_params = req_state.sampling_params

            if sampling_params is None:
                # Default to greedy
                token_id = torch.argmax(logits[req_idx]).item()
                sampled_token_ids.append(np.array([token_id], dtype=np.int32))
                continue

            # Get logits for this request
            req_logits = logits[req_idx].clone()

            # Apply temperature
            if sampling_params.temperature > 0:
                req_logits = req_logits / sampling_params.temperature

            # Apply top-k
            if sampling_params.top_k > 0:
                top_k = min(sampling_params.top_k, req_logits.size(-1))
                topk_values, topk_indices = torch.topk(req_logits, top_k)
                # Mask out tokens not in top-k
                mask = torch.full_like(req_logits, float("-inf"))
                mask[topk_indices] = 0.0
                req_logits = req_logits + mask

            # Apply top-p (nucleus sampling)
            if 0.0 < sampling_params.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    req_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
                # Keep at least one token
                if sorted_indices_to_remove.all():
                    sorted_indices_to_remove[0] = False
                else:
                    # Shift to keep first token above threshold
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                req_logits = req_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample
            if sampling_params.temperature == 0:
                # Greedy
                token_id = torch.argmax(req_logits).item()
            else:
                # Random sampling
                probs = F.softmax(req_logits, dim=-1)
                if req_state.generator is not None:
                    token_id = torch.multinomial(
                        probs, num_samples=1, generator=req_state.generator
                    ).item()
                else:
                    token_id = torch.multinomial(probs, num_samples=1).item()

            sampled_token_ids.append(np.array([token_id], dtype=np.int32))

        return sampled_token_ids
