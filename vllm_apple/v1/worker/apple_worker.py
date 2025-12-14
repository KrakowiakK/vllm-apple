# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Silicon worker for vLLM V1 engine.

This worker handles model execution on Apple Silicon using MPS (Metal Performance Shaders).
It follows the HPUWorker pattern from vllm-gaudi, inheriting from WorkerBase to avoid
Triton dependencies.

Key design decisions:
- Inherits from WorkerBase (NOT GPUWorker) to avoid Triton
- Uses gloo backend for distributed (NCCL unavailable on macOS)
- Forces world_size=1 (no multi-GPU on Apple Silicon)
- Uses unified memory management with psutil
- Implements full worker interface for V1 engine
"""

import gc
import os
from typing import TYPE_CHECKING, Any, Optional

import psutil
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

# Import Apple-specific components
from vllm_apple.v1.worker.apple_model_runner import AppleModelRunner
from vllm_apple.engine.strict_mode import enable_strict_mode

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


class AppleWorker(WorkerBase):
    """Worker for Apple Silicon (MPS) inference.

    This worker manages model execution on Apple Silicon devices using MPS
    (Metal Performance Shaders) for GPU acceleration. It follows the HPUWorker
    pattern from vllm-gaudi and inherits from WorkerBase to avoid Triton dependencies.

    Architecture:
    - MPS device for GPU acceleration
    - Unified memory management (shared CPU/GPU memory)
    - Gloo backend for distributed (world_size=1)
    - AppleModelRunner for model execution
    - KV cache in unified memory

    Key features:
    - Single-device operation (no multi-GPU)
    - Memory profiling with psutil
    - Model warmup for performance
    - Full V1 worker interface
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """Initialize the Apple worker.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device rank (always 0 for single-device Apple Silicon)
            rank: Global rank in distributed setup (always 0 for Apple Silicon)
            distributed_init_method: Method for distributed initialization
            is_driver_worker: Whether this is the driver worker
        """
        # Call parent __init__ to set common attributes
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # Apple Silicon specific settings
        # Force world_size=1 for Apple Silicon (no multi-GPU support)
        self.parallel_config.world_size = 1
        self.parallel_config.tensor_parallel_size = 1
        self.parallel_config.pipeline_parallel_size = 1

        # Disable custom all-reduce for Apple Silicon
        self.parallel_config.disable_custom_all_reduce = True

        # Handle trust_remote_code
        if self.model_config.trust_remote_code:
            from vllm.utils.import_utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Memory tracking
        self.init_memory_bytes: Optional[int] = None
        self.available_kv_cache_memory_bytes: Optional[int] = None
        self.model_memory_usage: int = 0

        logger.info(
            "AppleWorker initialized: rank=%d, local_rank=%d, is_driver=%s",
            self.rank,
            self.local_rank,
            self.is_driver_worker,
        )

    def init_device(self) -> None:
        """Initialize MPS device and distributed environment.

        This method:
        1. Sets up MPS device for GPU acceleration
        2. Enables MPS fallback for unsupported ops
        3. Initializes distributed environment with gloo backend
        4. Sets random seed
        5. Takes initial memory snapshot
        6. Creates AppleModelRunner instance

        Distributed setup uses gloo backend since NCCL is not available on macOS.
        World size is forced to 1 for single-device Apple Silicon.
        """
        logger.info("Initializing Apple Silicon (MPS) device...")

        # Set MPS device
        self.device = torch.device("mps")
        current_platform.set_device(self.device)

        # Enable MPS fallback for unsupported operations
        # This allows PyTorch to fall back to CPU for ops not supported by MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Set unique identifier for distributed shared memory
        if self.distributed_init_method:
            os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]

        # Check dtype support
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Initialize distributed environment with gloo backend
        # NCCL is not available on Apple Silicon, so we use gloo
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            backend="gloo",  # Use gloo instead of NCCL for Apple Silicon
        )

        # Set random seed for reproducibility
        set_random_seed(self.model_config.seed)

        # Get initial memory snapshot
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        # Track initial memory usage
        self.init_memory_bytes = self._get_available_memory_bytes()

        logger.info(
            "Initial memory: %.2f GiB",
            self.init_memory_bytes / GiB_bytes,
        )

        # Create AppleModelRunner
        self.model_runner = AppleModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
            is_driver_worker=self.is_driver_worker,
        )

        logger.info("Apple Silicon device initialized successfully")

        # Enable strict mode if VLLM_METAL_STRICT_NO_MPS=1
        # This applies monkey-patches to detect torch.mps ops in hot path
        enable_strict_mode()

    def load_model(self) -> None:
        """Load the model onto MPS device.

        Delegates to AppleModelRunner for actual model loading.
        The model is loaded using vLLM's standard model loader and
        moved to the MPS device.
        """
        logger.info("Loading model...")
        self.model_runner.load_model()

        # Apply MoE patches after model is loaded (critical for MoE models)
        self._apply_moe_patches_after_load()

        # Track model memory usage
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        mem_after_load = self._get_available_memory_bytes()
        self.model_memory_usage = self.init_memory_bytes - mem_after_load

        logger.info(
            "Model loaded successfully (model memory: %.2f GiB)",
            self.model_memory_usage / GiB_bytes,
        )

    def _apply_moe_patches_after_load(self) -> None:
        """Apply MoE patches after model loading.

        This patches topk_softmax and fused_experts for MPS compatibility.
        Must be called after model loading when all vLLM MoE modules are imported.
        """
        try:
            import sys

            # Define PyTorch fallback for topk_softmax
            def topk_softmax_mps(
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                token_expert_indices: torch.Tensor,
                gating_output: torch.Tensor,
                renormalize: bool = False,
            ) -> None:
                """PyTorch implementation of topk_softmax for MPS."""
                num_tokens, num_experts = gating_output.shape
                topk = topk_weights.shape[1]

                topk_weights_tmp, topk_ids_tmp = torch.topk(
                    gating_output, k=topk, dim=-1
                )
                topk_weights_tmp = torch.softmax(
                    topk_weights_tmp.float(), dim=-1
                ).to(topk_weights.dtype)

                if renormalize:
                    topk_weights_tmp = topk_weights_tmp / topk_weights_tmp.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-6)

                topk_weights.copy_(topk_weights_tmp)
                topk_ids.copy_(topk_ids_tmp)

                if token_expert_indices.numel() > 0:
                    indices = torch.arange(
                        num_tokens * topk,
                        device=gating_output.device,
                        dtype=token_expert_indices.dtype,
                    )
                    token_expert_indices.copy_(indices.view_as(token_expert_indices))

            # Define vllm_topk_softmax replacement
            def vllm_topk_softmax_mps(
                topk_weights: torch.Tensor,
                topk_indices: torch.Tensor,
                token_expert_indices: torch.Tensor,
                gating_output: torch.Tensor,
                renormalize: bool,
            ) -> tuple:
                """MPS replacement for vllm_topk_softmax."""
                topk_softmax_mps(
                    topk_weights,
                    topk_indices,
                    token_expert_indices,
                    gating_output,
                    renormalize,
                )
                return topk_weights, topk_indices

            # Define fused_experts replacement
            _fused_experts_debug_logged = [False]
            # OPTIMIZATION: Cache for pre-combined weights (19% faster gather)
            _combined_weights_cache = {}
            # Profiling counters
            _moe_profile = {'call_count': 0, 'total_ms': 0.0}
            import time as _time

            def fused_experts_mps(
                hidden_states: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                inplace: bool = False,
                activation: str = "silu",
                apply_router_weight_on_input: bool = False,
                global_num_experts: int = -1,
                expert_map: torch.Tensor | None = None,
                quant_config = None,
                allow_deep_gemm: bool = False,
                allow_cutlass_block_scaled_grouped_gemm: bool = False,
            ) -> torch.Tensor:
                """Optimized PyTorch implementation of fused_experts for MPS.

                Uses pre-combined weights for faster single gather (19% speedup).
                Falls back to loop-based approach for large batches during profiling.
                """
                _t_start = _time.perf_counter()
                orig_dtype = hidden_states.dtype
                num_tokens, hidden_size = hidden_states.shape
                num_experts = w1.shape[0]
                topk = topk_ids.shape[1]
                intermediate_size = w1.shape[1] // 2

                # Total expert tokens = num_tokens * topk (each token goes to topk experts)
                total_expert_tokens = num_tokens * topk

                # Mark as logged (kept for potential future debug logging)
                if not _fused_experts_debug_logged[0]:
                    _fused_experts_debug_logged[0] = True

                # ADAPTIVE MoE COMBINE LOGIC
                # Decision based on batch size AND gating pattern homogeneity
                def count_unique_experts(topk_ids_tensor):
                    """Count unique experts selected across all tokens."""
                    return topk_ids_tensor.unique().numel()

                unique_experts = count_unique_experts(topk_ids)

                # CRITICAL: Disable combined weight cache for large MoE models (>32 experts)
                # The combined weight cache creates ~1.2GB per layer, causing ~58GB memory
                # pressure for 128-expert models like Qwen3-30B-A3B
                if num_experts > 32:
                    use_combined = False
                    combine_reason = f"large_moe_model(experts={num_experts}>32)"
                # Adaptive thresholds based on batch size and expert diversity
                # 1. Small batches → always combine (cheap, cache hit)
                elif total_expert_tokens <= 256:
                    use_combined = True
                    combine_reason = "small_batch"
                # 2. Medium batches → combine if gating patterns are similar
                elif total_expert_tokens <= 4096:
                    # If batch is "homogeneous" (few unique experts), combine is efficient
                    use_combined = unique_experts <= 2 * topk
                    combine_reason = f"medium_batch(unique={unique_experts},thresh={2*topk})"
                # 3. Large batches → combine only if highly homogeneous
                elif total_expert_tokens <= 8192:
                    use_combined = unique_experts <= topk
                    combine_reason = f"large_batch(unique={unique_experts},thresh={topk})"
                # 4. Huge batches → fallback to non-combined (memory pressure)
                else:
                    use_combined = False
                    combine_reason = "huge_batch"

                # Calculate entropy for diversity metric
                expert_counts = torch.bincount(topk_ids.view(-1), minlength=num_experts).float()
                expert_probs = expert_counts / expert_counts.sum()
                expert_probs = expert_probs[expert_probs > 0]  # Remove zeros for log
                entropy = -(expert_probs * expert_probs.log()).sum().item()
                max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float32)).item()
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0


                w1_elements = w1.shape[1] * w1.shape[2]
                w2_elements = w2.shape[1] * w2.shape[2]

                if use_combined:
                    # Cache key based on weight tensor data_ptr (unique per layer)
                    cache_key = (w1.data_ptr(), w2.data_ptr())
                    if cache_key not in _combined_weights_cache:
                        w_combined = torch.cat([
                            w1.view(num_experts, -1),
                            w2.view(num_experts, -1),
                        ], dim=1).contiguous()
                        _combined_weights_cache[cache_key] = w_combined

                    w_combined = _combined_weights_cache[cache_key]

                # Chunk size tuned for Apple Silicon - 64 gives best perf/memory balance
                CHUNK_SIZE = 64

                # Detailed timing for batch 16+ debugging
                _detail_timing = os.environ.get("MOE_DETAIL_TIMING") and num_tokens >= 16
                if _detail_timing:
                    _t_flatten = _time.perf_counter()

                # Flatten inputs
                flat_topk_ids = topk_ids.view(-1).long()
                flat_topk_weights = topk_weights.view(-1).to(orig_dtype)
                expanded_hidden = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)

                # OPTIMIZATION: Sort by expert ID to improve memory locality for index_select
                # This groups consecutive accesses to the same expert, dramatically improving
                # cache hit rate when gathering 9.4MB weights per expert from 128 experts (1.2GB total)
                sorted_expert_indices = torch.argsort(flat_topk_ids)
                flat_topk_ids = flat_topk_ids[sorted_expert_indices]
                flat_topk_weights = flat_topk_weights[sorted_expert_indices]
                expanded_hidden = expanded_hidden[sorted_expert_indices]
                # Keep inverse permutation to restore original order later
                inverse_indices = torch.argsort(sorted_expert_indices)

                if _detail_timing:
                    torch.mps.synchronize()
                    _flatten_ms = (_time.perf_counter() - _t_flatten) * 1000
                    _t_alloc = _time.perf_counter()

                # Output tensor (in sorted order, will be reordered at the end)
                weighted_output = torch.zeros(
                    total_expert_tokens,
                    hidden_size,
                    dtype=orig_dtype,
                    device=hidden_states.device
                )

                if _detail_timing:
                    torch.mps.synchronize()
                    _alloc_ms = (_time.perf_counter() - _t_alloc) * 1000
                    _index_select_ms = 0.0
                    _bmm1_ms = 0.0
                    _act_ms = 0.0
                    _bmm2_ms = 0.0
                    _store_ms = 0.0

                # Process in chunks for memory efficiency
                for chunk_start in range(0, total_expert_tokens, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_expert_tokens)

                    # Get chunk indices
                    chunk_ids = flat_topk_ids[chunk_start:chunk_end]
                    chunk_weights = flat_topk_weights[chunk_start:chunk_end]
                    chunk_hidden = expanded_hidden[chunk_start:chunk_end]

                    if _detail_timing:
                        _t_op = _time.perf_counter()

                    if use_combined:
                        # OPTIMIZATION: Single gather from pre-combined weights (19% faster)
                        w_chunk = torch.index_select(w_combined, 0, chunk_ids)

                        # Split back into w1 and w2
                        w1_flat = w_chunk[:, :w1_elements]
                        w2_flat = w_chunk[:, w1_elements:]
                        w1_chunk = w1_flat.view(-1, intermediate_size * 2, hidden_size)
                        w2_chunk = w2_flat.view(-1, hidden_size, intermediate_size)
                    else:
                        # Fallback: separate gathers (for large batches during profiling)
                        w1_chunk = torch.index_select(w1, 0, chunk_ids)
                        w2_chunk = torch.index_select(w2, 0, chunk_ids)

                    if _detail_timing:
                        torch.mps.synchronize()
                        _index_select_ms += (_time.perf_counter() - _t_op) * 1000
                        _t_op = _time.perf_counter()

                    # Batched matmul: gate_up projection
                    gate_up = torch.bmm(
                        chunk_hidden.unsqueeze(1),
                        w1_chunk.transpose(1, 2)
                    ).squeeze(1)

                    if _detail_timing:
                        torch.mps.synchronize()
                        _bmm1_ms += (_time.perf_counter() - _t_op) * 1000
                        _t_op = _time.perf_counter()

                    # OPTIMIZATION: Fused gate/up activation (7% faster)
                    if activation == "silu":
                        activated = torch.nn.functional.silu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
                    elif activation == "gelu":
                        activated = torch.nn.functional.gelu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
                    else:
                        activated = gate_up[:, :intermediate_size] * gate_up[:, intermediate_size:]

                    if _detail_timing:
                        torch.mps.synchronize()
                        _act_ms += (_time.perf_counter() - _t_op) * 1000
                        _t_op = _time.perf_counter()

                    # Down projection
                    expert_out = torch.bmm(
                        activated.unsqueeze(1),
                        w2_chunk.transpose(1, 2)
                    ).squeeze(1)

                    if _detail_timing:
                        torch.mps.synchronize()
                        _bmm2_ms += (_time.perf_counter() - _t_op) * 1000
                        _t_op = _time.perf_counter()

                    # Apply weights and store
                    weighted_output[chunk_start:chunk_end] = expert_out * chunk_weights.unsqueeze(-1)

                    if _detail_timing:
                        torch.mps.synchronize()
                        _store_ms += (_time.perf_counter() - _t_op) * 1000

                if _detail_timing:
                    _t_reduce = _time.perf_counter()

                # Restore original token order before reshaping
                weighted_output = weighted_output[inverse_indices]

                # Reshape and sum across topk dimension
                weighted_output = weighted_output.view(num_tokens, topk, hidden_size)
                final_output = weighted_output.sum(dim=1)

                if _detail_timing:
                    torch.mps.synchronize()
                    _reduce_ms = (_time.perf_counter() - _t_reduce) * 1000

                # Profiling (counters kept for potential analysis)
                _moe_profile['call_count'] += 1
                _moe_profile['total_ms'] += (_time.perf_counter() - _t_start) * 1000
                if _moe_profile['call_count'] % 480 == 0:
                    _moe_profile['total_ms'] = 0.0
                    _moe_profile['call_count'] = 0

                return final_output

            # Apply patches
            patched = False
            if 'vllm.model_executor.layers.fused_moe.fused_moe' in sys.modules:
                fused_moe_module = sys.modules['vllm.model_executor.layers.fused_moe.fused_moe']
                fused_moe_module.vllm_topk_softmax = vllm_topk_softmax_mps
                fused_moe_module.fused_experts = fused_experts_mps
                patched = True

            if 'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method' in sys.modules:
                moe_method_module = sys.modules['vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method']
                moe_method_module.fused_experts = fused_experts_mps
                patched = True

            if 'vllm._custom_ops' in sys.modules:
                ops_module = sys.modules['vllm._custom_ops']
                ops_module.topk_softmax = topk_softmax_mps
                patched = True

            if patched:
                logger.info("Applied MoE patches for MPS compatibility")

            # ===== RMSNorm Optimization =====
            # F.rms_norm is 5.8x faster than manual implementation on MPS
            # This saves ~5ms per forward pass (96 RMSNorm calls)
            self._apply_rmsnorm_patch()

        except Exception as e:
            logger.warning(f"Failed to apply MoE patches: {e}")

    def _apply_rmsnorm_patch(self) -> None:
        """Apply optimized RMSNorm for MPS.

        F.rms_norm is 5.8x faster than manual implementation on MPS:
        - Manual: 0.065ms per call
        - F.rms_norm: 0.011ms per call
        - For 96 calls (48 layers × 2): saves ~5ms per forward
        """
        import torch.nn.functional as F

        try:
            from vllm.model_executor.layers.layernorm import RMSNorm

            # Store original for reference
            original_forward_native = RMSNorm.forward_native

            def forward_mps_optimized(
                self,
                x: torch.Tensor,
                residual: torch.Tensor | None = None,
            ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                """Optimized RMSNorm using F.rms_norm for MPS.

                F.rms_norm is implemented in Metal and is 5.8x faster
                than the manual PyTorch implementation.
                """
                # Handle variance_size_override (rare case)
                if self.variance_size_override is not None:
                    return original_forward_native(self, x, residual)

                # Handle residual addition
                if residual is not None:
                    x = x + residual
                    residual_out = x
                else:
                    residual_out = None

                # Use F.rms_norm - much faster on MPS
                # Note: F.rms_norm handles dtype internally
                out = F.rms_norm(
                    x,
                    (self.hidden_size,),
                    self.weight.data if self.has_weight else None,
                    self.variance_epsilon
                )

                if residual_out is None:
                    return out
                else:
                    return out, residual_out

            # Patch the forward_oot method on the class
            RMSNorm.forward_oot = forward_mps_optimized

            # Also need to re-bind _forward_method on existing instances
            # This is done by iterating through model modules
            if hasattr(self, 'model_runner') and self.model_runner is not None:
                model = self.model_runner.get_model()
                patched_count = 0
                for module in model.modules():
                    if isinstance(module, RMSNorm):
                        module._forward_method = module.forward_oot
                        patched_count += 1

                logger.info(f"Applied RMSNorm optimization to {patched_count} instances (5.8x speedup)")
            else:
                logger.info("Applied RMSNorm class patch (5.8x speedup)")

        except Exception as e:
            logger.warning(f"Failed to apply RMSNorm patch: {e}")

    def get_model(self) -> nn.Module:
        """Return the loaded model.

        Returns:
            The PyTorch model instance
        """
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Return supported tasks for this worker.

        Currently only supports text generation on Apple Silicon.

        Returns:
            Tuple of supported tasks (currently only "generate")
        """
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get KV cache specifications for each attention layer.

        Returns specifications for KV cache allocation, including:
        - Block size
        - Number of KV heads
        - Head size
        - Data type

        Returns:
            Dictionary mapping layer names to their KV cache specs
        """
        return self.model_runner.get_kv_cache_spec()

    def _get_available_memory_bytes(self) -> int:
        """Get available unified memory in bytes.

        Apple Silicon uses unified memory shared between CPU and GPU.
        Uses psutil to determine available memory.

        Returns:
            Available memory in bytes
        """
        # Get system memory info using psutil
        mem_info = psutil.virtual_memory()
        available_bytes = mem_info.available

        return available_bytes

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profile memory usage and determine available memory for KV cache.

        This method:
        1. Checks if user specified explicit KV cache memory
        2. Runs a profiling pass to measure model memory usage
        3. Calculates available memory for KV cache based on:
           - Total unified memory
           - Model memory footprint
           - Memory utilization setting
           - Safety buffer

        Memory calculation strategy:
        - Use gpu_memory_utilization config to determine target memory usage
        - Account for model weights and activations
        - Reserve buffer for system stability
        - Use psutil to measure unified memory (shared CPU/GPU on Apple Silicon)

        Returns:
            Available memory for KV cache in bytes

        Note:
            Apple Silicon uses unified memory, so we must be conservative
            to avoid system instability.
        """
        logger.info("Profiling memory usage...")

        GiB = lambda b: b / GiB_bytes

        # Check if user specified explicit KV cache memory
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            logger.info(
                "Using explicit kv_cache_memory_bytes: %.2f GiB",
                GiB(kv_cache_memory_bytes),
            )
            # Still need to run profile to warm up the model
            self.model_runner.profile_run()
            return kv_cache_memory_bytes

        # Get memory before profiling
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        mem_before = self._get_available_memory_bytes()

        # Run profiling pass to measure model memory usage
        self.model_runner.profile_run()

        # Synchronize MPS operations
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

        # Get memory after profiling
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        mem_after = self._get_available_memory_bytes()

        # Calculate activation memory from profiling
        activation_memory_bytes = mem_before - mem_after

        # Get total system memory
        total_memory = psutil.virtual_memory().total

        # Apple Silicon has unified memory shared between CPU and GPU.
        # We must be conservative to avoid system instability.
        # Default max KV cache: 16 GiB (or less if system has less memory)
        # This can be overridden by setting VLLM_CPU_KVCACHE_SPACE environment variable.
        DEFAULT_MAX_KV_CACHE_GIB = 16
        max_kv_cache_bytes = DEFAULT_MAX_KV_CACHE_GIB * GiB_bytes

        # Calculate requested memory based on utilization
        requested_memory_from_utilization = int(
            total_memory * self.cache_config.gpu_memory_utilization
        )

        # Use the smaller of: utilization-based or max KV cache limit
        # Unless user explicitly set VLLM_CPU_KVCACHE_SPACE (then use that)
        from vllm import envs
        if envs.VLLM_CPU_KVCACHE_SPACE is not None:
            # User explicitly set KV cache size - this is the KV cache memory ONLY
            # We need to add model + activation + buffer to get total requested
            kv_cache_bytes = int(envs.VLLM_CPU_KVCACHE_SPACE * GiB_bytes)
            logger.info(
                "Using explicit VLLM_CPU_KVCACHE_SPACE: %.2f GiB for KV cache",
                envs.VLLM_CPU_KVCACHE_SPACE,
            )
            # Return early with explicit KV cache size
            self.available_kv_cache_memory_bytes = kv_cache_bytes
            logger.info(
                "Memory profiling complete:\n"
                "  Total memory: %.2f GiB\n"
                "  Model memory: %.2f GiB\n"
                "  Explicit KV cache: %.2f GiB",
                GiB(total_memory),
                GiB(self.model_memory_usage),
                GiB(kv_cache_bytes),
            )
            return kv_cache_bytes
        else:
            # Use conservative default for Apple Silicon
            requested_memory = min(
                requested_memory_from_utilization,
                max_kv_cache_bytes + self.model_memory_usage,
            )

        # Calculate available memory for KV cache
        # Formula: requested - model_weights - activations - buffer
        # Reserve buffer for system stability (especially important for unified memory)
        buffer_bytes = 500 * (1 << 20)  # 500 MiB buffer

        non_kv_cache_memory = (
            self.model_memory_usage + activation_memory_bytes + buffer_bytes
        )

        self.available_kv_cache_memory_bytes = max(
            requested_memory - non_kv_cache_memory,
            0,
        )

        logger.info(
            "Memory profiling complete:\n"
            "  Total memory: %.2f GiB\n"
            "  Requested memory: %.2f GiB (utilization=%.2f)\n"
            "  Model memory: %.2f GiB\n"
            "  Activation memory: %.2f GiB\n"
            "  Buffer: %.2f GiB\n"
            "  Available for KV cache: %.2f GiB",
            GiB(total_memory),
            GiB(requested_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.model_memory_usage),
            GiB(activation_memory_bytes),
            GiB(buffer_bytes),
            GiB(self.available_kv_cache_memory_bytes),
        )

        return int(self.available_kv_cache_memory_bytes)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize cache configuration.

        Updates the cache config with the determined number of blocks.
        On Apple Silicon, we primarily use "GPU" blocks (unified memory).

        Args:
            num_gpu_blocks: Number of GPU blocks (stored in unified memory)
            num_cpu_blocks: Number of CPU blocks (typically 0 on Apple Silicon)
        """
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        logger.info(
            "Cache config updated: num_gpu_blocks=%d, num_cpu_blocks=%d",
            num_gpu_blocks,
            num_cpu_blocks,
        )

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate KV cache with the specified configuration.

        This is called after determine_available_memory() to actually allocate
        the KV cache tensors based on the computed configuration.

        Args:
            kv_cache_config: Configuration specifying cache size and layout
        """
        logger.info("Initializing KV cache...")
        self.model_runner.initialize_kv_cache(kv_cache_config)
        logger.info("KV cache initialized successfully")

    def compile_or_warm_up_model(self) -> None:
        """Warm up the model for better performance.

        Runs several forward passes with dummy inputs to:
        - Warm up MPS kernels
        - Compile frequently used operations
        - Stabilize memory allocation
        - Optimize MPS graph execution

        Note:
            MPS doesn't support CUDA graphs, but warmup is still beneficial
            for kernel compilation and memory stabilization.
        """
        logger.info("Warming up model...")
        self.model_runner.warmup_model()

        # Reset seed after warmup to ensure reproducibility
        set_random_seed(self.model_config.seed)

        logger.info("Model warmup complete")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        """Execute a single model forward pass.

        This is the main inference method called by the engine.
        It processes scheduled requests and returns sampled tokens.

        Args:
            scheduler_output: Output from scheduler containing:
                - scheduled_new_reqs: New requests to process
                - scheduled_cached_reqs: Cached requests (decode phase)
                - num_scheduled_tokens: Tokens to process per request
                - finished_req_ids: Requests that finished

        Returns:
            ModelRunnerOutput with sampled tokens and metadata,
            or None if no tokens to process

        Note:
            For AppleWorker, sampling is always done inside execute_model
            (not in a separate sample_tokens call).
        """
        # Skip if no tokens to process
        if scheduler_output.total_num_scheduled_tokens == 0:
            return None

        # Execute model forward pass and sample tokens
        output = self.model_runner.execute_model(scheduler_output)

        # Cache output for sample_tokens method
        self._last_model_output = output

        return output

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput",
    ) -> ModelRunnerOutput:
        """Sample tokens when called separately from execute_model.

        For AppleWorker, sampling is done inside execute_model. This method
        is called by vLLM 0.12.0 for grammar/structured output processing.
        Since we already sampled in execute_model, return the cached result.

        Args:
            grammar_output: Grammar output for structured generation

        Returns:
            The ModelRunnerOutput from the last execute_model call
        """
        # Return the last cached output from execute_model
        # Sampling already happened there
        if hasattr(self, '_last_model_output') and self._last_model_output is not None:
            return self._last_model_output

        # If no cached output, return empty output
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block in bytes.

        Used in speculative decoding to determine cache requirements.

        Returns:
            Cache block size in bytes

        Note:
            Calculation: block_size * num_kv_heads * head_size * dtype_size * 2
            (factor of 2 is for both K and V caches)
        """
        # Get KV cache spec to determine block size
        kv_cache_spec = self.get_kv_cache_spec()

        if not kv_cache_spec:
            return 0

        # Get first layer spec to determine block size
        first_spec = next(iter(kv_cache_spec.values()))

        # Calculate block size based on spec
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if isinstance(first_spec, FullAttentionSpec):
            block_size = first_spec.block_size
            num_kv_heads = first_spec.num_kv_heads
            head_size = first_spec.head_size
            dtype = first_spec.dtype

            # Get dtype size in bytes
            dtype_size = torch.tensor([], dtype=dtype).element_size()

            # Calculate total block size (K and V caches)
            return block_size * num_kv_heads * head_size * dtype_size * 2

        return 0

    # LoRA methods - not yet supported on Apple Silicon
    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA adapter.

        Args:
            lora_request: LoRA request to add

        Returns:
            False (LoRA not yet supported on Apple Silicon)
        """
        logger.warning("LoRA is not yet supported on Apple Silicon")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRA adapter.

        Args:
            lora_id: ID of LoRA to remove

        Returns:
            False (LoRA not yet supported on Apple Silicon)
        """
        logger.warning("LoRA is not yet supported on Apple Silicon")
        return False

    def pin_lora(self, lora_id: int) -> bool:
        """Pin a LoRA adapter in memory.

        Args:
            lora_id: ID of LoRA to pin

        Returns:
            False (LoRA not yet supported on Apple Silicon)
        """
        logger.warning("LoRA is not yet supported on Apple Silicon")
        return False

    def list_loras(self) -> set[int]:
        """List active LoRA adapters.

        Returns:
            Empty set (LoRA not yet supported on Apple Silicon)
        """
        return set()

    def check_health(self) -> None:
        """Check worker health.

        Performs basic health checks:
        - Verifies MPS is available
        - Worker is healthy as long as it's running

        Raises:
            RuntimeError: If MPS is not available
        """
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")

    def shutdown(self) -> None:
        """Clean up resources held by the worker.

        Performs cleanup:
        - Clears model runner
        - Empties MPS cache
        - Runs garbage collection
        """
        logger.info("Shutting down AppleWorker...")

        # Clean up model runner
        if self.model_runner is not None:
            self.model_runner = None

        # Clean up device memory
        if self.device is not None and self.device.type == "mps":
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        logger.info("AppleWorker shutdown complete")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration.

        Returns:
            Vocabulary size
        """
        return self.model_config.get_vocab_size()


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "gloo",
) -> None:
    """Initialize the distributed environment for Apple Silicon.

    This function sets up the distributed environment with the following:
    - Batch invariance initialization
    - Custom all-reduce configuration
    - Distributed process group (gloo backend for Apple Silicon)
    - Model parallel groups (TP, PP, CP)

    Args:
        vllm_config: Complete vLLM configuration
        rank: Global rank of this worker
        distributed_init_method: Distributed initialization method (e.g., "env://")
        local_rank: Local rank of this worker
        backend: Backend to use (default: "gloo" for Apple Silicon)

    Note:
        NCCL is not available on Apple Silicon, so we use gloo backend.
        World size is typically 1 for Apple Silicon (no multi-GPU).
    """
    parallel_config = vllm_config.parallel_config

    # Initialize batch invariance
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    init_batch_invariance()

    # Set custom all-reduce (disabled for Apple Silicon)
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # Initialize distributed environment
    init_method = distributed_init_method or "env://"
    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
    )

    # Initialize model parallel groups
    # For Apple Silicon, these are all size 1
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    logger.info(
        "Distributed environment initialized: backend=%s, world_size=%d, rank=%d",
        backend,
        parallel_config.world_size,
        rank,
    )
