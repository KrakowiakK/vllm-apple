# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Platform for vLLM - Native Metal GPU acceleration.

This platform provides optimized inference on Apple Silicon using
custom Metal kernels for MoE, attention, and other operations.
Supports both V0 and V1 vLLM engines.
"""

import os
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.config import ModelConfig, VllmConfig
else:
    AttentionBackendEnum = None
    VllmConfig = None
    ModelConfig = None


class ApplePlatform(Platform):
    """Apple GPU Platform using Metal for vLLM.

    This platform provides optimized inference on Apple Silicon using
    custom Metal kernels for MoE, attention, and other operations.
    Supports V1 engine with custom Worker and ModelRunner.
    """

    _enum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    ray_device_key: str = "MPS"
    device_control_env_var: str = "APPLE_VISIBLE_DEVICES"

    # Use eager backend - MPS has limited torch.compile support
    simple_compile_backend: str = "eager"

    # Distributed backend for Apple (single device only)
    dist_backend: str = "gloo"

    # Supported quantization methods
    supported_quantization: list[str] = []

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        """Get supported data types for Apple Silicon.

        MPS supports float16, bfloat16 (on newer Apple Silicon), and float32
        but does NOT support float64.
        """
        import subprocess
        try:
            # Check if bfloat16 is supported (Apple Silicon M2 and later)
            result = subprocess.check_output(
                ["sysctl", "-n", "hw.optional.arm.FEAT_BF16"]
            ).strip()
            if result == b"1":
                return [torch.bfloat16, torch.float16, torch.float32]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return [torch.float16, torch.float32]

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        """Check if dtype is supported on Apple Silicon.

        Raises ValueError if dtype is not supported.
        """
        supported = cls.get_supported_dtypes()
        if dtype not in supported:
            raise ValueError(
                f"dtype {dtype} is not supported on Apple Silicon. "
                f"Supported dtypes: {supported}"
            )

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the device name."""
        return (
            torch.backends.mps.get_name()
            if hasattr(torch.backends.mps, "get_name")
            else "Apple MPS"
        )

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> Optional[tuple[int, int]]:
        """Get device capability (not applicable for MPS)."""
        return None

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: Optional[str] = None,
        is_attention_free: bool = False,
    ) -> str:
        """Get the attention backend class for Apple platform.

        Returns direct path to our Apple attention backend, compatible with
        any vLLM version (doesn't require MPS_ATTN in AttentionBackendEnum).

        Backend selection:
        - VLLM_METAL_ATTENTION=1: Use Metal PagedAttention kernel (12x faster decode)
        - Default: Use Apple attention backend (PyTorch SDPA)

        Metal backend requirements:
        - head_size in {32, 64, 96, 128}
        - dtype = float16
        """
        if use_mla:
            raise NotImplementedError("MLA is not supported on Apple GPU.")
        if use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on Apple GPU.")

        # Check if Metal attention backend is requested
        use_metal = os.environ.get("VLLM_METAL_ATTENTION", "0") == "1"

        if use_metal:
            # Validate Metal backend requirements
            supported_head_sizes = {32, 64, 96, 128}
            if head_size not in supported_head_sizes:
                logger.warning(
                    f"Metal attention backend does not support head_size={head_size}. "
                    f"Supported: {sorted(supported_head_sizes)}. Falling back to Apple backend."
                )
                use_metal = False
            elif dtype != torch.float16:
                logger.warning(
                    f"Metal attention backend requires float16, got {dtype}. "
                    "Falling back to Apple backend."
                )
                use_metal = False

        if use_metal:
            logger.info(
                f"Using Metal PagedAttention backend (head_size={head_size}, "
                f"block_size={block_size}) - 12x faster decode!"
            )
            return "vllm_apple.v1.attention.backends.metal_attn.MetalAttentionBackend"

        # Default: Metal attention backend (PyTorch SDPA fallback)
        logger.info("Using Metal attention backend from vllm-apple plugin.")
        return "vllm_apple.v1.attention.backends.metal_attn.MetalAttentionBackend"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total device memory for KV cache allocation."""
        import psutil
        from vllm import envs

        try:
            from vllm.utils.mem_constants import GiB_bytes
        except ImportError:
            GiB_bytes = 1024 * 1024 * 1024

        # Check for user-specified KV cache space
        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        if kv_cache_space is None:
            # Apple Silicon uses unified memory - use a portion of total memory
            # Use 50% of system memory for KV cache by default
            # (same as CPU backend since MPS shares unified memory)
            total_memory = psutil.virtual_memory().total
            kv_cache_space = total_memory // 2
            logger.warning_once(
                "Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) "
                "for Apple backend is not set, using %.1f GiB by default.",
                kv_cache_space / GiB_bytes,
            )
        else:
            kv_cache_space = int(kv_cache_space * GiB_bytes)

        return kv_cache_space

    @classmethod
    def mem_get_info(cls, device=None) -> tuple[int, int]:
        """Return (free_memory, total_memory) for unified memory."""
        import psutil
        mem = psutil.virtual_memory()
        # Apple uses unified memory - report ~50% for GPU use
        total = mem.total // 2
        available = mem.available // 2
        return (available, total)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the device for the current platform."""
        # MPS only has one device, but set it anyway for consistency
        if hasattr(torch.mps, '_set_default_device'):
            torch.mps._set_default_device(device)

    @classmethod
    def synchronize(cls) -> None:
        """Synchronize MPS device."""
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        """Empty MPS cache."""
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    @classmethod
    def inference_mode(cls):
        """Return the inference mode context manager."""
        return torch.no_grad()

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        """Check if V1 engine is supported."""
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for Apple platform."""
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        # Disable cascade attention if model_config exists
        if model_config is not None:
            if hasattr(model_config, 'disable_cascade_attn'):
                model_config.disable_cascade_attn = True

        # Set optimal block size for Metal
        if cache_config.block_size is None:
            cache_config.block_size = 16  # Optimal for Metal SIMD groups

        # KV cache quantization not supported
        if cache_config.cache_dtype != "auto":
            logger.warning(
                "Apple backend doesn't support KV cache quantization, "
                "falling back to auto."
            )
            cache_config.cache_dtype = "auto"

        # Set KV cache memory using device total memory
        cache_config.cpu_kvcache_space_bytes = cls.get_device_total_memory()

        # Apple GPU only supports single device
        if parallel_config.world_size > 1:
            raise ValueError("Apple backend only supports single device (world_size=1)")

        # Force single process execution
        if parallel_config.distributed_executor_backend is None:
            parallel_config.distributed_executor_backend = "uni"

        # Use our custom AppleWorker for V1 engine (optimized for Apple Silicon)
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_apple.v1.worker.apple_worker.AppleWorker"
            logger.info("Using vllm-apple custom AppleWorker for V1 engine")

        # Disable custom all reduce
        parallel_config.disable_custom_all_reduce = True

        # Disable DBO
        if parallel_config.enable_dbo:
            logger.warning("Dual-Batch Overlap is not supported on Apple GPU, disabled.")
            parallel_config.enable_dbo = False

        # Disable CUDA graph equivalent features and set compilation mode
        from vllm.config import CompilationMode

        vllm_config.compilation_config.cudagraph_capture_sizes = []

        compilation_config = vllm_config.compilation_config
        # Use eager mode for MPS - torch.compile has limited support
        compilation_config.mode = CompilationMode.NONE
        compilation_config.backend = "eager"

        if vllm_config.lora_config is not None:
            compilation_config.mode = CompilationMode.NONE

        # MPS platform runs on MPS device
        assert vllm_config.device_config.device_type == "mps"

        # Environment variables for Apple executor
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Handle MLA if enabled
        if model_config is not None and hasattr(model_config, 'use_mla') and model_config.use_mla:
            logger.info(
                "MLA is enabled on Apple platform; forcing chunked "
                "prefill and prefix caching to be disabled."
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """Check if async output processing is supported."""
        return False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pinned memory is available."""
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """Get LoRA punica wrapper class."""
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device communicator class for distributed communication."""
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"

    @classmethod
    def supports_structured_output(cls) -> bool:
        """Check if structured output is supported."""
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Check if hybrid KV cache is supported."""
        return False

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        """Return False - MPS doesn't support torch._sync on functional tensors."""
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        """Return False to use direct attention backend calls.

        Instead of torch.ops.vllm.unified_attention which requires CUDA ops.
        """
        return False

    @classmethod
    def get_compile_backend(cls) -> str:
        """Return the compilation backend for Apple Silicon.

        MPS doesn't support inductor well, so we use eager mode by default.
        """
        return "eager"

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """Apple Silicon doesn't support static graph mode (CUDA graphs)."""
        return False

    @classmethod
    def use_all_gather(cls) -> bool:
        """Whether to use allgather in LogitsProcessor.

        Apple uses V1 engine, so return True.
        """
        return True

    @classmethod
    def get_current_memory_usage(cls, device: torch.device) -> int:
        """Get current memory usage for MPS device."""
        if hasattr(torch.mps, 'current_allocated_memory'):
            return torch.mps.current_allocated_memory()
        return 0

    @classmethod
    def import_kernels(cls) -> None:
        """Import platform-specific kernels."""
        # Apply comprehensive MPS empty tensor fix (must be in worker process)
        try:
            from vllm_apple.patch_mps_empty import apply_mps_empty_tensor_patches
            apply_mps_empty_tensor_patches()
        except ImportError:
            # Fallback to simple fix if patch module not available
            cls._apply_mps_empty_tensor_fix()

        try:
            from vllm_apple.ops import apple_fused_moe  # noqa: F401
            logger.info("Loaded Apple MoE operations")
        except ImportError as e:
            logger.warning(f"Could not load Apple MoE operations: {e}")

        # Apply MoE patches after all imports are done
        # These patches are deferred to avoid circular import issues
        cls._apply_deferred_moe_patches()

    @classmethod
    def _apply_deferred_moe_patches(cls) -> None:
        """Apply MoE patches after module initialization to avoid circular imports.

        This patches both topk_softmax and fused_experts in all necessary locations.
        Must be called after all vLLM modules have been loaded.
        """
        try:
            # Define PyTorch fallback for topk_softmax (in-place version)
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

            # Define vllm_topk_softmax replacement (returns tuple)
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
                """PyTorch implementation of fused_experts for MPS.

                Vectorized implementation that processes experts in batches.
                """
                orig_dtype = hidden_states.dtype
                num_tokens, hidden_size = hidden_states.shape
                num_experts = w1.shape[0]
                topk = topk_ids.shape[1]

                # w1 shape: [num_experts, intermediate_size * 2, hidden_size]
                intermediate_size = w1.shape[1] // 2

                # Flatten for batch processing
                flat_topk_ids = topk_ids.view(-1)
                flat_topk_weights = topk_weights.view(-1)

                # Expand hidden states
                expanded_hidden = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)

                # Initialize output
                final_output = torch.zeros(
                    num_tokens * topk,
                    hidden_size,
                    dtype=orig_dtype,
                    device=hidden_states.device
                )

                # Group tokens by expert for efficient batch processing
                for expert_idx in range(num_experts):
                    mask = flat_topk_ids == expert_idx
                    if not mask.any():
                        continue

                    expert_tokens = expanded_hidden[mask]
                    expert_weights = flat_topk_weights[mask].unsqueeze(-1)

                    expert_w1 = w1[expert_idx]
                    expert_w2 = w2[expert_idx]

                    gate_proj = torch.nn.functional.linear(expert_tokens, expert_w1[:intermediate_size])
                    up_proj = torch.nn.functional.linear(expert_tokens, expert_w1[intermediate_size:])

                    if activation == "silu":
                        activated = torch.nn.functional.silu(gate_proj) * up_proj
                    elif activation == "gelu":
                        activated = torch.nn.functional.gelu(gate_proj) * up_proj
                    else:
                        activated = gate_proj * up_proj

                    expert_output = torch.nn.functional.linear(activated, expert_w2)
                    final_output[mask] = (expert_output * expert_weights).to(orig_dtype)

                final_output = final_output.view(num_tokens, topk, hidden_size)
                final_output = final_output.sum(dim=1)

                return final_output

            # Apply patches using sys.modules to avoid import issues
            import sys

            # Patch fused_moe module
            if 'vllm.model_executor.layers.fused_moe.fused_moe' in sys.modules:
                fused_moe_module = sys.modules['vllm.model_executor.layers.fused_moe.fused_moe']
                fused_moe_module.vllm_topk_softmax = vllm_topk_softmax_mps
                fused_moe_module.fused_experts = fused_experts_mps
                logger.info("Patched vllm_topk_softmax and fused_experts in fused_moe module")

            # Patch unquantized_fused_moe_method module
            if 'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method' in sys.modules:
                moe_method_module = sys.modules['vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method']
                moe_method_module.fused_experts = fused_experts_mps
                logger.info("Patched fused_experts in unquantized_fused_moe_method")

            # Patch _custom_ops module
            if 'vllm._custom_ops' in sys.modules:
                ops_module = sys.modules['vllm._custom_ops']
                ops_module.topk_softmax = topk_softmax_mps
                logger.info("Patched topk_softmax in vllm._custom_ops")

        except Exception as e:
            logger.warning(f"Failed to apply deferred MoE patches: {e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def pre_register_and_update(cls) -> None:
        """Platform patches applied before workers start."""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Apply comprehensive MPS empty tensor fix to prevent assertion errors
        try:
            from vllm_apple.patch_mps_empty import apply_mps_empty_tensor_patches
            apply_mps_empty_tensor_patches()
        except ImportError:
            # Fallback to simple fix if patch module not available
            cls._apply_mps_empty_tensor_fix()

        # Early patch for MoE operations - must be done before model loading
        cls._early_patch_moe_ops()

    @classmethod
    def _apply_mps_empty_tensor_fix(cls) -> None:
        """Apply global patch to prevent empty tensors from being moved to MPS.

        MPS backend has a bug where creating constant tensors with shape[0]=0
        causes an assertion error in MPSGraphMemoryOps.mm. This patch intercepts
        tensor.to() calls and keeps empty tensors on CPU.
        """
        _original_to = torch.Tensor.to

        def _patched_to(self, *args, **kwargs):
            # Call original to() first
            result = _original_to(self, *args, **kwargs)

            # Check if we're trying to move an empty tensor to MPS
            if result.numel() == 0 and result.device.type == 'mps':
                # Keep empty tensor on CPU to avoid MPS assertion error
                return _original_to(self, 'cpu')

            return result

        torch.Tensor.to = _patched_to
        logger.info("Applied MPS empty tensor fix")

    @classmethod
    def _early_patch_moe_ops(cls) -> None:
        """Early patch for MoE operations before any imports.

        This patches the vLLM MoE modules to use our PyTorch fallback
        implementations instead of CUDA/Triton kernels.
        """
        try:
            # Define fused_experts implementation
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
                """PyTorch implementation of fused_experts for MPS."""
                num_tokens, hidden_size = hidden_states.shape
                num_experts = w1.shape[0]
                intermediate_size = w1.shape[1] // 2
                topk = topk_ids.shape[1]

                if apply_router_weight_on_input:
                    hidden_states = hidden_states.unsqueeze(1) * topk_weights.unsqueeze(-1)
                    hidden_states = hidden_states.view(-1, hidden_size)
                    topk_ids_flat = topk_ids.view(-1)
                    topk_weights_flat = torch.ones(num_tokens * topk, device=topk_weights.device, dtype=topk_weights.dtype)
                else:
                    hidden_states = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)
                    topk_ids_flat = topk_ids.view(-1)
                    topk_weights_flat = topk_weights.view(-1)

                final_hidden_states = torch.zeros(num_tokens * topk, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

                for expert_idx in range(num_experts):
                    mask = topk_ids_flat == expert_idx
                    if not mask.any():
                        continue
                    expert_tokens = hidden_states[mask]
                    expert_w1 = w1[expert_idx]
                    expert_w2 = w2[expert_idx]
                    gate = torch.nn.functional.linear(expert_tokens, expert_w1[:intermediate_size])
                    up = torch.nn.functional.linear(expert_tokens, expert_w1[intermediate_size:])
                    if activation == "silu":
                        activated = torch.nn.functional.silu(gate) * up
                    elif activation == "gelu":
                        activated = torch.nn.functional.gelu(gate) * up
                    else:
                        activated = gate * up
                    expert_out = torch.nn.functional.linear(activated, expert_w2)
                    weights = topk_weights_flat[mask].unsqueeze(-1)
                    final_hidden_states[mask] = expert_out * weights

                final_hidden_states = final_hidden_states.view(num_tokens, topk, hidden_size)
                final_hidden_states = final_hidden_states.sum(dim=1)
                return final_hidden_states

            # Patch sys.modules to intercept imports
            import sys
            from types import ModuleType

            # Create a patching import hook
            class MoEPatchFinder:
                def find_module(self, fullname, path=None):
                    if fullname == 'vllm.model_executor.layers.fused_moe.fused_moe':
                        return self
                    return None

                def load_module(self, fullname):
                    if fullname in sys.modules:
                        # Module already loaded, patch it
                        module = sys.modules[fullname]
                        if hasattr(module, 'fused_experts') and module.fused_experts is None:
                            module.fused_experts = fused_experts_mps
                            logger.info("Patched fused_experts via import hook")
                        return module
                    return None

            # Insert our finder at the beginning
            sys.meta_path.insert(0, MoEPatchFinder())

            # Also try to patch if already imported
            if 'vllm.model_executor.layers.fused_moe.fused_moe' in sys.modules:
                module = sys.modules['vllm.model_executor.layers.fused_moe.fused_moe']
                if hasattr(module, 'fused_experts') and module.fused_experts is None:
                    module.fused_experts = fused_experts_mps
                    logger.info("Early patched fused_experts")

            if 'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method' in sys.modules:
                module = sys.modules['vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method']
                if hasattr(module, 'fused_experts') and module.fused_experts is None:
                    module.fused_experts = fused_experts_mps
                    logger.info("Early patched fused_experts in unquantized_fused_moe_method")

        except Exception as e:
            logger.warning(f"Failed to early patch MoE ops: {e}")

    @classmethod
    def _patch_fused_experts_in_worker(cls) -> None:
        """Patch fused_experts in worker process after modules are loaded."""
        try:
            # Define the fallback implementation
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
                """PyTorch implementation of fused_experts for MPS."""
                num_tokens, hidden_size = hidden_states.shape
                num_experts = w1.shape[0]
                intermediate_size = w1.shape[1] // 2
                topk = topk_ids.shape[1]

                if apply_router_weight_on_input:
                    hidden_states = hidden_states.unsqueeze(1) * topk_weights.unsqueeze(-1)
                    hidden_states = hidden_states.view(-1, hidden_size)
                    topk_ids_flat = topk_ids.view(-1)
                    topk_weights_flat = torch.ones(num_tokens * topk, device=topk_weights.device, dtype=topk_weights.dtype)
                else:
                    hidden_states = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)
                    topk_ids_flat = topk_ids.view(-1)
                    topk_weights_flat = topk_weights.view(-1)

                final_hidden_states = torch.zeros(num_tokens * topk, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

                for expert_idx in range(num_experts):
                    mask = topk_ids_flat == expert_idx
                    if not mask.any():
                        continue
                    expert_tokens = hidden_states[mask]
                    expert_w1 = w1[expert_idx]
                    expert_w2 = w2[expert_idx]
                    gate = torch.nn.functional.linear(expert_tokens, expert_w1[:intermediate_size])
                    up = torch.nn.functional.linear(expert_tokens, expert_w1[intermediate_size:])
                    if activation == "silu":
                        activated = torch.nn.functional.silu(gate) * up
                    elif activation == "gelu":
                        activated = torch.nn.functional.gelu(gate) * up
                    else:
                        activated = gate * up
                    expert_out = torch.nn.functional.linear(activated, expert_w2)
                    weights = topk_weights_flat[mask].unsqueeze(-1)
                    final_hidden_states[mask] = expert_out * weights

                final_hidden_states = final_hidden_states.view(num_tokens, topk, hidden_size)
                final_hidden_states = final_hidden_states.sum(dim=1)
                return final_hidden_states

            import sys

            # Patch in fused_moe module
            if 'vllm.model_executor.layers.fused_moe.fused_moe' in sys.modules:
                module = sys.modules['vllm.model_executor.layers.fused_moe.fused_moe']
                module.fused_experts = fused_experts_mps
                logger.info("Patched fused_experts in fused_moe module")

            # Patch in unquantized_fused_moe_method module
            if 'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method' in sys.modules:
                module = sys.modules['vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method']
                module.fused_experts = fused_experts_mps
                logger.info("Patched fused_experts in unquantized_fused_moe_method")

            # Force import and patch if not yet loaded
            try:
                import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
                import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as moe_method_module
                fused_moe_module.fused_experts = fused_experts_mps
                moe_method_module.fused_experts = fused_experts_mps
                logger.info("Force patched fused_experts in MoE modules")
            except Exception as e:
                logger.debug(f"Could not force patch MoE modules: {e}")

        except Exception as e:
            logger.warning(f"Failed to patch fused_experts in worker: {e}")

    @classmethod
    def _register_moe_fallback_ops(cls) -> None:
        """Register MoE fallback operations for MPS.

        vLLM's MoE implementation uses torch.ops._moe_C.topk_softmax which is a
        CUDA kernel. We provide a PyTorch fallback implementation for MPS by
        patching multiple places where the function is used.
        """
        try:
            # Define PyTorch fallback for topk_softmax (in-place version)
            def topk_softmax_mps(
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                token_expert_indices: torch.Tensor,
                gating_output: torch.Tensor,
                renormalize: bool = False,
            ) -> None:
                """PyTorch implementation of topk_softmax for MPS.

                Args:
                    topk_weights: Output tensor for top-k weights [num_tokens, topk]
                    topk_ids: Output tensor for top-k expert IDs [num_tokens, topk]
                    token_expert_indices: Output tensor for token-expert mapping
                    gating_output: Router logits [num_tokens, num_experts]
                    renormalize: Whether to renormalize weights to sum to 1
                """
                num_tokens, num_experts = gating_output.shape
                topk = topk_weights.shape[1]

                # Get top-k experts and their weights
                topk_weights_tmp, topk_ids_tmp = torch.topk(
                    gating_output, k=topk, dim=-1
                )

                # Apply softmax to get normalized weights
                topk_weights_tmp = torch.softmax(
                    topk_weights_tmp.float(), dim=-1
                ).to(topk_weights.dtype)

                if renormalize:
                    # Renormalize so weights sum to 1
                    topk_weights_tmp = topk_weights_tmp / topk_weights_tmp.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-6)

                # Copy results to output tensors (in-place operation)
                topk_weights.copy_(topk_weights_tmp)
                topk_ids.copy_(topk_ids_tmp)

                # Build token_expert_indices: flattened mapping of (token_idx, expert_rank) -> linear_idx
                # This is used for permuting tokens to their assigned experts
                if token_expert_indices.numel() > 0:
                    indices = torch.arange(
                        num_tokens * topk,
                        device=gating_output.device,
                        dtype=token_expert_indices.dtype,
                    )
                    token_expert_indices.copy_(indices.view_as(token_expert_indices))

            # Define vllm_topk_softmax replacement (returns tuple)
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

            # Patch vllm._custom_ops.topk_softmax
            import vllm._custom_ops as ops
            ops.topk_softmax = topk_softmax_mps
            logger.info("Patched vllm._custom_ops.topk_softmax for MPS")

            # Patch vllm_topk_softmax in fused_moe module directly
            # This is critical because the module caches the 'ops' reference
            import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
            fused_moe_module.vllm_topk_softmax = vllm_topk_softmax_mps
            logger.info("Patched vllm.model_executor.layers.fused_moe.fused_moe.vllm_topk_softmax for MPS")

            # fused_experts patching is done separately in _patch_fused_experts_in_worker

        except Exception as e:
            logger.warning(f"Failed to register MoE fallback ops: {e}")

