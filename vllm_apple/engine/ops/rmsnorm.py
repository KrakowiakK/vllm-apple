# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNorm Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only RMSNorm operations. The operation
encodes RMSNorm computation to a command buffer WITHOUT executing.
Execution happens at step boundary.

Key Principle: Encode-only API. No internal waits.

RMSNorm: x * rsqrt(mean(x^2) + eps) * weight

Usage:
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm

    # Create op during initialization
    rmsnorm = EngineRMSNorm(
        context=engine_context,
        hidden_size=4096,
        eps=1e-6,
        weight_buffer=weight_buffer,
    )

    # Encode RMSNorm (no wait)
    rmsnorm.encode(
        step_ctx=step_ctx,
        input_buffer=hidden_states,
        output_buffer=normed_output,
        num_tokens=num_tokens,
    )
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLSize
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


@dataclass
class RMSNormConfig:
    """Configuration for RMSNorm operation."""
    hidden_size: int
    eps: float = 1e-6
    dtype: EngineDType = EngineDType.FLOAT16


class EngineRMSNorm:
    """Encode-only RMSNorm operation.

    This op encodes RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight

    The operation uses a custom Metal kernel for efficiency.

    Attributes:
        context: MetalEngineContext
        config: RMSNormConfig
        weight_buffer: MTLBuffer containing weights
    """

    # Kernel source for RMSNorm
    KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// RMSNorm kernel
// Each threadgroup processes one token (row)
// Threads within group cooperate on reduction
kernel void rmsnorm_kernel(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float shared_sum[256];

    const uint row_start = token_idx * hidden_size;

    // Step 1: Compute sum of squares (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {
        float val = float(input[row_start + i]);
        local_sum += val * val;
    }

    // Store in shared memory
    shared_sum[thread_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] += shared_sum[thread_idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute rsqrt(mean + eps)
    float mean = shared_sum[0] / float(hidden_size);
    float rsqrt_val = rsqrt(mean + eps);

    // Step 2: Apply normalization and weight
    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {
        float val = float(input[row_start + i]);
        float w = float(weight[i]);
        output[row_start + i] = half(val * rsqrt_val * w);
    }
}

// Fused RMSNorm + residual add kernel
// output = (input * rsqrt(mean(input^2) + eps) * weight) + residual
kernel void rmsnorm_residual_kernel(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const half* residual [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    const uint row_start = token_idx * hidden_size;

    // Compute sum of squares
    float local_sum = 0.0f;
    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {
        float val = float(input[row_start + i]);
        local_sum += val * val;
    }

    shared_sum[thread_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] += shared_sum[thread_idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(hidden_size);
    float rsqrt_val = rsqrt(mean + eps);

    // Apply normalization, weight, and add residual
    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {
        float val = float(input[row_start + i]);
        float w = float(weight[i]);
        float res = float(residual[row_start + i]);
        output[row_start + i] = half(val * rsqrt_val * w + res);
    }
}
"""

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        hidden_size: int,
        eps: float = 1e-6,
        weight_buffer: Optional[Any] = None,  # MTLBuffer
    ):
        """Initialize RMSNorm op.

        Args:
            context: MetalEngineContext
            hidden_size: Hidden dimension
            eps: Epsilon for numerical stability
            weight_buffer: Pre-loaded weight buffer [hidden_size]
        """
        self._context = context
        self.config = RMSNormConfig(
            hidden_size=hidden_size,
            eps=eps,
        )
        self._weight_buffer = weight_buffer
        self._pipeline = None
        self._pipeline_residual = None

        # Compile kernel
        self._compile_kernels()

        logger.info(
            f"EngineRMSNorm initialized: hidden_size={hidden_size}, eps={eps}"
        )

    def _compile_kernels(self) -> None:
        """Compile RMSNorm Metal kernels."""
        # Compile library
        self._context.compile_library("rmsnorm", source_code=self.KERNEL_SOURCE)

        # Get pipelines
        self._pipeline = self._context.get_pipeline("rmsnorm", "rmsnorm_kernel")
        self._pipeline_residual = self._context.get_pipeline(
            "rmsnorm", "rmsnorm_residual_kernel"
        )

    def set_weights(self, weight_buffer: Any) -> None:
        """Set weight buffer.

        Args:
            weight_buffer: Weight buffer [hidden_size]
        """
        self._weight_buffer = weight_buffer

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        input_buffer: Union[EngineTensor, Any],  # MTLBuffer
        output_buffer: Union[EngineTensor, Any],  # MTLBuffer
        num_tokens: int,
        residual_buffer: Optional[Any] = None,  # MTLBuffer for fused residual add
    ) -> None:
        """Encode RMSNorm to command buffer.

        Computes: output = input * rsqrt(mean(input^2) + eps) * weight
        If residual_buffer is provided: output = norm(input) + residual

        Args:
            step_ctx: EngineStepContext with encoder
            input_buffer: Input tensor [num_tokens, hidden_size]
            output_buffer: Output tensor [num_tokens, hidden_size]
            num_tokens: Number of tokens
            residual_buffer: Optional residual to add after norm
        """
        if self._weight_buffer is None:
            raise RuntimeError("Weights not set - call set_weights() first")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Get MTLBuffer from EngineTensor if needed
        if isinstance(input_buffer, EngineTensor):
            input_buf = input_buffer.buffer
            input_offset = input_buffer.offset
        else:
            input_buf = input_buffer
            input_offset = 0

        if isinstance(output_buffer, EngineTensor):
            output_buf = output_buffer.buffer
            output_offset = output_buffer.offset
        else:
            output_buf = output_buffer
            output_offset = 0

        # Choose pipeline (with or without residual)
        if residual_buffer is not None:
            pipeline = self._pipeline_residual
        else:
            pipeline = self._pipeline

        # Get compute encoder
        encoder = step_ctx.get_compute_encoder()

        # Set pipeline
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(input_buf, input_offset, 0)
        encoder.setBuffer_offset_atIndex_(output_buf, output_offset, 1)
        encoder.setBuffer_offset_atIndex_(self._weight_buffer, 0, 2)

        if residual_buffer is not None:
            if isinstance(residual_buffer, EngineTensor):
                res_buf = residual_buffer.buffer
                res_offset = residual_buffer.offset
            else:
                res_buf = residual_buffer
                res_offset = 0
            encoder.setBuffer_offset_atIndex_(res_buf, res_offset, 3)
            # Hidden size and eps at indices 4, 5
            hidden_size_idx = 4
            eps_idx = 5
        else:
            # Hidden size and eps at indices 3, 4
            hidden_size_idx = 3
            eps_idx = 4

        # Set constants via small buffers
        import struct
        hidden_size_data = struct.pack("I", self.config.hidden_size)
        eps_data = struct.pack("f", self.config.eps)

        encoder.setBytes_length_atIndex_(hidden_size_data, 4, hidden_size_idx)
        encoder.setBytes_length_atIndex_(eps_data, 4, eps_idx)

        # Dispatch: one threadgroup per token
        # Use 256 threads per group for reduction
        threads_per_group = min(256, self.config.hidden_size)
        threads_per_group = max(32, threads_per_group)  # At least 32

        thread_groups = MTLSize(num_tokens, 1, 1)
        threads_per_threadgroup = MTLSize(threads_per_group, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, threads_per_threadgroup
        )

    def get_output_size(self, num_tokens: int) -> int:
        """Get required output buffer size in bytes.

        Args:
            num_tokens: Number of tokens

        Returns:
            Size in bytes
        """
        element_size = 2  # float16
        return num_tokens * self.config.hidden_size * element_size

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "hidden_size": self.config.hidden_size,
            "eps": self.config.eps,
        }
