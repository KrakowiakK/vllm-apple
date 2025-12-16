# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elementwise Operations for vLLM-Apple Metal Engine v2.0.

This module provides encode-only elementwise operations:
- Residual addition
- Activation functions (SiLU, GELU)
- RoPE (Rotary Position Embedding)

Key Principle: Encode-only API. No internal waits.

Usage:
    from vllm_apple.engine.ops.elementwise import (
        EngineResidualAdd,
        EngineSiLU,
        EngineRoPE,
    )

    # Residual add
    residual_op = EngineResidualAdd(context)
    residual_op.encode(step_ctx, x, residual, output, num_elements)

    # SiLU activation
    silu_op = EngineSiLU(context)
    silu_op.encode(step_ctx, input, output, num_elements)
"""

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import math

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLSize
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


# Combined kernel source for all elementwise ops
ELEMENTWISE_KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Copy kernel: output = input
kernel void copy_kernel(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = input[idx];
}

// Residual add: output = x + residual
kernel void residual_add_kernel(
    device const half* x [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = x[idx] + residual[idx];
}

// In-place residual add: x += residual
kernel void residual_add_inplace_kernel(
    device half* x [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    x[idx] = x[idx] + residual[idx];
}

// SiLU activation: output = x * sigmoid(x)
kernel void silu_kernel(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    float x = float(input[idx]);
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[idx] = half(x * sigmoid_x);
}

// In-place SiLU
kernel void silu_inplace_kernel(
    device half* data [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
) {
    float x = float(data[idx]);
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    data[idx] = half(x * sigmoid_x);
}

// GELU activation (approximate): output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_kernel(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    float x = float(input[idx]);
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    output[idx] = half(0.5f * x * (1.0f + tanh(inner)));
}

// Fused gate * up with SiLU: output = silu(gate) * up
// For gated MLP: gate_proj and up_proj outputs are multiplied
kernel void silu_mul_kernel(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float g = float(gate[idx]);
    float u = float(up[idx]);
    float silu_g = g / (1.0f + exp(-g));
    output[idx] = half(silu_g * u);
}

// RoPE (Rotary Position Embedding) - Interleaved style (GPT-J)
// Pairs: (x[0], x[1]), (x[2], x[3]), ...
// positions buffer contains the actual position ID for each token
kernel void rope_kernel(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    device const int* positions [[buffer(4)]],  // Position ID for each token
    constant uint& head_size [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& rotary_dim [[buffer(8)]],
    uint3 pos [[thread_position_in_grid]]  // (head_pair_idx, head_idx, token_idx)
) {
    uint pair_idx = pos.x;  // Which pair within rotary_dim/2
    uint head_idx = pos.y;  // Which head
    uint token_idx = pos.z; // Which token in batch

    if (pair_idx >= rotary_dim / 2) return;

    // Get actual position ID for this token (NOT token_idx which is batch order)
    int position_id = positions[token_idx];

    // Get cos/sin for this position and pair
    uint cache_idx = position_id * (rotary_dim / 2) + pair_idx;
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];

    // Apply to query (interleaved: pairs are adjacent)
    if (head_idx < num_heads) {
        uint q_base = token_idx * num_heads * head_size + head_idx * head_size;
        uint q_idx0 = q_base + pair_idx * 2;
        uint q_idx1 = q_base + pair_idx * 2 + 1;

        float q0 = float(query[q_idx0]);
        float q1 = float(query[q_idx1]);

        query[q_idx0] = half(q0 * cos_val - q1 * sin_val);
        query[q_idx1] = half(q0 * sin_val + q1 * cos_val);
    }

    // Apply to key (KV heads may be fewer)
    if (head_idx < num_kv_heads) {
        uint k_base = token_idx * num_kv_heads * head_size + head_idx * head_size;
        uint k_idx0 = k_base + pair_idx * 2;
        uint k_idx1 = k_base + pair_idx * 2 + 1;

        float k0 = float(key[k_idx0]);
        float k1 = float(key[k_idx1]);

        key[k_idx0] = half(k0 * cos_val - k1 * sin_val);
        key[k_idx1] = half(k0 * sin_val + k1 * cos_val);
    }
}

// RoPE (Rotary Position Embedding) - Neox style (GPT-NeoX / Llama / Mistral)
// Pairs: (x[0], x[d/2]), (x[1], x[d/2+1]), ... (split-half)
// This is the DEFAULT style used by Llama, Mistral, and most modern LLMs
kernel void rope_neox_kernel(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    device const int* positions [[buffer(4)]],  // Position ID for each token
    constant uint& head_size [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& rotary_dim [[buffer(8)]],
    uint3 pos [[thread_position_in_grid]]  // (head_pair_idx, head_idx, token_idx)
) {
    uint pair_idx = pos.x;  // Which pair within rotary_dim/2
    uint head_idx = pos.y;  // Which head
    uint token_idx = pos.z; // Which token in batch

    uint half_rotary = rotary_dim / 2;
    if (pair_idx >= half_rotary) return;

    // Get actual position ID for this token (NOT token_idx which is batch order)
    int position_id = positions[token_idx];

    // Get cos/sin for this position and pair
    uint cache_idx = position_id * half_rotary + pair_idx;
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];

    // Apply to query (neox: first half pairs with second half)
    if (head_idx < num_heads) {
        uint q_base = token_idx * num_heads * head_size + head_idx * head_size;
        // Neox style: x[pair_idx] pairs with x[pair_idx + half_rotary]
        uint q_idx0 = q_base + pair_idx;                // First half: 0, 1, 2, ..., d/2-1
        uint q_idx1 = q_base + pair_idx + half_rotary;  // Second half: d/2, d/2+1, ..., d-1

        float q0 = float(query[q_idx0]);
        float q1 = float(query[q_idx1]);

        query[q_idx0] = half(q0 * cos_val - q1 * sin_val);
        query[q_idx1] = half(q0 * sin_val + q1 * cos_val);
    }

    // Apply to key (KV heads may be fewer)
    if (head_idx < num_kv_heads) {
        uint k_base = token_idx * num_kv_heads * head_size + head_idx * head_size;
        // Neox style: x[pair_idx] pairs with x[pair_idx + half_rotary]
        uint k_idx0 = k_base + pair_idx;
        uint k_idx1 = k_base + pair_idx + half_rotary;

        float k0 = float(key[k_idx0]);
        float k1 = float(key[k_idx1]);

        key[k_idx0] = half(k0 * cos_val - k1 * sin_val);
        key[k_idx1] = half(k0 * sin_val + k1 * cos_val);
    }
}

// Scalar multiply: output = input * scalar
kernel void scalar_mul_kernel(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = half(float(input[idx]) * scalar);
}

// Bias add in-place: data[idx] += bias[idx % feature_dim]
kernel void bias_add_inplace_kernel(
    device half* data [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    constant uint& feature_dim [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    data[idx] = data[idx] + bias[idx % feature_dim];
}
"""


class EngineElementwiseOps:
    """Collection of elementwise operations.

    This class compiles all elementwise kernels once and provides
    methods to encode each operation.
    """

    def __init__(self, context: Any):  # MetalEngineContext
        """Initialize elementwise ops.

        Args:
            context: MetalEngineContext
        """
        self._context = context
        self._pipelines: Dict[str, Any] = {}

        # Compile kernels
        self._compile_kernels()

        logger.info("EngineElementwiseOps initialized")

    def _compile_kernels(self) -> None:
        """Compile all elementwise Metal kernels."""
        self._context.compile_library("elementwise", source_code=ELEMENTWISE_KERNEL_SOURCE)

        # Get all pipelines
        kernel_names = [
            "copy_kernel",
            "residual_add_kernel",
            "residual_add_inplace_kernel",
            "silu_kernel",
            "silu_inplace_kernel",
            "gelu_kernel",
            "silu_mul_kernel",
            "rope_kernel",
            "rope_neox_kernel",  # Neox style (default for Llama/Mistral)
            "scalar_mul_kernel",
            "bias_add_inplace_kernel",
        ]
        for name in kernel_names:
            self._pipelines[name] = self._context.get_pipeline("elementwise", name)

    def _dispatch_simple(
        self,
        encoder: Any,
        pipeline_name: str,
        num_elements: int,
    ) -> None:
        """Dispatch a simple elementwise kernel.

        Args:
            encoder: MTLComputeCommandEncoder
            pipeline_name: Name of pipeline to use
            num_elements: Total elements to process
        """
        pipeline = self._pipelines[pipeline_name]
        encoder.setComputePipelineState_(pipeline)

        # Dispatch with optimal thread configuration
        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        thread_group_size = min(max_threads, 256)

        threads = MTLSize(num_elements, 1, 1)
        threads_per_group = MTLSize(thread_group_size, 1, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(threads, threads_per_group)

    def encode_residual_add(
        self,
        step_ctx: Any,
        x: Union[EngineTensor, Any],
        residual: Union[EngineTensor, Any],
        output: Union[EngineTensor, Any],
        num_elements: int,
    ) -> None:
        """Encode residual addition: output = x + residual.

        Args:
            step_ctx: EngineStepContext
            x: Input tensor
            residual: Residual tensor
            output: Output tensor
            num_elements: Number of elements
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        # Get buffers
        x_buf = x.buffer if isinstance(x, EngineTensor) else x
        res_buf = residual.buffer if isinstance(residual, EngineTensor) else residual
        out_buf = output.buffer if isinstance(output, EngineTensor) else output

        x_off = x.offset if isinstance(x, EngineTensor) else 0
        res_off = residual.offset if isinstance(residual, EngineTensor) else 0
        out_off = output.offset if isinstance(output, EngineTensor) else 0

        encoder.setBuffer_offset_atIndex_(x_buf, x_off, 0)
        encoder.setBuffer_offset_atIndex_(res_buf, res_off, 1)
        encoder.setBuffer_offset_atIndex_(out_buf, out_off, 2)

        self._dispatch_simple(encoder, "residual_add_kernel", num_elements)

    def encode_residual_add_inplace(
        self,
        step_ctx: Any,
        x: Union[EngineTensor, Any],
        residual: Union[EngineTensor, Any],
        num_elements: int,
    ) -> None:
        """Encode in-place residual addition: x += residual.

        Args:
            step_ctx: EngineStepContext
            x: Input/output tensor (modified in-place)
            residual: Residual tensor
            num_elements: Number of elements
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        x_buf = x.buffer if isinstance(x, EngineTensor) else x
        res_buf = residual.buffer if isinstance(residual, EngineTensor) else residual

        x_off = x.offset if isinstance(x, EngineTensor) else 0
        res_off = residual.offset if isinstance(residual, EngineTensor) else 0

        encoder.setBuffer_offset_atIndex_(x_buf, x_off, 0)
        encoder.setBuffer_offset_atIndex_(res_buf, res_off, 1)

        self._dispatch_simple(encoder, "residual_add_inplace_kernel", num_elements)

    def encode_copy(
        self,
        step_ctx: Any,
        input_buffer: Union[EngineTensor, Any],
        output: Union[EngineTensor, Any],
        num_elements: int,
    ) -> None:
        """Encode copy operation: output = input.

        Args:
            step_ctx: EngineStepContext
            input_buffer: Source tensor
            output: Destination tensor
            num_elements: Number of elements to copy
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        in_buf = input_buffer.buffer if isinstance(input_buffer, EngineTensor) else input_buffer
        out_buf = output.buffer if isinstance(output, EngineTensor) else output

        in_off = input_buffer.offset if isinstance(input_buffer, EngineTensor) else 0
        out_off = output.offset if isinstance(output, EngineTensor) else 0

        encoder.setBuffer_offset_atIndex_(in_buf, in_off, 0)
        encoder.setBuffer_offset_atIndex_(out_buf, out_off, 1)

        self._dispatch_simple(encoder, "copy_kernel", num_elements)

    def encode_silu(
        self,
        step_ctx: Any,
        input_tensor: Union[EngineTensor, Any],
        output: Union[EngineTensor, Any],
        num_elements: int,
        inplace: bool = False,
    ) -> None:
        """Encode SiLU activation: output = x * sigmoid(x).

        Args:
            step_ctx: EngineStepContext
            input_tensor: Input tensor
            output: Output tensor (ignored if inplace=True)
            num_elements: Number of elements
            inplace: If True, modify input in-place
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        if inplace:
            in_buf = input_tensor.buffer if isinstance(input_tensor, EngineTensor) else input_tensor
            in_off = input_tensor.offset if isinstance(input_tensor, EngineTensor) else 0
            encoder.setBuffer_offset_atIndex_(in_buf, in_off, 0)
            self._dispatch_simple(encoder, "silu_inplace_kernel", num_elements)
        else:
            in_buf = input_tensor.buffer if isinstance(input_tensor, EngineTensor) else input_tensor
            out_buf = output.buffer if isinstance(output, EngineTensor) else output
            in_off = input_tensor.offset if isinstance(input_tensor, EngineTensor) else 0
            out_off = output.offset if isinstance(output, EngineTensor) else 0

            encoder.setBuffer_offset_atIndex_(in_buf, in_off, 0)
            encoder.setBuffer_offset_atIndex_(out_buf, out_off, 1)
            self._dispatch_simple(encoder, "silu_kernel", num_elements)

    def encode_gelu(
        self,
        step_ctx: Any,
        input_tensor: Union[EngineTensor, Any],
        output: Union[EngineTensor, Any],
        num_elements: int,
    ) -> None:
        """Encode GELU activation (approximate).

        Args:
            step_ctx: EngineStepContext
            input_tensor: Input tensor
            output: Output tensor
            num_elements: Number of elements
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        in_buf = input_tensor.buffer if isinstance(input_tensor, EngineTensor) else input_tensor
        out_buf = output.buffer if isinstance(output, EngineTensor) else output
        in_off = input_tensor.offset if isinstance(input_tensor, EngineTensor) else 0
        out_off = output.offset if isinstance(output, EngineTensor) else 0

        encoder.setBuffer_offset_atIndex_(in_buf, in_off, 0)
        encoder.setBuffer_offset_atIndex_(out_buf, out_off, 1)

        self._dispatch_simple(encoder, "gelu_kernel", num_elements)

    def encode_silu_mul(
        self,
        step_ctx: Any,
        gate: Union[EngineTensor, Any],
        up: Union[EngineTensor, Any],
        output: Union[EngineTensor, Any],
        num_elements: int,
    ) -> None:
        """Encode fused SiLU * multiply: output = silu(gate) * up.

        Used in gated MLPs (LLaMA, Qwen, etc).

        Args:
            step_ctx: EngineStepContext
            gate: Gate projection output
            up: Up projection output
            output: Output tensor
            num_elements: Number of elements
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        gate_buf = gate.buffer if isinstance(gate, EngineTensor) else gate
        up_buf = up.buffer if isinstance(up, EngineTensor) else up
        out_buf = output.buffer if isinstance(output, EngineTensor) else output

        gate_off = gate.offset if isinstance(gate, EngineTensor) else 0
        up_off = up.offset if isinstance(up, EngineTensor) else 0
        out_off = output.offset if isinstance(output, EngineTensor) else 0

        encoder.setBuffer_offset_atIndex_(gate_buf, gate_off, 0)
        encoder.setBuffer_offset_atIndex_(up_buf, up_off, 1)
        encoder.setBuffer_offset_atIndex_(out_buf, out_off, 2)

        self._dispatch_simple(encoder, "silu_mul_kernel", num_elements)

    def encode_scalar_mul(
        self,
        step_ctx: Any,
        input_tensor: Union[EngineTensor, Any],
        output: Union[EngineTensor, Any],
        scalar: float,
        num_elements: int,
    ) -> None:
        """Encode scalar multiplication: output = input * scalar.

        Args:
            step_ctx: EngineStepContext
            input_tensor: Input tensor
            output: Output tensor
            scalar: Scalar value
            num_elements: Number of elements
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()

        in_buf = input_tensor.buffer if isinstance(input_tensor, EngineTensor) else input_tensor
        out_buf = output.buffer if isinstance(output, EngineTensor) else output
        in_off = input_tensor.offset if isinstance(input_tensor, EngineTensor) else 0
        out_off = output.offset if isinstance(output, EngineTensor) else 0

        import struct
        scalar_data = struct.pack("f", scalar)

        encoder.setBuffer_offset_atIndex_(in_buf, in_off, 0)
        encoder.setBuffer_offset_atIndex_(out_buf, out_off, 1)
        encoder.setBytes_length_atIndex_(scalar_data, 4, 2)

        self._dispatch_simple(encoder, "scalar_mul_kernel", num_elements)

    def encode_bias_add_inplace(
        self,
        step_ctx: Any,
        data: Union[EngineTensor, Any],
        bias: Union[EngineTensor, Any],
        feature_dim: int,
        num_elements: int,
    ) -> None:
        """Encode in-place bias addition (broadcasted).
        
        data[i] += bias[i % feature_dim]
        
        Args:
            step_ctx: EngineStepContext
            data: Input/Output tensor (flattened)
            bias: Bias tensor [feature_dim]
            feature_dim: Dimension of bias vector
            num_elements: Total elements in data
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        encoder = step_ctx.get_compute_encoder()
        
        d_buf = data.buffer if isinstance(data, EngineTensor) else data
        b_buf = bias.buffer if isinstance(bias, EngineTensor) else bias
        
        d_off = data.offset if isinstance(data, EngineTensor) else 0
        b_off = bias.offset if isinstance(bias, EngineTensor) else 0
        
        import struct
        dim_bytes = struct.pack("I", feature_dim)
        
        encoder.setBuffer_offset_atIndex_(d_buf, d_off, 0)
        encoder.setBuffer_offset_atIndex_(b_buf, b_off, 1)
        encoder.setBytes_length_atIndex_(dim_bytes, 4, 2)
        
        self._dispatch_simple(encoder, "bias_add_inplace_kernel", num_elements)


class EngineRoPE:
    """Rotary Position Embedding operation.

    Applies rotation to Q and K based on position.

    Supports two RoPE styles:
    - is_neox_style=True (default): GPT-NeoX/Llama/Mistral style
      Pairs: (x[0], x[d/2]), (x[1], x[d/2+1]), ... (split-half)
    - is_neox_style=False: GPT-J style
      Pairs: (x[0], x[1]), (x[2], x[3]), ... (interleaved)

    Most modern LLMs (Llama, Mistral, etc.) use is_neox_style=True.
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: Optional[int] = None,
        max_position: int = 8192,
        base: float = 10000.0,
        is_neox_style: bool = True,  # Default True for Llama/Mistral compatibility
    ):
        """Initialize RoPE.

        Args:
            context: MetalEngineContext
            head_size: Dimension per head
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads
            rotary_dim: Dimension to apply rotation (default: head_size)
            max_position: Maximum sequence length
            base: RoPE base frequency
            is_neox_style: If True, use GPT-NeoX/Llama style (split-half).
                          If False, use GPT-J style (interleaved).
                          Default is True for compatibility with most LLMs.
        """
        self._context = context
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = rotary_dim or head_size
        self.max_position = max_position
        self.base = base
        self.is_neox_style = is_neox_style

        # Precompute cos/sin cache
        self._cos_cache = None
        self._sin_cache = None
        self._precompute_cache()

        # Compile kernel (use shared elementwise library)
        context.compile_library("elementwise", source_code=ELEMENTWISE_KERNEL_SOURCE)

        # Select kernel based on RoPE style
        if self.is_neox_style:
            kernel_name = "rope_neox_kernel"
        else:
            kernel_name = "rope_kernel"
        self._pipeline = context.get_pipeline("elementwise", kernel_name)

        style_name = "neox (split-half)" if is_neox_style else "interleaved (GPT-J)"
        logger.info(
            f"EngineRoPE initialized: head_size={head_size}, "
            f"rotary_dim={self.rotary_dim}, max_pos={max_position}, "
            f"style={style_name}"
        )

    def _precompute_cache(self) -> None:
        """Precompute cos/sin cache for all positions."""
        import numpy as np

        half_dim = self.rotary_dim // 2
        inv_freq = 1.0 / (
            self.base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
        )

        positions = np.arange(self.max_position, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)  # [max_pos, half_dim]

        cos_cache = np.cos(freqs).astype(np.float32)
        sin_cache = np.sin(freqs).astype(np.float32)

        # Upload to GPU
        self._cos_cache = self._context.create_buffer_from_bytes(cos_cache.tobytes())
        self._sin_cache = self._context.create_buffer_from_bytes(sin_cache.tobytes())

    def encode(
        self,
        step_ctx: Any,
        query: Union[EngineTensor, Any],
        key: Union[EngineTensor, Any],
        positions: Union[EngineTensor, Any],  # [num_tokens] position indices (int32)
        num_tokens: int,
        max_position_in_batch: Optional[int] = None,  # For bounds checking
    ) -> None:
        """Encode RoPE to command buffer.

        Modifies query and key in-place with rotary embeddings.

        Args:
            step_ctx: EngineStepContext
            query: Query tensor [num_tokens, num_heads, head_size]
            key: Key tensor [num_tokens, num_kv_heads, head_size]
            positions: Position indices [num_tokens] - MUST be int32 dtype
            num_tokens: Number of tokens
            max_position_in_batch: Maximum position ID in this batch (for bounds check)

        Note:
            The positions buffer must contain int32 values. The kernel reads these
            as position IDs to index into the cos/sin cache. Position IDs must be
            less than max_position (checked if max_position_in_batch is provided).
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Range check: position_id must be < max_position to stay within cache bounds
        if max_position_in_batch is not None and max_position_in_batch >= self.max_position:
            raise ValueError(
                f"Position ID {max_position_in_batch} exceeds max_position {self.max_position}. "
                f"Increase max_position in EngineRoPE constructor."
            )

        encoder = step_ctx.get_compute_encoder()
        encoder.setComputePipelineState_(self._pipeline)

        # Get buffers
        q_buf = query.buffer if isinstance(query, EngineTensor) else query
        k_buf = key.buffer if isinstance(key, EngineTensor) else key
        pos_buf = positions.buffer if isinstance(positions, EngineTensor) else positions

        q_off = query.offset if isinstance(query, EngineTensor) else 0
        k_off = key.offset if isinstance(key, EngineTensor) else 0
        pos_off = positions.offset if isinstance(positions, EngineTensor) else 0

        encoder.setBuffer_offset_atIndex_(q_buf, q_off, 0)
        encoder.setBuffer_offset_atIndex_(k_buf, k_off, 1)
        encoder.setBuffer_offset_atIndex_(self._cos_cache, 0, 2)
        encoder.setBuffer_offset_atIndex_(self._sin_cache, 0, 3)
        encoder.setBuffer_offset_atIndex_(pos_buf, pos_off, 4)  # Positions buffer

        # Set constants (indices shifted by 1 due to positions buffer)
        import struct
        encoder.setBytes_length_atIndex_(struct.pack("I", self.head_size), 4, 5)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.num_heads), 4, 6)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.num_kv_heads), 4, 7)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.rotary_dim), 4, 8)

        # Dispatch
        # Grid: (rotary_dim/2, max(num_heads, num_kv_heads), num_tokens)
        max_heads = max(self.num_heads, self.num_kv_heads)
        threads = MTLSize(self.rotary_dim // 2, max_heads, num_tokens)
        threads_per_group = MTLSize(min(32, self.rotary_dim // 2), 1, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(threads, threads_per_group)
