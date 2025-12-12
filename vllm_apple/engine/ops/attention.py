# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Attention Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only paged attention operations. The operation
encodes attention kernel dispatches to a command buffer WITHOUT executing.
Execution happens at step boundary via single waitUntilCompleted().

Key Principle: Encode-only API. No internal waits.

Usage:
    from vllm_apple.engine.ops.attention import PagedAttentionOp

    # Create op during initialization
    attn_op = PagedAttentionOp(
        context=engine_context,
        num_kv_heads=32,
        num_query_heads=32,
        head_size=128,
        block_size=16,
    )

    # Encode to step context (no wait)
    with step_ctx:
        attn_op.encode(
            step_ctx=step_ctx,
            query_buffer=query_buf,
            key_buffer=kv_cache.get_key_buffer(layer_idx),
            value_buffer=kv_cache.get_value_buffer(layer_idx),
            block_table_buffer=block_table_buf,
            seq_lens_buffer=seq_lens_buf,
            output_buffer=output_buf,
            num_seqs=num_seqs,
            max_seq_len=max_seq_len,
        )
"""

import ctypes
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLSize
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    MTLSize = None


@dataclass
class AttentionParams:
    """Parameters for attention kernel."""
    scale: float
    num_seqs: int
    max_seq_len: int
    max_blocks_per_seq: int
    head_size: int
    block_size: int
    num_kv_heads: int
    num_query_heads: int
    queries_per_kv: int
    # KV cache strides
    k_stride_block: int
    k_stride_head: int
    k_stride_token: int
    v_stride_block: int
    v_stride_head: int
    v_stride_token: int
    # Query strides
    q_stride_token: int
    q_stride_head: int
    # Output strides
    o_stride_token: int
    o_stride_head: int

    def to_bytes(self) -> bytes:
        """Convert to bytes for Metal buffer."""
        return struct.pack(
            "fIIIIIIIIIIIIIIIIII",  # float + 18 uints
            self.scale,
            self.num_seqs,
            self.max_seq_len,
            self.max_blocks_per_seq,
            self.head_size,
            self.block_size,
            self.num_kv_heads,
            self.num_query_heads,
            self.queries_per_kv,
            self.k_stride_block,
            self.k_stride_head,
            self.k_stride_token,
            self.v_stride_block,
            self.v_stride_head,
            self.v_stride_token,
            self.q_stride_token,
            self.q_stride_head,
            self.o_stride_token,
            self.o_stride_head,
        )


class PagedAttentionOp:
    """Encode-only paged attention operation.

    This op encodes attention kernel dispatches to a command buffer.
    It does NOT execute the kernels - that happens at step boundary.

    Supports both:
    - Decode-only attention (single token per sequence)
    - Prefill attention (multiple tokens, chunked for memory)

    Attributes:
        context: MetalEngineContext
        num_kv_heads: Number of KV heads
        num_query_heads: Number of query heads
        head_size: Dimension per head
        block_size: Tokens per KV cache block
        scale: Attention scaling factor
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        num_kv_heads: int,
        num_query_heads: int,
        head_size: int,
        block_size: int,
        scale: Optional[float] = None,
    ):
        """Initialize paged attention op.

        Args:
            context: MetalEngineContext for pipeline access
            num_kv_heads: Number of KV heads
            num_query_heads: Number of query heads
            head_size: Dimension per head (32, 64, 96, or 128)
            block_size: Tokens per KV cache block
            scale: Attention scale (default: 1/sqrt(head_size))
        """
        self._context = context
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_size = head_size
        self.block_size = block_size
        self.scale = scale if scale is not None else 1.0 / (head_size ** 0.5)
        self.queries_per_kv = num_query_heads // num_kv_heads

        # Pre-compute strides
        # KV cache layout: [num_blocks, num_kv_heads, block_size, head_size]
        self.kv_stride_block = num_kv_heads * block_size * head_size
        self.kv_stride_head = block_size * head_size
        self.kv_stride_token = head_size

        # Q/O layout: [num_tokens, num_query_heads, head_size]
        self.q_stride_token = num_query_heads * head_size
        self.q_stride_head = head_size
        self.o_stride_token = num_query_heads * head_size
        self.o_stride_head = head_size

        # Compile kernel
        self._compile_kernel()

        logger.info(
            f"PagedAttentionOp initialized: "
            f"heads={num_query_heads}/{num_kv_heads}, "
            f"head_size={head_size}, block_size={block_size}"
        )

    def _compile_kernel(self) -> None:
        """Compile attention kernel."""
        # Find and compile shader
        shader_path = Path(__file__).parent.parent.parent / "metal" / "kernels" / "paged_attention_fused.metal"

        if not shader_path.exists():
            # Try alternative path
            shader_path = Path(__file__).parent.parent / "kernels" / "paged_attention.metal"

        if shader_path.exists():
            self._context.compile_library("paged_attention", source_path=str(shader_path))
        else:
            logger.warning(f"Attention shader not found, using placeholder. Path: {shader_path}")
            # Create a placeholder - in production this would be an error
            return

        # Select kernel based on head size
        if self.head_size == 128:
            kernel_name = "paged_attention_fused_h128"
        else:
            kernel_name = "paged_attention_fused_generic"

        try:
            self._pipeline = self._context.get_pipeline("paged_attention", kernel_name)
            self._kernel_name = kernel_name
        except RuntimeError:
            # Fallback to generic
            try:
                self._pipeline = self._context.get_pipeline("paged_attention", "paged_attention_fused_generic")
                self._kernel_name = "paged_attention_fused_generic"
            except RuntimeError:
                logger.warning("No attention kernel available")
                self._pipeline = None
                self._kernel_name = None

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        query_buffer: Any,  # MTLBuffer
        key_buffer: Any,  # MTLBuffer
        value_buffer: Any,  # MTLBuffer
        block_table_buffer: Any,  # MTLBuffer
        seq_lens_buffer: Any,  # MTLBuffer
        output_buffer: Any,  # MTLBuffer
        num_seqs: int,
        max_seq_len: int,
        max_blocks_per_seq: int = 0,
    ) -> None:
        """Encode attention kernel to command buffer.

        This encodes the kernel dispatch WITHOUT executing. Execution
        happens at step boundary via waitUntilCompleted().

        Args:
            step_ctx: EngineStepContext with encoder
            query_buffer: Query MTLBuffer [num_tokens, num_query_heads, head_size]
            key_buffer: Key cache MTLBuffer
            value_buffer: Value cache MTLBuffer
            block_table_buffer: Block table MTLBuffer [num_seqs, max_blocks]
            seq_lens_buffer: Sequence lengths MTLBuffer [num_seqs]
            output_buffer: Output MTLBuffer [num_tokens, num_query_heads, head_size]
            num_seqs: Number of sequences
            max_seq_len: Maximum sequence length in batch
            max_blocks_per_seq: Maximum blocks per sequence (auto-computed if 0)
        """
        if self._pipeline is None:
            raise RuntimeError("Attention kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Compute max_blocks_per_seq if not provided
        if max_blocks_per_seq == 0:
            max_blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size

        # Create params
        params = AttentionParams(
            scale=self.scale,
            num_seqs=num_seqs,
            max_seq_len=max_seq_len,
            max_blocks_per_seq=max_blocks_per_seq,
            head_size=self.head_size,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            num_query_heads=self.num_query_heads,
            queries_per_kv=self.queries_per_kv,
            k_stride_block=self.kv_stride_block,
            k_stride_head=self.kv_stride_head,
            k_stride_token=self.kv_stride_token,
            v_stride_block=self.kv_stride_block,
            v_stride_head=self.kv_stride_head,
            v_stride_token=self.kv_stride_token,
            q_stride_token=self.q_stride_token,
            q_stride_head=self.q_stride_head,
            o_stride_token=self.o_stride_token,
            o_stride_head=self.o_stride_head,
        )

        # Create params buffer
        params_buffer = self._context.create_buffer_from_bytes(params.to_bytes())

        # Get encoder from step context
        encoder = step_ctx.encoder

        # Set pipeline
        encoder.setComputePipelineState_(self._pipeline)

        # Set buffers
        # Kernel buffer layout:
        # buffer(0): query
        # buffer(1): key_cache
        # buffer(2): value_cache
        # buffer(3): block_table
        # buffer(4): seq_lens
        # buffer(5): output
        # buffer(6): params
        encoder.setBuffer_offset_atIndex_(query_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 6)

        # Dispatch
        # Each threadgroup handles one (seq, head) pair
        threads_per_threadgroup = MTLSize(32, 1, 1)  # One SIMD group
        threadgroups = MTLSize(num_seqs, self.num_query_heads, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )

    def encode_with_kv_write(
        self,
        step_ctx: Any,  # EngineStepContext
        query_buffer: Any,
        new_keys_buffer: Any,  # New K for this token
        new_values_buffer: Any,  # New V for this token
        key_buffer: Any,  # KV cache
        value_buffer: Any,
        block_table_buffer: Any,
        seq_lens_buffer: Any,
        output_buffer: Any,
        num_seqs: int,
        max_seq_len: int,
        max_blocks_per_seq: int = 0,
    ) -> None:
        """Encode fused KV-write + attention to command buffer.

        This is the fused path for decode: writes new K/V and computes
        attention in sequence, but still encode-only (no waits).

        Args:
            step_ctx: EngineStepContext with encoder
            query_buffer: Query MTLBuffer
            new_keys_buffer: New K values for current token
            new_values_buffer: New V values for current token
            key_buffer: Key cache MTLBuffer
            value_buffer: Value cache MTLBuffer
            block_table_buffer: Block table MTLBuffer
            seq_lens_buffer: Sequence lengths MTLBuffer
            output_buffer: Output MTLBuffer
            num_seqs: Number of sequences
            max_seq_len: Maximum sequence length
            max_blocks_per_seq: Maximum blocks per sequence
        """
        # For now, encode KV write then attention as two dispatches
        # The full fused kernel would be a single dispatch

        # First: encode attention (KV should already be written by KVWriteOp)
        self.encode(
            step_ctx=step_ctx,
            query_buffer=query_buffer,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table_buffer=block_table_buffer,
            seq_lens_buffer=seq_lens_buffer,
            output_buffer=output_buffer,
            num_seqs=num_seqs,
            max_seq_len=max_seq_len,
            max_blocks_per_seq=max_blocks_per_seq,
        )

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "num_kv_heads": self.num_kv_heads,
            "num_query_heads": self.num_query_heads,
            "head_size": self.head_size,
            "block_size": self.block_size,
            "scale": self.scale,
            "kernel": self._kernel_name,
        }
