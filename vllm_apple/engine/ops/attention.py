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
        # Decode (1 token/seq): fused KV-write + attention
        attn_op.encode_decode_fused(...)

        # Prefill/mixed: token-parallel attention (KV write via KVWriteOp)
        attn_op.encode_prefill(...)
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
class FusedAttentionParams:
    """Parameters matching `FusedAttentionParams` in paged_attention_fused.metal."""
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
    # Query/Output strides
    q_stride_token: int
    q_stride_head: int
    o_stride_token: int
    o_stride_head: int
    # New K/V strides
    new_kv_stride_token: int
    new_kv_stride_head: int

    def to_bytes(self) -> bytes:
        """Convert to bytes for Metal buffer."""
        fmt = "f" + ("I" * 20)  # float + 20 uints
        return struct.pack(
            fmt,
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
            self.new_kv_stride_token,
            self.new_kv_stride_head,
        )


@dataclass
class PagedAttentionParams:
    """Parameters matching `PagedAttentionParams` in paged_attention_v2.metal."""
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
    # Query/Output strides
    q_stride_token: int
    q_stride_head: int
    o_stride_token: int
    o_stride_head: int

    def to_bytes(self) -> bytes:
        """Convert to bytes for Metal buffer."""
        fmt = "f" + ("I" * 18)  # float + 18 uints
        return struct.pack(
            fmt,
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

    Supports:
    - Decode (fused KV-write + attention)
    - Prefill/mixed (token-parallel attention; KV write handled separately)

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

        # New K/V layout (decode): [num_seqs, num_kv_heads, head_size]
        self.new_kv_stride_token = num_kv_heads * head_size
        self.new_kv_stride_head = head_size

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

        # Prefill kernel (token-parallel) lives in paged_attention_v2.metal.
        v2_shader_path = Path(__file__).parent.parent.parent / "metal" / "kernels" / "paged_attention_v2.metal"
        self._prefill_pipeline = None
        if v2_shader_path.exists():
            try:
                self._context.compile_library("paged_attention_v2", source_path=str(v2_shader_path))
                self._prefill_pipeline = self._context.get_pipeline(
                    "paged_attention_v2",
                    "paged_attention_v2_prefill",
                )
            except Exception as e:
                logger.warning(f"Prefill attention kernel unavailable: {e}")
                self._prefill_pipeline = None
        else:
            logger.warning(f"Prefill attention shader not found: {v2_shader_path}")

        self._kv_write_pipeline = None

        # Kernel selection:
        # - h128 uses a two-phase approach: kv_write_decode + paged_attention_fused_h128
        # - non-128 uses paged_attention_fused_generic (single-dispatch fused kernel)
        if self.head_size == 128:
            attn_kernel = "paged_attention_fused_h128"
            kv_write_kernel = "kv_write_decode"
            try:
                self._kv_write_pipeline = self._context.get_pipeline("paged_attention", kv_write_kernel)
                self._pipeline = self._context.get_pipeline("paged_attention", attn_kernel)
                self._kernel_name = attn_kernel
            except RuntimeError:
                logger.warning("No attention kernel available")
                self._kv_write_pipeline = None
                self._pipeline = None
                self._kernel_name = None
        else:
            attn_kernel = "paged_attention_fused_generic"
            try:
                self._pipeline = self._context.get_pipeline("paged_attention", attn_kernel)
                self._kernel_name = attn_kernel
            except RuntimeError:
                logger.warning("No attention kernel available")
                self._pipeline = None
                self._kernel_name = None

    def encode_prefill(
        self,
        step_ctx: Any,  # EngineStepContext
        query_buffer: Any,  # MTLBuffer
        key_buffer: Any,  # MTLBuffer
        value_buffer: Any,  # MTLBuffer
        block_table_buffer: Any,  # MTLBuffer
        token_to_seq_buffer: Any,  # MTLBuffer [num_tokens]
        positions_buffer: Any,  # MTLBuffer [num_tokens]
        output_buffer: Any,  # MTLBuffer
        num_tokens: int,
        num_seqs: int,
        max_seq_len: int,
        max_blocks_per_seq: int,
        query_offset: int = 0,
        output_offset: int = 0,
    ) -> None:
        """Encode prefill (token-parallel) paged attention.

        This kernel computes causal attention for each token in the flattened
        query batch using engine-owned KV cache buffers.
        """
        if self._prefill_pipeline is None:
            raise RuntimeError("Prefill attention kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode_prefill() called outside ENCODE phase")

        params = PagedAttentionParams(
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

        params_buffer = self._context.create_buffer_from_bytes(params.to_bytes())

        encoder = step_ctx.get_compute_encoder()
        encoder.setComputePipelineState_(self._prefill_pipeline)

        # Kernel buffer layout:
        # buffer(0): query
        # buffer(1): key_cache
        # buffer(2): value_cache
        # buffer(3): block_table
        # buffer(4): token_to_seq
        # buffer(5): positions
        # buffer(6): output
        # buffer(7): params
        encoder.setBuffer_offset_atIndex_(query_buffer, query_offset, 0)
        encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(token_to_seq_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(positions_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(output_buffer, output_offset, 6)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 7)

        threads_per_threadgroup = MTLSize(32, 1, 1)
        threadgroups = MTLSize(num_tokens, self.num_query_heads, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )

    def encode_decode_fused(
        self,
        step_ctx: Any,  # EngineStepContext
        query_buffer: Any,  # MTLBuffer
        new_keys_buffer: Any,  # MTLBuffer
        new_values_buffer: Any,  # MTLBuffer
        key_buffer: Any,  # MTLBuffer
        value_buffer: Any,  # MTLBuffer
        block_table_buffer: Any,  # MTLBuffer
        seq_lens_buffer: Any,  # MTLBuffer
        output_buffer: Any,  # MTLBuffer
        num_seqs: int,
        max_seq_len: int,
        max_blocks_per_seq: int = 0,
        query_offset: int = 0,
        new_keys_offset: int = 0,
        new_values_offset: int = 0,
        output_offset: int = 0,
    ) -> None:
        """Encode decode attention with fused KV-write.

        Uses either:
        - Two-phase path (head_size=128): kv_write_decode + attention kernel
        - Single-dispatch fused kernel (other head sizes): writes KV and computes attention

        Args:
            step_ctx: EngineStepContext with encoder
            query_buffer: Query MTLBuffer [num_seqs, num_query_heads, head_size]
            new_keys_buffer: New K values [num_seqs, num_kv_heads, head_size]
            new_values_buffer: New V values [num_seqs, num_kv_heads, head_size]
            key_buffer: Key cache MTLBuffer
            value_buffer: Value cache MTLBuffer
            block_table_buffer: Block table MTLBuffer [num_seqs, max_blocks]
            seq_lens_buffer: Sequence lengths MTLBuffer [num_seqs]
            output_buffer: Output MTLBuffer [num_seqs, num_query_heads, head_size]
            num_seqs: Number of sequences
            max_seq_len: Maximum sequence length in batch
            max_blocks_per_seq: Maximum blocks per sequence (auto-computed if 0)
            query_offset: Byte offset into query_buffer
            new_keys_offset: Byte offset into new_keys_buffer
            new_values_offset: Byte offset into new_values_buffer
            output_offset: Byte offset into output_buffer
        """
        if self._pipeline is None:
            raise RuntimeError("Attention kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Compute max_blocks_per_seq if not provided
        if max_blocks_per_seq == 0:
            max_blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size

        # Create params
        params = FusedAttentionParams(
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
            new_kv_stride_token=self.new_kv_stride_token,
            new_kv_stride_head=self.new_kv_stride_head,
        )

        # Create params buffer
        params_buffer = self._context.create_buffer_from_bytes(params.to_bytes())

        encoder = step_ctx.get_compute_encoder()

        if self._kernel_name == "paged_attention_fused_h128":
            if self._kv_write_pipeline is None:
                raise RuntimeError("KV write pipeline not available for h128 fused attention")

            # Phase 1: KV write
            encoder.setComputePipelineState_(self._kv_write_pipeline)

            # KV write kernel buffer layout:
            # buffer(0): key_cache (writable)
            # buffer(1): value_cache (writable)
            # buffer(2): block_table
            # buffer(3): seq_lens
            # buffer(4): params
            # buffer(5): new_keys
            # buffer(6): new_values
            encoder.setBuffer_offset_atIndex_(key_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(value_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(new_keys_buffer, new_keys_offset, 5)
            encoder.setBuffer_offset_atIndex_(new_values_buffer, new_values_offset, 6)

            threads_per_threadgroup = MTLSize(32, 1, 1)
            kv_write_threadgroups = MTLSize(num_seqs, self.num_kv_heads, 1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                kv_write_threadgroups, threads_per_threadgroup
            )

            step_ctx.memory_barrier()
            encoder = step_ctx.get_compute_encoder()

            # Phase 2: attention
            encoder.setComputePipelineState_(self._pipeline)

            # Attention kernel buffer layout:
            # buffer(0): query
            # buffer(1): key_cache
            # buffer(2): value_cache
            # buffer(3): block_table
            # buffer(4): seq_lens
            # buffer(5): output
            # buffer(6): params
            encoder.setBuffer_offset_atIndex_(query_buffer, query_offset, 0)
            encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(output_buffer, output_offset, 5)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 6)

            threads_per_threadgroup = MTLSize(32, 1, 1)
            threadgroups = MTLSize(num_seqs, self.num_query_heads, 1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                threadgroups, threads_per_threadgroup
            )
        elif self._kernel_name == "paged_attention_fused_generic":
            encoder.setComputePipelineState_(self._pipeline)

            # Generic fused kernel buffer layout:
            # buffer(0): query
            # buffer(1): key_cache
            # buffer(2): value_cache
            # buffer(3): block_table
            # buffer(4): seq_lens
            # buffer(5): output
            # buffer(6): params
            # buffer(7): new_keys
            # buffer(8): new_values
            encoder.setBuffer_offset_atIndex_(query_buffer, query_offset, 0)
            encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(output_buffer, output_offset, 5)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 6)
            encoder.setBuffer_offset_atIndex_(new_keys_buffer, new_keys_offset, 7)
            encoder.setBuffer_offset_atIndex_(new_values_buffer, new_values_offset, 8)

            threads_per_threadgroup = MTLSize(32, 1, 1)
            threadgroups = MTLSize(num_seqs, self.num_query_heads, 1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                threadgroups, threads_per_threadgroup
            )
        else:
            raise RuntimeError(f"Unknown attention kernel: {self._kernel_name}")

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
        # Prefer the decode fused path when new K/V are provided.
        self.encode_decode_fused(
            step_ctx=step_ctx,
            query_buffer=query_buffer,
            new_keys_buffer=new_keys_buffer,
            new_values_buffer=new_values_buffer,
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
            "prefill_kernel": "paged_attention_v2_prefill" if self._prefill_pipeline else None,
        }
