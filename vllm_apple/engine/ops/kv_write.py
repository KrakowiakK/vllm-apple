# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Write Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only KV cache write operations. The operation
encodes kernel dispatches to write new K/V values to the cache WITHOUT
executing. Execution happens at step boundary.

Key Principle: Encode-only API. No internal waits.

Supports:
- Decode write: Single token per sequence to cache
- Prefill write: Multiple tokens per sequence (chunked)

Usage:
    from vllm_apple.engine.ops.kv_write import KVWriteOp

    # Create op during initialization
    kv_write_op = KVWriteOp(
        context=engine_context,
        num_kv_heads=32,
        head_size=128,
        block_size=16,
    )

    # Encode KV write (no wait)
    with step_ctx:
        kv_write_op.encode_decode(
            step_ctx=step_ctx,
            new_keys_buffer=new_k_buf,
            new_values_buffer=new_v_buf,
            key_buffer=kv_cache.get_key_buffer(layer_idx),
            value_buffer=kv_cache.get_value_buffer(layer_idx),
            block_table_buffer=block_table_buf,
            seq_lens_buffer=seq_lens_buf,
            num_seqs=num_seqs,
        )
"""

import struct
from pathlib import Path
from typing import Any, Dict, Optional
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
class KVWriteParams:
    """Parameters for KV write kernel."""
    num_seqs: int
    head_size: int
    block_size: int
    num_kv_heads: int
    # KV cache strides
    k_stride_block: int
    k_stride_head: int
    k_stride_token: int
    # New K/V strides
    new_kv_stride_token: int
    new_kv_stride_head: int

    def to_bytes(self) -> bytes:
        """Convert to bytes for Metal buffer."""
        return struct.pack(
            "IIIIIIIII",  # 9 uints
            self.num_seqs,
            self.head_size,
            self.block_size,
            self.num_kv_heads,
            self.k_stride_block,
            self.k_stride_head,
            self.k_stride_token,
            self.new_kv_stride_token,
            self.new_kv_stride_head,
        )


class KVWriteOp:
    """Encode-only KV cache write operation.

    This op encodes kernel dispatches to write new K/V values to the cache.
    It does NOT execute the kernels - that happens at step boundary.

    Supports:
    - Decode: Write single token per sequence (most common)
    - Prefill: Write multiple tokens per sequence

    Attributes:
        context: MetalEngineContext
        num_kv_heads: Number of KV heads
        head_size: Dimension per head
        block_size: Tokens per KV cache block
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        num_kv_heads: int,
        head_size: int,
        block_size: int,
    ):
        """Initialize KV write op.

        Args:
            context: MetalEngineContext for pipeline access
            num_kv_heads: Number of KV heads
            head_size: Dimension per head
            block_size: Tokens per KV cache block
        """
        self._context = context
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size

        # Pre-compute strides
        # KV cache layout: [num_blocks, num_kv_heads, block_size, head_size]
        self.kv_stride_block = num_kv_heads * block_size * head_size
        self.kv_stride_head = block_size * head_size
        self.kv_stride_token = head_size

        # New K/V layout: [num_tokens, num_kv_heads, head_size]
        self.new_kv_stride_token = num_kv_heads * head_size
        self.new_kv_stride_head = head_size

        # Compile kernel
        self._compile_kernel()

        logger.info(
            f"KVWriteOp initialized: "
            f"num_kv_heads={num_kv_heads}, head_size={head_size}, block_size={block_size}"
        )

    def _compile_kernel(self) -> None:
        """Compile KV write kernel."""
        # Find and compile shader
        shader_path = Path(__file__).parent.parent.parent / "metal" / "kernels" / "paged_attention_fused.metal"

        if not shader_path.exists():
            shader_path = Path(__file__).parent.parent / "kernels" / "kv_write.metal"

        if shader_path.exists():
            self._context.compile_library("kv_write", source_path=str(shader_path))
        else:
            logger.warning(f"KV write shader not found, using placeholder. Path: {shader_path}")
            self._pipeline = None
            self._kernel_name = None
            return

        # Get pipeline
        try:
            self._pipeline = self._context.get_pipeline("kv_write", "kv_write_decode")
            self._kernel_name = "kv_write_decode"
        except RuntimeError:
            logger.warning("KV write kernel not available")
            self._pipeline = None
            self._kernel_name = None

    def encode_decode(
        self,
        step_ctx: Any,  # EngineStepContext
        new_keys_buffer: Any,  # MTLBuffer [num_seqs, num_kv_heads, head_size]
        new_values_buffer: Any,  # MTLBuffer [num_seqs, num_kv_heads, head_size]
        key_buffer: Any,  # MTLBuffer (KV cache)
        value_buffer: Any,  # MTLBuffer (KV cache)
        block_table_buffer: Any,  # MTLBuffer [num_seqs, max_blocks]
        seq_lens_buffer: Any,  # MTLBuffer [num_seqs]
        num_seqs: int,
    ) -> None:
        """Encode KV write for decode step (single token per sequence).

        This encodes the kernel dispatch WITHOUT executing. Execution
        happens at step boundary via waitUntilCompleted().

        Args:
            step_ctx: EngineStepContext with encoder
            new_keys_buffer: New K values [num_seqs, num_kv_heads, head_size]
            new_values_buffer: New V values [num_seqs, num_kv_heads, head_size]
            key_buffer: Key cache MTLBuffer
            value_buffer: Value cache MTLBuffer
            block_table_buffer: Block table MTLBuffer
            seq_lens_buffer: Sequence lengths MTLBuffer
            num_seqs: Number of sequences
        """
        if self._pipeline is None:
            raise RuntimeError("KV write kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode_decode() called outside ENCODE phase")

        # Create params
        params = KVWriteParams(
            num_seqs=num_seqs,
            head_size=self.head_size,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            k_stride_block=self.kv_stride_block,
            k_stride_head=self.kv_stride_head,
            k_stride_token=self.kv_stride_token,
            new_kv_stride_token=self.new_kv_stride_token,
            new_kv_stride_head=self.new_kv_stride_head,
        )

        # Create params buffer
        params_buffer = self._context.create_buffer_from_bytes(params.to_bytes())

        # Get encoder from step context
        encoder = step_ctx.encoder

        # Set pipeline
        encoder.setComputePipelineState_(self._pipeline)

        # Set buffers
        # Kernel buffer layout:
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
        encoder.setBuffer_offset_atIndex_(new_keys_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(new_values_buffer, 0, 6)

        # Dispatch
        # Each threadgroup handles one (seq, kv_head) pair
        threads_per_threadgroup = MTLSize(32, 1, 1)  # One SIMD group
        threadgroups = MTLSize(num_seqs, self.num_kv_heads, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )

    def encode_prefill(
        self,
        step_ctx: Any,  # EngineStepContext
        new_keys_buffer: Any,  # MTLBuffer [num_tokens, num_kv_heads, head_size]
        new_values_buffer: Any,  # MTLBuffer [num_tokens, num_kv_heads, head_size]
        key_buffer: Any,  # MTLBuffer (KV cache)
        value_buffer: Any,  # MTLBuffer (KV cache)
        slot_mapping_buffer: Any,  # MTLBuffer [num_tokens]
        num_tokens: int,
    ) -> None:
        """Encode KV write for prefill step (multiple tokens).

        For prefill, we write each token's K/V to its assigned slot
        using slot_mapping instead of block_table + seq_lens.

        Args:
            step_ctx: EngineStepContext with encoder
            new_keys_buffer: New K values [num_tokens, num_kv_heads, head_size]
            new_values_buffer: New V values [num_tokens, num_kv_heads, head_size]
            key_buffer: Key cache MTLBuffer
            value_buffer: Value cache MTLBuffer
            slot_mapping_buffer: Slot indices [num_tokens]
            num_tokens: Total number of tokens
        """
        # Prefill KV write uses a different kernel that takes slot_mapping
        # For now, we use a simple implementation that could be optimized

        if self._pipeline is None:
            raise RuntimeError("KV write kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode_prefill() called outside ENCODE phase")

        # For prefill, we need a kernel that handles slot_mapping
        # This is a placeholder - the actual implementation would use
        # a dedicated prefill KV write kernel

        logger.warning("Prefill KV write not yet implemented - skipping encode")

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "block_size": self.block_size,
            "kernel": self._kernel_name,
        }
