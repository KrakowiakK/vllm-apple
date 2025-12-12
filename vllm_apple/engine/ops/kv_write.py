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


# Inline kernel source for KV write operations
KV_WRITE_KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// KV write for decode step (single token per sequence)
// Each threadgroup handles one (seq, kv_head) pair
// Threads cooperate to copy head_size elements
kernel void kv_write_decode(
    device half* key_cache [[buffer(0)]],
    device half* value_cache [[buffer(1)]],
    device const int* block_table [[buffer(2)]],
    device const int* seq_lens [[buffer(3)]],
    constant uint& num_seqs [[buffer(4)]],
    constant uint& head_size [[buffer(5)]],
    constant uint& block_size [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& max_blocks_per_seq [[buffer(8)]],
    device const half* new_keys [[buffer(9)]],
    device const half* new_values [[buffer(10)]],
    uint2 gid [[threadgroup_position_in_grid]],  // (seq_idx, head_idx)
    uint tid [[thread_index_in_threadgroup]]
) {
    uint seq_idx = gid.x;
    uint head_idx = gid.y;

    if (seq_idx >= num_seqs || head_idx >= num_kv_heads) return;

    // Get sequence length (current position is seq_len - 1 for decode)
    int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    int token_pos = seq_len - 1;  // Position of new token
    int block_idx = token_pos / int(block_size);
    int token_in_block = token_pos % int(block_size);

    // Get physical block from block table
    int physical_block = block_table[seq_idx * max_blocks_per_seq + block_idx];
    if (physical_block < 0) return;  // Invalid block

    // Cache layout: [num_blocks, num_kv_heads, block_size, head_size]
    uint cache_offset = physical_block * num_kv_heads * block_size * head_size
                      + head_idx * block_size * head_size
                      + token_in_block * head_size;

    // New K/V layout: [num_seqs, num_kv_heads, head_size]
    uint new_offset = seq_idx * num_kv_heads * head_size + head_idx * head_size;

    // Copy elements (threads cooperate)
    for (uint i = tid; i < head_size; i += 32) {
        key_cache[cache_offset + i] = new_keys[new_offset + i];
        value_cache[cache_offset + i] = new_values[new_offset + i];
    }
}

// KV write for prefill step (multiple tokens via slot_mapping)
// Each threadgroup handles one (token, kv_head) pair
kernel void kv_write_prefill(
    device half* key_cache [[buffer(0)]],
    device half* value_cache [[buffer(1)]],
    device const int* slot_mapping [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& head_size [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    device const half* new_keys [[buffer(7)]],
    device const half* new_values [[buffer(8)]],
    uint2 gid [[threadgroup_position_in_grid]],  // (token_idx, head_idx)
    uint tid [[thread_index_in_threadgroup]]
) {
    uint token_idx = gid.x;
    uint head_idx = gid.y;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;

    // Get slot (absolute position in cache)
    int slot = slot_mapping[token_idx];
    if (slot < 0) return;  // Invalid slot

    int block_idx = slot / int(block_size);
    int token_in_block = slot % int(block_size);

    // Cache layout: [num_blocks, num_kv_heads, block_size, head_size]
    uint cache_offset = block_idx * num_kv_heads * block_size * head_size
                      + head_idx * block_size * head_size
                      + token_in_block * head_size;

    // New K/V layout: [num_tokens, num_kv_heads, head_size]
    uint new_offset = token_idx * num_kv_heads * head_size + head_idx * head_size;

    // Copy elements
    for (uint i = tid; i < head_size; i += 32) {
        key_cache[cache_offset + i] = new_keys[new_offset + i];
        value_cache[cache_offset + i] = new_values[new_offset + i];
    }
}
"""


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
        """Compile KV write kernels from inline source."""
        # Use inline kernel source
        self._context.compile_library("kv_write_inline", source_code=KV_WRITE_KERNEL_SOURCE)

        # Get pipelines
        try:
            self._decode_pipeline = self._context.get_pipeline("kv_write_inline", "kv_write_decode")
            self._prefill_pipeline = self._context.get_pipeline("kv_write_inline", "kv_write_prefill")
            logger.debug("KV write kernels compiled successfully")
        except RuntimeError as e:
            logger.warning(f"KV write kernel compilation failed: {e}")
            self._decode_pipeline = None
            self._prefill_pipeline = None

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
        new_keys_offset: int = 0,  # Byte offset into new_keys_buffer
        new_values_offset: int = 0,  # Byte offset into new_values_buffer
        max_blocks_per_seq: int = 128,  # Max blocks per sequence in block_table
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
            new_keys_offset: Byte offset into new_keys_buffer (for views into QKV buffer)
            new_values_offset: Byte offset into new_values_buffer (for views into QKV buffer)
            max_blocks_per_seq: Maximum blocks per sequence (block_table second dim)
        """
        if self._decode_pipeline is None:
            raise RuntimeError("KV write decode kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode_decode() called outside ENCODE phase")

        # Get encoder from step context
        encoder = step_ctx.get_compute_encoder()

        # Set pipeline
        encoder.setComputePipelineState_(self._decode_pipeline)

        # Set buffers matching new kernel signature:
        # buffer(0): key_cache
        # buffer(1): value_cache
        # buffer(2): block_table
        # buffer(3): seq_lens
        # buffer(4-8): constants (num_seqs, head_size, block_size, num_kv_heads, max_blocks_per_seq)
        # buffer(9): new_keys
        # buffer(10): new_values
        encoder.setBuffer_offset_atIndex_(key_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(value_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 3)

        # Set constants
        encoder.setBytes_length_atIndex_(struct.pack("I", num_seqs), 4, 4)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.head_size), 4, 5)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.block_size), 4, 6)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.num_kv_heads), 4, 7)
        encoder.setBytes_length_atIndex_(struct.pack("I", max_blocks_per_seq), 4, 8)

        encoder.setBuffer_offset_atIndex_(new_keys_buffer, new_keys_offset, 9)
        encoder.setBuffer_offset_atIndex_(new_values_buffer, new_values_offset, 10)

        # Dispatch: one threadgroup per (seq, head) pair, 32 threads per group
        threads_per_threadgroup = MTLSize(32, 1, 1)
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
        new_keys_offset: int = 0,  # Byte offset into new_keys_buffer
        new_values_offset: int = 0,  # Byte offset into new_values_buffer
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
            slot_mapping_buffer: Slot indices [num_tokens] (int32)
            num_tokens: Total number of tokens
            new_keys_offset: Byte offset into new_keys_buffer
            new_values_offset: Byte offset into new_values_buffer
        """
        if self._prefill_pipeline is None:
            raise RuntimeError("KV write prefill kernel not compiled")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode_prefill() called outside ENCODE phase")

        # Get encoder from step context
        encoder = step_ctx.get_compute_encoder()

        # Set pipeline
        encoder.setComputePipelineState_(self._prefill_pipeline)

        # Set buffers matching kernel signature:
        # buffer(0): key_cache
        # buffer(1): value_cache
        # buffer(2): slot_mapping
        # buffer(3-6): constants (num_tokens, head_size, block_size, num_kv_heads)
        # buffer(7): new_keys
        # buffer(8): new_values
        encoder.setBuffer_offset_atIndex_(key_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(value_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(slot_mapping_buffer, 0, 2)

        # Set constants
        encoder.setBytes_length_atIndex_(struct.pack("I", num_tokens), 4, 3)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.head_size), 4, 4)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.block_size), 4, 5)
        encoder.setBytes_length_atIndex_(struct.pack("I", self.num_kv_heads), 4, 6)

        encoder.setBuffer_offset_atIndex_(new_keys_buffer, new_keys_offset, 7)
        encoder.setBuffer_offset_atIndex_(new_values_buffer, new_values_offset, 8)

        # Dispatch: one threadgroup per (token, head) pair, 32 threads per group
        threads_per_threadgroup = MTLSize(32, 1, 1)
        threadgroups = MTLSize(num_tokens, self.num_kv_heads, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "block_size": self.block_size,
            "kernel": self._kernel_name,
        }
