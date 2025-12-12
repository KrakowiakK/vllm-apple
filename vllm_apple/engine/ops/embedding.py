# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embedding Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only token embedding operations.
Encodes token ID to embedding vector lookup without internal waits.

Key Principle: Encode-only API. No internal waits.

Usage:
    from vllm_apple.engine.ops.embedding import EngineEmbedding

    # Create op during initialization
    embedding = EngineEmbedding(
        context=engine_context,
        vocab_size=32000,
        hidden_size=4096,
    )

    # Set weight buffer
    embedding.set_weights(embedding_weights)

    # Encode embedding lookup (no wait)
    embedding.encode(step_ctx, token_ids, output_buffer, num_tokens)
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import struct

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
class EmbeddingConfig:
    """Configuration for embedding operation."""
    vocab_size: int
    hidden_size: int
    dtype: EngineDType = EngineDType.FLOAT16


class EngineEmbedding:
    """Encode-only token embedding operation.

    This op looks up token embeddings from a vocabulary embedding table.
    Uses a simple gather kernel to fetch rows from the embedding matrix.

    Attributes:
        context: MetalEngineContext
        config: EmbeddingConfig
        weight_buffer: MTLBuffer containing embedding weights [vocab_size, hidden_size]
    """

    # Kernel source for embedding lookup
    KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Embedding lookup kernel
// Each thread copies one element of the embedding vector for one token
kernel void embedding_lookup_kernel(
    device const half* embeddings [[buffer(0)]],    // [vocab_size, hidden_size]
    device const int* token_ids [[buffer(1)]],      // [num_tokens]
    device half* output [[buffer(2)]],              // [num_tokens, hidden_size]
    constant uint& hidden_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]           // (element_idx, token_idx)
) {
    uint element_idx = gid.x;
    uint token_idx = gid.y;

    // Bounds check
    if (element_idx >= hidden_size) return;

    // Get token ID with bounds check
    int token_id = token_ids[token_idx];
    if (token_id < 0 || uint(token_id) >= vocab_size) {
        // Out of bounds - zero output
        output[token_idx * hidden_size + element_idx] = half(0.0f);
        return;
    }

    // Lookup embedding value
    uint embed_idx = uint(token_id) * hidden_size + element_idx;
    output[token_idx * hidden_size + element_idx] = embeddings[embed_idx];
}

// Embedding lookup with position offset (for encoder-decoder)
kernel void embedding_lookup_with_offset_kernel(
    device const half* embeddings [[buffer(0)]],
    device const int* token_ids [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    constant uint& token_offset [[buffer(5)]],      // Offset into output buffer
    uint2 gid [[thread_position_in_grid]]
) {
    uint element_idx = gid.x;
    uint token_idx = gid.y;

    if (element_idx >= hidden_size) return;

    int token_id = token_ids[token_idx];
    if (token_id < 0 || uint(token_id) >= vocab_size) {
        output[(token_offset + token_idx) * hidden_size + element_idx] = half(0.0f);
        return;
    }

    uint embed_idx = uint(token_id) * hidden_size + element_idx;
    output[(token_offset + token_idx) * hidden_size + element_idx] = embeddings[embed_idx];
}
"""

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        vocab_size: int,
        hidden_size: int,
        dtype: EngineDType = EngineDType.FLOAT16,
    ):
        """Initialize embedding op.

        Args:
            context: MetalEngineContext
            vocab_size: Size of vocabulary
            hidden_size: Embedding dimension
            dtype: Data type for embeddings
        """
        self._context = context
        self.config = EmbeddingConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        self._weight_buffer = None
        self._pipeline = None
        self._pipeline_offset = None

        # Compile kernel
        self._compile_kernels()

        logger.info(
            f"EngineEmbedding initialized: vocab_size={vocab_size}, "
            f"hidden_size={hidden_size}"
        )

    def _compile_kernels(self) -> None:
        """Compile embedding Metal kernels."""
        self._context.compile_library("embedding", source_code=self.KERNEL_SOURCE)
        self._pipeline = self._context.get_pipeline(
            "embedding", "embedding_lookup_kernel"
        )
        self._pipeline_offset = self._context.get_pipeline(
            "embedding", "embedding_lookup_with_offset_kernel"
        )

    def set_weights(self, weight_buffer: Any) -> None:
        """Set embedding weight buffer.

        Args:
            weight_buffer: MTLBuffer containing [vocab_size, hidden_size] embeddings
        """
        self._weight_buffer = weight_buffer

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        token_ids: Union[EngineTensor, Any],  # MTLBuffer [num_tokens] int32
        output_buffer: Union[EngineTensor, Any],  # MTLBuffer [num_tokens, hidden_size]
        num_tokens: int,
        token_offset: int = 0,  # Offset into output buffer
    ) -> None:
        """Encode embedding lookup to command buffer.

        Args:
            step_ctx: EngineStepContext with encoder
            token_ids: Token IDs tensor [num_tokens] (int32)
            output_buffer: Output tensor [num_tokens, hidden_size]
            num_tokens: Number of tokens
            token_offset: Offset for output position
        """
        if self._weight_buffer is None:
            raise RuntimeError("Weights not set - call set_weights() first")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Get buffers
        if isinstance(token_ids, EngineTensor):
            token_buf = token_ids.buffer
            token_offset_bytes = token_ids.offset
        else:
            token_buf = token_ids
            token_offset_bytes = 0

        if isinstance(output_buffer, EngineTensor):
            out_buf = output_buffer.buffer
            out_offset = output_buffer.offset
        else:
            out_buf = output_buffer
            out_offset = 0

        # Get compute encoder
        encoder = step_ctx.get_compute_encoder()

        # Choose pipeline based on whether we have an offset
        if token_offset == 0:
            encoder.setComputePipelineState_(self._pipeline)
        else:
            encoder.setComputePipelineState_(self._pipeline_offset)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(self._weight_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(token_buf, token_offset_bytes, 1)
        encoder.setBuffer_offset_atIndex_(out_buf, out_offset, 2)

        # Set constants
        hidden_size_data = struct.pack("I", self.config.hidden_size)
        vocab_size_data = struct.pack("I", self.config.vocab_size)
        encoder.setBytes_length_atIndex_(hidden_size_data, 4, 3)
        encoder.setBytes_length_atIndex_(vocab_size_data, 4, 4)

        if token_offset != 0:
            offset_data = struct.pack("I", token_offset)
            encoder.setBytes_length_atIndex_(offset_data, 4, 5)

        # Dispatch: 2D grid (hidden_size x num_tokens)
        # Round hidden_size up to multiple of 32 for efficient threadgroup dispatch
        threads_x = min(256, self.config.hidden_size)
        threads_y = 1

        threadgroups_x = (self.config.hidden_size + threads_x - 1) // threads_x
        threadgroups_y = num_tokens

        thread_groups = MTLSize(threadgroups_x, threadgroups_y, 1)
        threads_per_threadgroup = MTLSize(threads_x, threads_y, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, threads_per_threadgroup
        )

    def get_output_size(self, num_tokens: int) -> int:
        """Get required output buffer size in bytes."""
        element_size = 2  # float16
        return num_tokens * self.config.hidden_size * element_size

    def get_weight_size(self) -> int:
        """Get required weight buffer size in bytes."""
        element_size = 2  # float16
        return self.config.vocab_size * self.config.hidden_size * element_size

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
        }
