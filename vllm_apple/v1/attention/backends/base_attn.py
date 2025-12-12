# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base Attention Classes for Apple Platform.

This module provides shared base classes for attention backends on Apple Silicon.
Both Metal and MPS backends inherit from these classes to share common functionality
like PyTorch SDPA-based attention computation.

Key design principles:
- Base classes expose hooks for KV update, decode execution, metadata caching
- No Metal-specific state (like _metal_kv_cache) in base classes
- Subclasses implement backend-specific optimization paths
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BaseAppleAttentionMetadata:
    """Base metadata for Apple attention backends.

    This metadata is shared between Metal and MPS backends.

    Attributes:
        num_actual_tokens: Number of tokens excluding padding
        max_query_len: Maximum query length in batch
        query_start_loc: Start location of each query in batch
        max_seq_len: Maximum sequence length
        seq_lens: Length of each sequence
        block_table: Block table mapping sequences to cache blocks
        slot_mapping: Mapping from token positions to cache slots
        scheduler_metadata: Optional scheduler metadata
        causal: Whether to use causal masking
        num_decode_tokens: Number of decode tokens (at front of batch)
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: Optional[torch.Tensor]
    causal: bool = True
    num_decode_tokens: int = 0

    # Cached CPU versions to avoid repeated .tolist() calls
    _query_start_loc_cpu: Optional[list] = None
    _seq_lens_cpu: Optional[list] = None

    @property
    def query_start_loc_cpu(self) -> list:
        """Get query_start_loc as CPU list (cached)."""
        if self._query_start_loc_cpu is None:
            self._query_start_loc_cpu = self.query_start_loc.tolist()
        return self._query_start_loc_cpu

    @property
    def seq_lens_cpu(self) -> list:
        """Get seq_lens as CPU list (cached)."""
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.tolist()
        return self._seq_lens_cpu


class BaseAppleAttentionBackend(AttentionBackend):
    """Base attention backend for Apple Silicon platforms.

    This class provides common functionality for both Metal and MPS backends.
    Subclasses implement backend-specific optimizations.
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Common head sizes supported by PyTorch SDPA
        return [32, 64, 96, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Apple backends support decoder and encoder attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Get KV cache shape compatible with Apple platform.

        Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        - Dimension 0: [key_cache, value_cache]
        """
        return 2, num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_kv_cache_device() -> str:
        """Return device for KV cache allocation.

        Apple backends use unified memory accessible from CPU.
        """
        return "cpu"


class BaseAppleAttentionImpl(AttentionImpl):
    """Base attention implementation for Apple Silicon.

    This class provides shared functionality including:
    - PyTorch SDPA-based attention computation
    - Common initialization parameters
    - Utility methods for GQA support

    Subclasses (MetalAttentionImpl, MPSAttentionImpl) add backend-specific
    optimizations for decode path.

    IMPORTANT: This base class does NOT contain any Metal-specific state
    (like _metal_kv_cache) to ensure clean separation of concerns.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize base attention implementation.

        Args:
            num_heads: Number of query attention heads
            head_size: Size of each attention head
            scale: Attention scale factor (usually 1/sqrt(head_size))
            num_kv_heads: Number of key-value heads (for GQA)
            alibi_slopes: ALiBi slopes (not supported, will log warning)
            sliding_window: Sliding window size (optional)
            kv_cache_dtype: Data type for KV cache
            logits_soft_cap: Soft cap for logits (not supported, will log warning)
            attn_type: Type of attention (decoder, encoder, etc.)
            kv_sharing_target_layer_name: Name of layer to share KV cache with
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # Track if this layer shares KV cache with another layer
        self._is_kv_shared = kv_sharing_target_layer_name is not None

        # Handle unsupported features with warnings
        if logits_soft_cap is not None:
            logger.warning_once(
                "Apple attention backends do not support logits softcap, "
                "outputs may be slightly off"
            )
        self.logits_soft_cap = logits_soft_cap or 0

        if alibi_slopes is not None:
            logger.warning_once(
                "Apple attention backends do not support ALiBi slopes, ignoring"
            )
        self.alibi_slopes = None

        # Parse sliding window
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        # Validate KV cache dtype
        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError(
                "FP8 KV cache is unsupported in Apple attention backends"
            )

    def _compute_attention_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: BaseAppleAttentionMetadata,
        is_causal: bool,
    ) -> torch.Tensor:
        """Compute attention using PyTorch SDPA (Scaled Dot-Product Attention).

        This is the shared implementation used for prefill and encoder attention
        by both Metal and MPS backends.

        Args:
            query: [num_tokens, num_heads, head_size]
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            output: [num_tokens, num_heads, head_size] - pre-allocated output
            attn_metadata: Attention metadata with query_start_loc
            is_causal: Whether to use causal attention mask

        Returns:
            Output tensor with attention results
        """
        # Use cached CPU list to avoid device sync
        start_locs = attn_metadata.query_start_loc_cpu
        num_seqs = len(start_locs) - 1

        # Fast path: single sequence
        if num_seqs == 1:
            q = query.transpose(0, 1)
            k = key.transpose(0, 1)
            v = value.transpose(0, 1)

            # Expand KV heads for GQA
            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            output.copy_(attn_out.squeeze(0).transpose(0, 1))
            return output

        # Multiple sequences: iterate
        for seq_idx in range(num_seqs):
            start = start_locs[seq_idx]
            end = start_locs[seq_idx + 1]
            seq_len = end - start

            if seq_len == 0:
                continue

            q = query[start:end].transpose(0, 1)
            k = key[start:end].transpose(0, 1)
            v = value[start:end].transpose(0, 1)

            # Expand KV heads for GQA
            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            output[start:end] = attn_out.squeeze(0).transpose(0, 1)

        return output

    def _update_kv_cache_scatter(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache using vectorized scatter operations.

        This is a common implementation used by both Metal and MPS backends
        as a fallback or primary method.

        Args:
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            key_cache: [num_blocks, block_size, num_kv_heads, head_size]
            value_cache: [num_blocks, block_size, num_kv_heads, head_size]
            slot_mapping: [num_tokens] - indices into flattened cache
        """
        num_tokens = key.shape[0]
        num_kv_heads_key = key.shape[1]
        head_size_key = key.shape[2]
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

        # Reshape cache to [num_blocks * block_size, num_kv_heads, head_size]
        key_cache_flat = key_cache.view(-1, num_kv_heads, head_size)
        value_cache_flat = value_cache.view(-1, num_kv_heads, head_size)

        # Use scatter_ for fast vectorized updates
        idx = slot_mapping.to(torch.int64).view(-1, 1, 1).expand(
            num_tokens, num_kv_heads_key, head_size_key
        )
        key_cache_flat.scatter_(0, idx, key)
        value_cache_flat.scatter_(0, idx, value)

    @abstractmethod
    def forward(
        self,
        layer: "AttentionLayer",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BaseAppleAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for attention computation.

        Subclasses must implement this method to handle:
        - KV cache updates (if not KV sharing)
        - Prefill path (typically using _compute_attention_sdpa)
        - Decode path (backend-specific optimizations)

        Args:
            layer: Attention layer instance
            query: Query tensor [num_tokens, num_heads, head_size]
            key: Key tensor [num_tokens, num_kv_heads, head_size] (may be None)
            value: Value tensor [num_tokens, num_kv_heads, head_size] (may be None)
            kv_cache: KV cache tensor
            attn_metadata: Attention metadata
            output: Pre-allocated output tensor

        Returns:
            Output tensor with attention results
        """
        raise NotImplementedError


# Re-export for convenience
__all__ = [
    "BaseAppleAttentionBackend",
    "BaseAppleAttentionImpl",
    "BaseAppleAttentionMetadata",
]
