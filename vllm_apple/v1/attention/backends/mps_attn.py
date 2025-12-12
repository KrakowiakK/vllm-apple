# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS Attention Backend for vLLM on Apple Silicon.

This backend provides a pure PyTorch SDPA-based attention implementation
for Apple Silicon. It serves as a fallback when:
- VLLM_METAL_ATTENTION=0
- Head size is not supported by Metal kernel
- Dtype is not float16

Unlike the Metal backend, this backend:
- Does NOT use custom Metal compute shaders
- Does NOT require Metal-specific dependencies
- Uses standard PyTorch SDPA for all operations
- Works with any head size supported by PyTorch

Performance characteristics:
- Prefill: Similar to Metal backend (both use SDPA)
- Decode: Slower than Metal (no custom kernel optimization)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (
    AttentionLayer,
    AttentionType,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from .base_attn import (
    BaseAppleAttentionBackend,
    BaseAppleAttentionImpl,
    BaseAppleAttentionMetadata,
)

logger = init_logger(__name__)


@dataclass
class MPSAttentionMetadata(BaseAppleAttentionMetadata):
    """Metadata for MPS attention.

    Inherits from BaseAppleAttentionMetadata with cached CPU properties.
    """

    pass


class MPSAttentionBackend(BaseAppleAttentionBackend):
    """MPS attention backend using PyTorch SDPA.

    This is a fallback backend that uses only PyTorch operations,
    without any custom Metal shaders. It works with any configuration
    supported by PyTorch SDPA.
    """

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # PyTorch SDPA supports any head size
        return [16, 32, 48, 64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_impl_cls() -> type["MPSAttentionImpl"]:
        return MPSAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MPSAttentionMetadataBuilder"]:
        return MPSAttentionMetadataBuilder


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    """Builder for MPS attention metadata."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable reorder batch to separate decode and prefill tokens
        self._init_reorder_batch_threshold(1, False)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MPSAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Split decode and prefill tokens
        num_decode_tokens = 0
        if causal:
            assert self.reorder_batch_threshold is not None
            (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=True,
                )
            )

        return MPSAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            scheduler_metadata=None,
            causal=causal,
            num_decode_tokens=num_decode_tokens,
        )


class MPSAttentionImpl(BaseAppleAttentionImpl):
    """MPS attention implementation using PyTorch SDPA.

    This implementation uses PyTorch scaled_dot_product_attention for
    both prefill and decode operations. It does not have the same
    decode performance as the Metal backend but works with any
    configuration.
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
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **kwargs,
        )

        logger.info_once(
            f"Using MPS attention backend (SDPA) for head_size={head_size}"
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using PyTorch SDPA for all operations.

        Args:
            layer: Attention layer instance
            query: Query tensor [num_tokens, num_heads, head_size]
            key: Key tensor [num_tokens, num_kv_heads, head_size] (may be None)
            value: Value tensor [num_tokens, num_kv_heads, head_size] (may be None)
            kv_cache: KV cache tensor [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Attention metadata
            output: Pre-allocated output tensor

        Returns:
            Output tensor with attention results
        """
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder-only attention (no KV cache)
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return self._compute_attention_sdpa(
                query[:num_actual_tokens],
                key[:num_actual_tokens] if key is not None else query[:num_actual_tokens],
                value[:num_actual_tokens] if value is not None else query[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                is_causal=attn_metadata.causal,
            )

        # Split KV cache
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Update KV cache if not sharing with another layer
        if not self._is_kv_shared and key is not None and value is not None:
            self._update_kv_cache_scatter(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata.slot_mapping[:num_actual_tokens],
            )

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = num_actual_tokens - num_decode_tokens

        # Process prefill tokens using SDPA
        if num_prefill_tokens > 0:
            self._compute_attention_sdpa(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens] if key is not None else None,
                value[num_decode_tokens:num_actual_tokens] if value is not None else None,
                output[num_decode_tokens:num_actual_tokens],
                attn_metadata,
                is_causal=True,
            )

        # Process decode tokens using SDPA with KV cache
        if num_decode_tokens > 0:
            self._compute_decode_sdpa(
                query[:num_decode_tokens],
                key_cache,
                value_cache,
                output[:num_decode_tokens],
                attn_metadata,
            )

        return output

    def _compute_decode_sdpa(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
    ) -> torch.Tensor:
        """Compute decode attention using SDPA with KV cache.

        This is a simple implementation that reads from the KV cache
        and computes attention for each decode token.

        Args:
            query: [num_decode_tokens, num_heads, head_size]
            key_cache: [num_blocks, block_size, num_kv_heads, head_size]
            value_cache: [num_blocks, block_size, num_kv_heads, head_size]
            output: [num_decode_tokens, num_heads, head_size]
            attn_metadata: Attention metadata

        Returns:
            Output tensor with attention results
        """
        num_decode_tokens = query.shape[0]
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens_cpu

        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

        # Process each decode token
        for seq_idx in range(num_decode_tokens):
            seq_len = seq_lens[seq_idx]
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            # Gather KV from cache blocks
            blocks = block_table[seq_idx, :num_blocks_for_seq]

            # Gather key and value from cache
            # Shape: [num_blocks_for_seq, block_size, num_kv_heads, head_size]
            k_blocks = key_cache[blocks]
            v_blocks = value_cache[blocks]

            # Reshape to [seq_len, num_kv_heads, head_size]
            k = k_blocks.view(-1, num_kv_heads, head_size)[:seq_len]
            v = v_blocks.view(-1, num_kv_heads, head_size)[:seq_len]

            # Query for this token: [1, num_heads, head_size]
            q = query[seq_idx:seq_idx + 1]

            # Transpose for SDPA: [num_heads, 1, head_size], [num_kv_heads, seq_len, head_size]
            q_t = q.transpose(0, 1)
            k_t = k.transpose(0, 1)
            v_t = v.transpose(0, 1)

            # Expand KV heads for GQA
            if self.num_kv_heads != self.num_heads:
                k_t = k_t.repeat_interleave(self.num_queries_per_kv, dim=0)
                v_t = v_t.repeat_interleave(self.num_queries_per_kv, dim=0)

            # Compute attention using SDPA (single query attending to all KV)
            attn_out = F.scaled_dot_product_attention(
                q_t.unsqueeze(0),  # [1, num_heads, 1, head_size]
                k_t.unsqueeze(0),  # [1, num_heads, seq_len, head_size]
                v_t.unsqueeze(0),  # [1, num_heads, seq_len, head_size]
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,  # Decode is not causal (single query)
                scale=self.scale,
            )

            # Store result: [1, num_heads, head_size]
            output[seq_idx] = attn_out.squeeze(0).transpose(0, 1).squeeze(0)

        return output


# Re-export for convenience
__all__ = [
    "MPSAttentionBackend",
    "MPSAttentionImpl",
    "MPSAttentionMetadata",
    "MPSAttentionMetadataBuilder",
]
