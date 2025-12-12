# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test KV cache sharing functionality.

This test verifies that KV cache sharing between layers works correctly,
particularly for encoder-decoder models where some layers share KV cache
with other layers.
"""

import pytest
import torch

# Skip if Metal not available
pytest.importorskip("Metal")


class TestKVSharingFlag:
    """Test _is_kv_shared flag behavior."""

    def test_shared_layer_flag_set(self):
        """Test _is_kv_shared is True when kv_sharing_target_layer_name is set."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import is_metal_available

        if not is_metal_available():
            pytest.skip("Metal not available")

        from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionImpl

        impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="encoder.layer_0",  # Shared with encoder
        )

        assert impl._is_kv_shared is True
        assert impl.kv_sharing_target_layer_name == "encoder.layer_0"

    def test_non_shared_layer_flag_not_set(self):
        """Test _is_kv_shared is False when no sharing."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import is_metal_available

        if not is_metal_available():
            pytest.skip("Metal not available")

        from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionImpl

        impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name=None,  # Not shared
        )

        assert impl._is_kv_shared is False
        assert impl.kv_sharing_target_layer_name is None


class TestMetalKVCacheAllocation:
    """Test MetalKVCache allocation skipping for shared layers."""

    def test_shared_layer_skips_cache_init(self):
        """Test that shared layers skip MetalKVCache initialization."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import is_metal_available

        if not is_metal_available():
            pytest.skip("Metal not available")

        from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionImpl

        impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="encoder.layer_0",
        )

        # Shared layer should have _metal_kv_cache_initialized = True
        # (to skip initialization) but _metal_kv_cache = None (no allocation)
        assert impl._metal_kv_cache_initialized is True
        assert impl._metal_kv_cache is None

    def test_non_shared_layer_allocates_cache(self):
        """Test that non-shared layers allocate MetalKVCache when initialized."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import is_metal_available

        if not is_metal_available():
            pytest.skip("Metal not available")

        from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionImpl

        impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name=None,
        )

        # Non-shared layer should start with _metal_kv_cache_initialized = False
        assert impl._metal_kv_cache_initialized is False
        assert impl._metal_kv_cache is None

        # After calling _get_or_create_metal_kv_cache, cache should be allocated
        impl._get_or_create_metal_kv_cache(num_blocks=64, block_size=16)
        assert impl._metal_kv_cache_initialized is True
        assert impl._metal_kv_cache is not None


class TestKVSharingAssertions:
    """Test safety assertions for KV sharing."""

    def test_shared_layer_cannot_use_metal_zero_copy(self):
        """Test that shared layer raises assertion when trying to use Metal zero-copy."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import is_metal_available

        if not is_metal_available():
            pytest.skip("Metal not available")

        from vllm_apple.v1.attention.backends.metal_attn import (
            MetalAttentionImpl,
            MetalAttentionMetadata,
        )

        impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="encoder.layer_0",
        )

        # Create dummy tensors and metadata
        query = torch.randn(1, 8, 64, dtype=torch.float16)
        output = torch.zeros_like(query)
        metadata = MetalAttentionMetadata(
            num_actual_tokens=1,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1]),
            max_seq_len=10,
            seq_lens=torch.tensor([10]),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.tensor([0]),
            scheduler_metadata=None,
            causal=True,
            num_decode_tokens=1,
        )

        # Should raise assertion error
        with pytest.raises(AssertionError, match="KV sharing layer"):
            impl._compute_attention_with_metal_zero_copy(query, output, metadata)


class TestKVSharingMemory:
    """Test memory usage with KV sharing."""

    def test_no_double_allocation(self):
        """Test that shared layers don't allocate memory for KV cache."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import is_metal_available

        if not is_metal_available():
            pytest.skip("Metal not available")

        from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionImpl

        # Create non-shared (target) layer
        target_impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name=None,
        )

        # Initialize target layer's cache
        target_impl._get_or_create_metal_kv_cache(num_blocks=64, block_size=16)

        # Create shared layer
        shared_impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="target_layer",
        )

        # Target layer should have cache
        assert target_impl._metal_kv_cache is not None

        # Shared layer should NOT have cache (no double allocation)
        assert shared_impl._metal_kv_cache is None
        assert shared_impl._metal_kv_cache_initialized is True  # Skip flag set


class TestMPSKVSharing:
    """Test KV sharing with MPS backend."""

    def test_mps_impl_kv_sharing_flag(self):
        """Test MPS implementation handles KV sharing flag."""
        from vllm_apple.v1.attention.backends.mps_attn import MPSAttentionImpl

        impl = MPSAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="encoder.layer_0",
        )

        assert impl._is_kv_shared is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
