# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test MPS fallback behavior.

This test verifies that the MPS attention backend is properly selected
when Metal backend requirements are not met, ensuring graceful degradation.
"""

import os
from unittest import mock

import pytest
import torch


class TestPlatformBackendSelection:
    """Test platform backend selection logic."""

    def test_metal_disabled_by_env(self):
        """Test MPS backend is selected when VLLM_METAL_ATTENTION=0."""
        from vllm_apple.platform import ApplePlatform

        with mock.patch.dict(os.environ, {"VLLM_METAL_ATTENTION": "0"}):
            backend = ApplePlatform.get_attn_backend_cls(
                selected_backend=None,
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            )

            assert "mps_attn.MPSAttentionBackend" in backend

    def test_metal_enabled_valid_config(self):
        """Test Metal backend is selected with valid config and VLLM_METAL_ATTENTION=1."""
        from vllm_apple.platform import ApplePlatform

        with mock.patch.dict(os.environ, {"VLLM_METAL_ATTENTION": "1"}):
            backend = ApplePlatform.get_attn_backend_cls(
                selected_backend=None,
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            )

            assert "metal_attn.MetalAttentionBackend" in backend

    def test_fallback_unsupported_head_size(self):
        """Test MPS backend is selected for unsupported head size."""
        from vllm_apple.platform import ApplePlatform

        with mock.patch.dict(os.environ, {"VLLM_METAL_ATTENTION": "1"}):
            # head_size=48 is not in {32, 64, 96, 128}
            backend = ApplePlatform.get_attn_backend_cls(
                selected_backend=None,
                head_size=48,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            )

            assert "mps_attn.MPSAttentionBackend" in backend

    def test_fallback_unsupported_dtype(self):
        """Test MPS backend is selected for unsupported dtype."""
        from vllm_apple.platform import ApplePlatform

        with mock.patch.dict(os.environ, {"VLLM_METAL_ATTENTION": "1"}):
            # float32 is not supported by Metal backend
            backend = ApplePlatform.get_attn_backend_cls(
                selected_backend=None,
                head_size=64,
                dtype=torch.float32,
                kv_cache_dtype="auto",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
            )

            assert "mps_attn.MPSAttentionBackend" in backend

    def test_all_supported_head_sizes(self):
        """Test Metal backend is selected for all supported head sizes."""
        from vllm_apple.platform import ApplePlatform

        supported_sizes = [32, 64, 96, 128]

        with mock.patch.dict(os.environ, {"VLLM_METAL_ATTENTION": "1"}):
            for head_size in supported_sizes:
                backend = ApplePlatform.get_attn_backend_cls(
                    selected_backend=None,
                    head_size=head_size,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=16,
                    use_mla=False,
                    has_sink=False,
                    use_sparse=False,
                )

                assert "metal_attn.MetalAttentionBackend" in backend, (
                    f"Metal backend should be selected for head_size={head_size}"
                )

    def test_unsupported_head_sizes(self):
        """Test MPS backend is selected for unsupported head sizes."""
        from vllm_apple.platform import ApplePlatform

        unsupported_sizes = [16, 48, 80, 112, 256]

        with mock.patch.dict(os.environ, {"VLLM_METAL_ATTENTION": "1"}):
            for head_size in unsupported_sizes:
                backend = ApplePlatform.get_attn_backend_cls(
                    selected_backend=None,
                    head_size=head_size,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=16,
                    use_mla=False,
                    has_sink=False,
                    use_sparse=False,
                )

                assert "mps_attn.MPSAttentionBackend" in backend, (
                    f"MPS backend should be selected for unsupported head_size={head_size}"
                )


class TestMPSBackendFunctionality:
    """Test MPS backend functionality."""

    def test_mps_backend_import(self):
        """Test MPS backend can be imported."""
        from vllm_apple.v1.attention.backends.mps_attn import (
            MPSAttentionBackend,
            MPSAttentionImpl,
            MPSAttentionMetadata,
            MPSAttentionMetadataBuilder,
        )

        assert MPSAttentionBackend.get_name() == "MPS_ATTN"

    def test_mps_backend_supports_all_head_sizes(self):
        """Test MPS backend supports all common head sizes."""
        from vllm_apple.v1.attention.backends.mps_attn import MPSAttentionBackend

        supported_sizes = MPSAttentionBackend.get_supported_head_sizes()

        # Should support many head sizes (PyTorch SDPA is flexible)
        assert 32 in supported_sizes
        assert 64 in supported_sizes
        assert 128 in supported_sizes
        # Also support sizes that Metal doesn't
        assert 48 in supported_sizes
        assert 80 in supported_sizes

    def test_mps_impl_initialization(self):
        """Test MPS attention implementation can be initialized."""
        from vllm_apple.v1.attention.backends.mps_attn import MPSAttentionImpl

        impl = MPSAttentionImpl(
            num_heads=8,
            head_size=48,  # Use a head size not supported by Metal
            scale=1.0 / 48**0.5,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        assert impl.num_heads == 8
        assert impl.head_size == 48
        assert impl.num_kv_heads == 8

    def test_mps_impl_with_gqa(self):
        """Test MPS attention implementation with GQA (grouped query attention)."""
        from vllm_apple.v1.attention.backends.mps_attn import MPSAttentionImpl

        impl = MPSAttentionImpl(
            num_heads=32,
            head_size=64,
            scale=1.0 / 64**0.5,
            num_kv_heads=8,  # GQA: 32 query heads, 8 KV heads
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        assert impl.num_heads == 32
        assert impl.num_kv_heads == 8
        assert impl.num_queries_per_kv == 4  # 32 / 8 = 4


class TestBaseAttentionClasses:
    """Test base attention classes."""

    def test_base_metadata_cpu_caching(self):
        """Test that base metadata caches CPU lists."""
        from vllm_apple.v1.attention.backends.base_attn import BaseAppleAttentionMetadata

        query_start_loc = torch.tensor([0, 10, 25, 40])
        seq_lens = torch.tensor([10, 15, 15])

        metadata = BaseAppleAttentionMetadata(
            num_actual_tokens=40,
            max_query_len=15,
            query_start_loc=query_start_loc,
            max_seq_len=15,
            seq_lens=seq_lens,
            block_table=torch.zeros((3, 2), dtype=torch.int32),
            slot_mapping=torch.arange(40),
            scheduler_metadata=None,
            causal=True,
            num_decode_tokens=0,
        )

        # First access should compute and cache
        start_locs_1 = metadata.query_start_loc_cpu
        seq_lens_1 = metadata.seq_lens_cpu

        # Second access should return cached version
        start_locs_2 = metadata.query_start_loc_cpu
        seq_lens_2 = metadata.seq_lens_cpu

        # Should be the same object (cached)
        assert start_locs_1 is start_locs_2
        assert seq_lens_1 is seq_lens_2

        # Values should be correct
        assert start_locs_1 == [0, 10, 25, 40]
        assert seq_lens_1 == [10, 15, 15]


class TestMetalNotTouched:
    """Verify Metal is not touched when MPS backend is used."""

    def test_mps_impl_has_no_metal_cache(self):
        """Test MPS implementation does not have Metal-specific state."""
        from vllm_apple.v1.attention.backends.mps_attn import MPSAttentionImpl

        impl = MPSAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        # Should NOT have Metal-specific attributes
        assert not hasattr(impl, "_metal_kv_cache")
        assert not hasattr(impl, "_metal_kernel")
        assert not hasattr(impl, "_fused_kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
