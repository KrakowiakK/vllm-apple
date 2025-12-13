# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Metal PagedAttention kernels.

Consolidated tests for:
- MetalKVCache (unified memory management)
- MetalPagedAttentionV2 (prefill path)
- MetalPagedAttentionFused (decode path)
- Fused kernel correctness with various configurations
"""

import pytest
import torch
import numpy as np

# Skip all tests if Metal not available
pytest.importorskip("Metal")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def metal_device():
    """Create Metal device for tests."""
    from Metal import MTLCreateSystemDefaultDevice
    device = MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("Metal device not available")
    return device


@pytest.fixture
def kv_cache_h128():
    """KV cache fixture with head_size=128."""
    from vllm_apple.metal.kv_cache import MetalKVCache

    return MetalKVCache(
        num_blocks=64,
        num_kv_heads=4,
        block_size=16,
        head_size=128,
        num_layers=1,
    )


@pytest.fixture
def fused_kernel_h128():
    """Fused kernel fixture with head_size=128."""
    from vllm_apple.metal.bridge.metal_paged_attention_fused import (
        MetalPagedAttentionFused,
    )

    return MetalPagedAttentionFused(
        num_kv_heads=4,
        num_query_heads=32,
        head_size=128,
        block_size=16,
    )


# ============================================================================
# MetalKVCache Tests
# ============================================================================


class TestMetalKVCache:
    """Tests for MetalKVCache."""

    def test_kv_cache_creation(self):
        """Test basic KV cache creation."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        cache = MetalKVCache(
            num_blocks=64,
            num_kv_heads=4,
            block_size=16,
            head_size=128,
            num_layers=1,
        )

        assert cache.num_blocks == 64
        assert cache.num_kv_heads == 4
        assert cache.block_size == 16
        assert cache.head_size == 128

    def test_kv_cache_buffers(self):
        """Test MTLBuffer access."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        cache = MetalKVCache(
            num_blocks=32,
            num_kv_heads=4,
            block_size=16,
            head_size=128,
            num_layers=1,
        )

        key_buffer, value_buffer = cache.get_buffers(layer_idx=0)

        # Buffers should be valid MTLBuffer objects
        assert key_buffer is not None
        assert value_buffer is not None

        # Expected size: num_blocks * num_kv_heads * block_size * head_size * 2 bytes
        expected_bytes = 32 * 4 * 16 * 128 * 2
        assert key_buffer.length() == expected_bytes
        assert value_buffer.length() == expected_bytes

    def test_kv_cache_strides(self):
        """Test stride calculations."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        cache = MetalKVCache(
            num_blocks=32,
            num_kv_heads=4,
            block_size=16,
            head_size=128,
            num_layers=1,
        )

        # Strides for layout [num_blocks, num_kv_heads, block_size, head_size]
        assert cache.strides['block'] == 4 * 16 * 128  # num_kv_heads * block_size * head_size
        assert cache.strides['head'] == 16 * 128       # block_size * head_size
        assert cache.strides['token'] == 128           # head_size


# ============================================================================
# MetalPagedAttentionV2 Tests (Prefill)
# ============================================================================


class TestMetalPagedAttentionV2:
    """Tests for MetalPagedAttentionV2 (prefill path)."""

    def test_v2_kernel_creation(self):
        """Test V2 kernel initialization."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import (
            MetalPagedAttentionV2,
        )

        kernel = MetalPagedAttentionV2(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=128,
            block_size=16,
            scale=0.088388347648318,
            max_num_blocks=4096,
        )

        assert kernel.num_kv_heads == 4
        assert kernel.num_query_heads == 32
        assert kernel.head_size == 128
        assert kernel.block_size == 16


# ============================================================================
# MetalPagedAttentionFused Tests (Decode)
# ============================================================================


class TestMetalPagedAttentionFused:
    """Tests for MetalPagedAttentionFused (decode path)."""

    def test_fused_kernel_creation(self):
        """Test fused kernel initialization."""
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=128,
            block_size=16,
            scale=0.088388347648318,
            max_num_blocks=4096,
        )

        assert kernel.num_kv_heads == 4
        assert kernel.num_query_heads == 32
        assert kernel.head_size == 128
        assert kernel.using_fused is True
        assert kernel.kernel_name == "paged_attention_fused_h128"

    def test_fused_kernel_h128_selected(self, fused_kernel_h128):
        """Test that h128 kernel is selected for head_size=128."""
        assert fused_kernel_h128.using_fused is True
        assert fused_kernel_h128.kernel_name == "paged_attention_fused_h128"
        assert fused_kernel_h128.head_size == 128

    def test_fused_kernel_h64_uses_generic(self):
        """Test fused kernel for head_size=64 uses generic (h64 removed in V1.5)."""
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=64,
            block_size=16,
            scale=0.125,
        )

        assert kernel.head_size == 64
        # V1.5: h64 removed, falls back to generic
        assert kernel.kernel_name == "paged_attention_fused_generic"

    def test_fused_kernel_generic_h96(self):
        """Test fused kernel falls back to generic for head_size=96."""
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=96,
            block_size=16,
            scale=0.102062,
        )

        assert kernel.head_size == 96
        assert kernel.kernel_name == "paged_attention_fused_generic"


# ============================================================================
# Fused Kernel Compute Tests
# ============================================================================


class TestFusedKernelCompute:
    """Tests for fused kernel attention computation."""

    def test_fused_kernel_single_sequence(self, kv_cache_h128, fused_kernel_h128):
        """Test fused kernel with single sequence."""
        num_seqs = 1
        seq_len = 10
        num_query_heads = 32
        num_kv_heads = 4
        head_size = 128

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        block_table = torch.zeros(num_seqs, 4, dtype=torch.int32)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)

        key_buffer, value_buffer = kv_cache_h128.get_buffers(0)

        fused_kernel_h128.forward_fused(
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        assert not torch.all(output == 0), "Output should not be all zeros"

    def test_fused_kernel_batch(self, kv_cache_h128, fused_kernel_h128):
        """Test fused kernel with batch of sequences."""
        num_seqs = 4
        num_query_heads = 32
        num_kv_heads = 4
        head_size = 128

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        block_table = torch.tensor([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
        ], dtype=torch.int32)
        seq_lens = torch.tensor([5, 10, 15, 8], dtype=torch.int32)

        key_buffer, value_buffer = kv_cache_h128.get_buffers(0)

        fused_kernel_h128.forward_fused(
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        for i in range(num_seqs):
            assert not torch.all(output[i] == 0), f"Output for seq {i} should not be all zeros"

    def test_fused_kernel_varying_seq_lengths(self, kv_cache_h128, fused_kernel_h128):
        """Test fused kernel handles varying sequence lengths."""
        num_seqs = 3
        num_query_heads = 32
        num_kv_heads = 4
        head_size = 128

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        # Very different sequence lengths (some require multiple blocks)
        block_table = torch.tensor([
            [0, 1, 0, 0],   # 20 tokens = 2 blocks
            [2, 0, 0, 0],   # 5 tokens = 1 block
            [3, 4, 5, 0],   # 40 tokens = 3 blocks
        ], dtype=torch.int32)
        seq_lens = torch.tensor([20, 5, 40], dtype=torch.int32)

        key_buffer, value_buffer = kv_cache_h128.get_buffers(0)

        fused_kernel_h128.forward_fused(
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        for i in range(num_seqs):
            assert not torch.all(output[i] == 0), f"Output for seq {i} should not be all zeros"


# ============================================================================
# Generic Kernel Tests
# ============================================================================


class TestGenericKernel:
    """Tests for generic fused kernel (non-128 head sizes)."""

    def test_generic_kernel_head_size_64(self):
        """Test generic kernel for head_size=64."""
        from vllm_apple.metal.kv_cache import MetalKVCache
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        head_size = 64
        num_kv_heads = 4
        num_query_heads = 32
        block_size = 16

        kv_cache = MetalKVCache(
            num_blocks=32,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
            num_layers=1,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            head_size=head_size,
            block_size=block_size,
        )

        # V1.5: h64 removed, should use generic
        assert kernel.kernel_name == "paged_attention_fused_generic"

    def test_generic_kernel_compute_single_token(self):
        """Generic fused kernel should write KV and attend correctly (seq_len=1).

        For a single-token sequence, attention reduces to selecting that token's
        value vector, repeated across query heads per GQA mapping.
        """
        from vllm_apple.metal.kv_cache import MetalKVCache
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        head_size = 64
        num_seqs = 1
        num_kv_heads = 4
        num_query_heads = 32
        block_size = 16

        kv_cache = MetalKVCache(
            num_blocks=8,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
            num_layers=1,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            head_size=head_size,
            block_size=block_size,
        )
        assert kernel.kernel_name == "paged_attention_fused_generic"

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.empty_like(query)

        # Single token lives in block 0 at offset 0.
        block_table = torch.zeros(num_seqs, 1, dtype=torch.int32)
        seq_lens = torch.tensor([1], dtype=torch.int32)

        key_buffer, value_buffer = kv_cache.get_buffers(0)

        kernel.forward_fused(
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        queries_per_kv = num_query_heads // num_kv_heads
        expected = new_values.repeat_interleave(queries_per_kv, dim=1)
        assert output.shape == expected.shape
        assert torch.allclose(output, expected, atol=0, rtol=0)


# ============================================================================
# Metal Availability Tests
# ============================================================================


class TestMetalAvailability:
    """Tests for Metal availability checks."""

    def test_is_metal_available(self):
        """Test Metal availability detection."""
        from vllm_apple.metal.bridge import is_metal_available

        result = is_metal_available()
        assert isinstance(result, bool)
        # We're running tests on Apple Silicon, so should be True
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
