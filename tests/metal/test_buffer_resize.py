# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test Metal buffer auto-resize functionality.

This test verifies that Metal buffers properly auto-resize when batch sizes
exceed the initial allocation, preventing buffer overflow and memory corruption.
"""

import numpy as np
import pytest
import torch

# Skip if Metal not available
pytest.importorskip("Metal")


class TestBufferResize:
    """Test suite for Metal buffer auto-resize functionality."""

    @pytest.fixture
    def metal_kernel(self):
        """Create a MetalPagedAttentionV2 instance with small initial batch size."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import (
            MetalPagedAttentionV2,
            is_metal_available,
        )

        if not is_metal_available():
            pytest.skip("Metal not available")

        # Create kernel with small initial batch size (32) to test resize
        kernel = MetalPagedAttentionV2(
            num_kv_heads=8,
            num_query_heads=8,
            head_size=64,
            block_size=16,
            scale=1.0 / np.sqrt(64),
            max_num_blocks=256,
            max_batch_size=32,  # Small initial size to trigger resize
        )
        return kernel

    def test_initial_batch_size(self, metal_kernel):
        """Verify initial batch size is set correctly."""
        assert metal_kernel.max_batch_size == 32

    def test_resize_not_triggered_under_limit(self, metal_kernel):
        """Verify no resize when batch size is under limit."""
        initial_size = metal_kernel.max_batch_size

        # Call _ensure_batch_buffers with size under limit
        metal_kernel._ensure_batch_buffers(16)

        # Should not have resized
        assert metal_kernel.max_batch_size == initial_size

    def test_resize_at_boundary(self, metal_kernel):
        """Verify resize triggers when batch size exceeds limit."""
        initial_size = metal_kernel.max_batch_size

        # Call _ensure_batch_buffers with size over limit
        metal_kernel._ensure_batch_buffers(64)

        # Should have resized with 1.5x growth
        expected_new_size = int(64 * 1.5)
        assert metal_kernel.max_batch_size == expected_new_size

    def test_resize_512_sequences(self, metal_kernel):
        """Test resize to handle 512 sequences (exceeding default 256)."""
        # Resize to 512
        metal_kernel._ensure_batch_buffers(512)

        # Should have resized
        assert metal_kernel.max_batch_size >= 512

        # Verify buffers are properly allocated (check buffer lengths)
        expected_query_size = metal_kernel.max_batch_size * 8 * 64 * 2  # float16
        assert metal_kernel.query_buf.length() >= expected_query_size

        expected_output_size = metal_kernel.max_batch_size * 8 * 64 * 2  # float16
        assert metal_kernel.output_buf.length() >= expected_output_size

        expected_seq_lens_size = metal_kernel.max_batch_size * 4  # int32
        assert metal_kernel.seq_lens_buf.length() >= expected_seq_lens_size

    def test_resize_sequence_256_257_400_512(self, metal_kernel):
        """Test progressive resizes: 256 -> 257 -> 400 -> 512."""
        # First resize to handle boundary case
        metal_kernel._ensure_batch_buffers(256)
        size_at_256 = metal_kernel.max_batch_size
        assert size_at_256 >= 256

        # Resize just over initial default
        metal_kernel._ensure_batch_buffers(257)
        size_at_257 = metal_kernel.max_batch_size
        assert size_at_257 >= 257

        # Resize to 400
        metal_kernel._ensure_batch_buffers(400)
        size_at_400 = metal_kernel.max_batch_size
        assert size_at_400 >= 400

        # Resize to 512
        metal_kernel._ensure_batch_buffers(512)
        size_at_512 = metal_kernel.max_batch_size
        assert size_at_512 >= 512

    def test_forward_with_large_batch(self, metal_kernel):
        """Test forward pass with batch size exceeding initial allocation."""
        batch_size = 128  # Exceeds initial 32
        num_blocks = 64
        block_size = 16
        num_kv_heads = 8
        num_query_heads = 8
        head_size = 64
        max_blocks_per_seq = 8

        # Create test tensors
        query = torch.randn(batch_size, num_query_heads, head_size, dtype=torch.float16)
        key_cache = torch.randn(
            num_blocks, num_kv_heads, block_size, head_size, dtype=torch.float16
        )
        value_cache = torch.randn(
            num_blocks, num_kv_heads, block_size, head_size, dtype=torch.float16
        )
        block_table = torch.randint(
            0, num_blocks, (batch_size, max_blocks_per_seq), dtype=torch.int32
        )
        seq_lens = torch.randint(1, block_size * 2, (batch_size,), dtype=torch.int32)
        output = torch.zeros_like(query)

        # This should trigger auto-resize and not crash
        result = metal_kernel.forward(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        # Verify we got a result
        assert result.shape == query.shape
        assert metal_kernel.max_batch_size >= batch_size

    def test_forward_with_metal_buffers_large_batch(self, metal_kernel):
        """Test forward_with_metal_buffers with batch size exceeding initial allocation."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        batch_size = 100  # Exceeds initial 32
        num_blocks = 64
        block_size = 16
        num_kv_heads = 8
        num_query_heads = 8
        head_size = 64
        max_blocks_per_seq = 8

        # Create MetalKVCache
        kv_cache = MetalKVCache(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
        )

        # Create test tensors (CPU for forward_with_metal_buffers)
        query = torch.randn(
            batch_size, num_query_heads, head_size, dtype=torch.float16
        ).cpu()
        block_table = torch.randint(
            0, num_blocks, (batch_size, max_blocks_per_seq), dtype=torch.int32
        ).cpu()
        seq_lens = torch.randint(1, block_size * 2, (batch_size,), dtype=torch.int32).cpu()
        output = torch.zeros_like(query)

        # Get buffers for layer 0
        key_buffer, value_buffer = kv_cache.get_buffers(0)

        # This should trigger auto-resize and not crash
        result = metal_kernel.forward_with_metal_buffers(
            query=query,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        # Verify we got a result
        assert result.shape == query.shape
        assert metal_kernel.max_batch_size >= batch_size

    def test_no_memory_corruption_after_resize(self, metal_kernel):
        """Verify no memory corruption after resize by checking output consistency."""
        batch_size = 64
        num_blocks = 32
        block_size = 16
        num_kv_heads = 8
        num_query_heads = 8
        head_size = 64
        max_blocks_per_seq = 4

        # Create deterministic test data
        torch.manual_seed(42)
        query = torch.randn(batch_size, num_query_heads, head_size, dtype=torch.float16)
        key_cache = torch.randn(
            num_blocks, num_kv_heads, block_size, head_size, dtype=torch.float16
        )
        value_cache = torch.randn(
            num_blocks, num_kv_heads, block_size, head_size, dtype=torch.float16
        )
        block_table = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32
        )
        for i in range(batch_size):
            block_table[i] = torch.randint(0, num_blocks, (max_blocks_per_seq,))
        seq_lens = torch.full((batch_size,), block_size, dtype=torch.int32)

        output1 = torch.zeros_like(query)
        output2 = torch.zeros_like(query)

        # Run forward twice with same input
        result1 = metal_kernel.forward(
            query=query.clone(),
            key_cache=key_cache.clone(),
            value_cache=value_cache.clone(),
            block_table=block_table.clone(),
            seq_lens=seq_lens.clone(),
            output=output1,
        )

        result2 = metal_kernel.forward(
            query=query.clone(),
            key_cache=key_cache.clone(),
            value_cache=value_cache.clone(),
            block_table=block_table.clone(),
            seq_lens=seq_lens.clone(),
            output=output2,
        )

        # Results should be identical (no corruption)
        torch.testing.assert_close(result1, result2, rtol=1e-3, atol=1e-3)


class TestMetalAttentionImplBatchSize:
    """Test MetalAttentionImpl default batch size handling."""

    def test_default_initial_batch_size(self):
        """Verify default initial batch size is set correctly."""
        from vllm_apple.v1.attention.backends.metal_attn import MetalAttentionImpl

        # Check the class-level default
        assert MetalAttentionImpl._DEFAULT_INITIAL_BATCH_SIZE == 64

    def test_kernel_uses_default_batch_size(self):
        """Verify kernel creation uses the configurable default."""
        pytest.importorskip("Metal")
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
        )

        # Get or create kernel should use the default
        kernel = impl._get_or_create_metal_kernel(block_size=16, max_num_blocks=256)

        # Kernel should have been created with default initial batch size
        assert kernel.max_batch_size == MetalAttentionImpl._DEFAULT_INITIAL_BATCH_SIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
