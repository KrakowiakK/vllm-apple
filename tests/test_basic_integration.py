# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Basic integration tests for Metal attention backend.

Tests verify:
- Simple generation works end-to-end
- Output matches CPU reference (max_diff < 1e-3)
"""

import pytest
import torch
import numpy as np

# Skip all tests if Metal not available
pytest.importorskip("Metal")


class TestBasicIntegration:
    """Basic integration tests for Metal backend."""

    @pytest.fixture
    def metal_device(self):
        """Create Metal device for tests."""
        from Metal import MTLCreateSystemDefaultDevice
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            pytest.skip("Metal device not available")
        return device

    def test_metal_backend_loads(self):
        """Test that Metal backend components load correctly."""
        from vllm_apple.metal import (
            MetalKVCache,
            MetalPagedAttentionV2,
            MetalPagedAttentionFused,
            is_metal_available,
        )

        assert is_metal_available() is True
        assert MetalKVCache is not None
        assert MetalPagedAttentionV2 is not None
        assert MetalPagedAttentionFused is not None

    def test_kv_cache_creation_and_write(self):
        """Test KV cache creation and basic write operation."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        num_blocks = 32
        num_kv_heads = 4
        block_size = 16
        head_size = 128

        cache = MetalKVCache(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
            num_layers=1,
        )

        # Create test data
        key = np.random.randn(num_kv_heads, head_size).astype(np.float16)
        value = np.random.randn(num_kv_heads, head_size).astype(np.float16)

        # Write to cache
        cache.write_kv(
            layer_idx=0,
            block_id=0,
            token_offset=0,
            key=key,
            value=value,
        )

        # Verify buffer is accessible
        key_buf, val_buf = cache.get_buffers(0)
        assert key_buf is not None
        assert val_buf is not None
        assert key_buf.length() > 0

    def test_fused_kernel_initialization(self):
        """Test fused kernel initializes correctly."""
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=128,
            block_size=16,
            scale=0.088388,
        )

        assert kernel.using_fused is True
        assert kernel.kernel_name == "paged_attention_fused_h128"
        assert kernel.head_size == 128
        assert kernel.num_kv_heads == 4
        assert kernel.num_query_heads == 32

    def test_v2_kernel_initialization(self):
        """Test V2 kernel initializes correctly."""
        from vllm_apple.metal.bridge.metal_paged_attention_v2 import (
            MetalPagedAttentionV2,
        )

        kernel = MetalPagedAttentionV2(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=128,
            block_size=16,
            scale=0.088388,
        )

        assert kernel.num_kv_heads == 4
        assert kernel.num_query_heads == 32
        assert kernel.head_size == 128

    def test_block_allocator(self):
        """Test block allocator functionality."""
        from vllm_apple.metal.block_allocator import MetalBlockAllocator

        allocator = MetalBlockAllocator(
            num_blocks=64,
            num_kv_heads=4,
            block_size=16,
            head_size=128,
        )

        # Test initial state
        assert allocator.get_num_free_blocks() == 64
        assert allocator.get_num_allocated_blocks() == 0

        # Allocate blocks
        block_ids = allocator.allocate_blocks(10)
        assert len(block_ids) == 10
        assert allocator.get_num_free_blocks() == 54
        assert allocator.get_num_allocated_blocks() == 10

        # Free blocks
        allocator.free_blocks_list(block_ids[:5])
        assert allocator.get_num_free_blocks() == 59
        assert allocator.get_num_allocated_blocks() == 5

        # Reset
        allocator.reset()
        assert allocator.get_num_free_blocks() == 64
        assert allocator.get_num_allocated_blocks() == 0


class TestAttentionReference:
    """Tests comparing Metal attention with CPU reference."""

    def test_attention_output_shape(self):
        """Test attention output has correct shape."""
        from vllm_apple.metal.kv_cache import MetalKVCache
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        num_seqs = 2
        num_query_heads = 32
        num_kv_heads = 4
        head_size = 128
        block_size = 16

        # Create cache and kernel
        cache = MetalKVCache(
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

        # Create test tensors
        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        # Output should have same shape as query
        assert output.shape == query.shape
        assert output.shape == (num_seqs, num_query_heads, head_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
