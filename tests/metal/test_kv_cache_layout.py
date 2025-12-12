# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV cache memory layout and strides.

Tests verify:
- Correct memory layout [num_blocks, num_kv_heads, block_size, head_size]
- Stride calculations are correct
- Different head_size and block_size configurations work
"""

import pytest
import numpy as np

# Skip all tests if Metal not available
pytest.importorskip("Metal")


class TestKVCacheLayout:
    """Tests for KV cache memory layout."""

    def test_layout_default(self):
        """Test default layout [num_blocks, num_kv_heads, block_size, head_size]."""
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

        # Verify strides
        assert cache.strides['block'] == num_kv_heads * block_size * head_size
        assert cache.strides['head'] == block_size * head_size
        assert cache.strides['token'] == head_size
        assert cache.strides['element'] == 1

    def test_buffer_size(self):
        """Test buffer size is correctly calculated."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        num_blocks = 64
        num_kv_heads = 8
        block_size = 16
        head_size = 128
        dtype_size = 2  # float16

        cache = MetalKVCache(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
            num_layers=1,
        )

        expected_bytes = num_blocks * num_kv_heads * block_size * head_size * dtype_size
        key_buf, val_buf = cache.get_buffers(0)

        assert key_buf.length() == expected_bytes
        assert val_buf.length() == expected_bytes

    def test_write_and_read_roundtrip(self):
        """Test writing to cache and verifying data."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        num_blocks = 16
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

        # Create unique test data
        key = np.ones((num_kv_heads, head_size), dtype=np.float16) * 1.5
        value = np.ones((num_kv_heads, head_size), dtype=np.float16) * 2.5

        # Write to specific location
        block_id = 3
        token_offset = 5
        cache.write_kv(
            layer_idx=0,
            block_id=block_id,
            token_offset=token_offset,
            key=key,
            value=value,
        )

        # Read back and verify
        key_buf, val_buf = cache.get_buffers(0)
        key_mv = key_buf.contents().as_buffer(key_buf.length())
        val_mv = val_buf.contents().as_buffer(val_buf.length())

        # Calculate expected offset for head 0
        stride_block = cache.strides['block']
        stride_head = cache.strides['head']
        stride_token = cache.strides['token']
        dtype_size = 2

        for head_idx in range(num_kv_heads):
            element_offset = (
                block_id * stride_block +
                head_idx * stride_head +
                token_offset * stride_token
            )
            byte_offset = element_offset * dtype_size

            # Read back data
            read_key = np.frombuffer(
                key_mv[byte_offset:byte_offset + head_size * dtype_size],
                dtype=np.float16
            )
            read_value = np.frombuffer(
                val_mv[byte_offset:byte_offset + head_size * dtype_size],
                dtype=np.float16
            )

            # Verify
            np.testing.assert_array_almost_equal(read_key, key[head_idx], decimal=3)
            np.testing.assert_array_almost_equal(read_value, value[head_idx], decimal=3)

    def test_different_head_sizes(self):
        """Test cache with different head sizes."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        for head_size in [64, 96, 128]:
            cache = MetalKVCache(
                num_blocks=16,
                num_kv_heads=4,
                block_size=16,
                head_size=head_size,
                num_layers=1,
            )

            assert cache.strides['token'] == head_size
            assert cache.head_size == head_size

    def test_different_block_sizes(self):
        """Test cache with different block sizes."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        for block_size in [8, 16, 32]:
            cache = MetalKVCache(
                num_blocks=16,
                num_kv_heads=4,
                block_size=block_size,
                head_size=128,
                num_layers=1,
            )

            assert cache.strides['head'] == block_size * 128
            assert cache.block_size == block_size

    def test_multi_layer_cache(self):
        """Test cache with multiple layers."""
        from vllm_apple.metal.kv_cache import MetalKVCache

        num_layers = 4

        cache = MetalKVCache(
            num_blocks=16,
            num_kv_heads=4,
            block_size=16,
            head_size=128,
            num_layers=num_layers,
        )

        assert cache.num_layers == num_layers
        assert len(cache.key_buffers) == num_layers
        assert len(cache.value_buffers) == num_layers

        # Verify each layer has independent buffers
        for layer_idx in range(num_layers):
            key_buf, val_buf = cache.get_buffers(layer_idx)
            assert key_buf is not None
            assert val_buf is not None


class TestBlockAllocatorStrides:
    """Tests for block allocator stride calculations."""

    def test_allocator_strides(self):
        """Test block allocator stride calculations."""
        from vllm_apple.metal.block_allocator import MetalBlockAllocator

        num_kv_heads = 4
        block_size = 16
        head_size = 128

        allocator = MetalBlockAllocator(
            num_blocks=32,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
        )

        assert allocator.strides['block'] == num_kv_heads * block_size * head_size
        assert allocator.strides['head'] == block_size * head_size
        assert allocator.strides['token'] == head_size

    def test_block_offset_calculation(self):
        """Test block offset calculations."""
        from vllm_apple.metal.block_allocator import MetalBlockAllocator

        num_kv_heads = 4
        block_size = 16
        head_size = 128
        dtype_size = 2

        allocator = MetalBlockAllocator(
            num_blocks=32,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
        )

        # Block 0 should be at offset 0
        assert allocator.get_block_offset(0) == 0

        # Block 1 should be at block_size_bytes
        expected_bytes = num_kv_heads * block_size * head_size * dtype_size
        assert allocator.get_block_offset(1) == expected_bytes

        # Block N should be at N * block_size_bytes
        for n in range(10):
            assert allocator.get_block_offset(n) == n * expected_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
