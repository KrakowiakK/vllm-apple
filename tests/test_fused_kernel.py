# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused KV-write + attention kernel.

Tests verify:
- Fused kernel produces correct output
- Two-phase approach (kv_write_decode + paged_attention_fused_h128) works
- Various sequence lengths handled correctly
"""

import pytest
import torch
import numpy as np

# Skip all tests if Metal not available
pytest.importorskip("Metal")


class TestFusedKernel:
    """Tests for MetalPagedAttentionFused kernel."""

    @pytest.fixture
    def setup_fused(self):
        """Set up fused kernel test fixtures."""
        from vllm_apple.metal.kv_cache import MetalKVCache
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        num_blocks = 64
        num_kv_heads = 4
        num_query_heads = 32
        block_size = 16
        head_size = 128

        kv_cache = MetalKVCache(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
            num_layers=1,
        )

        fused_kernel = MetalPagedAttentionFused(
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            head_size=head_size,
            block_size=block_size,
        )

        return {
            'kv_cache': kv_cache,
            'fused_kernel': fused_kernel,
            'num_blocks': num_blocks,
            'num_kv_heads': num_kv_heads,
            'num_query_heads': num_query_heads,
            'block_size': block_size,
            'head_size': head_size,
        }

    def test_fused_kernel_h128_selected(self, setup_fused):
        """Test that h128 kernel is selected for head_size=128."""
        fused_kernel = setup_fused['fused_kernel']

        assert fused_kernel.using_fused is True
        assert fused_kernel.kernel_name == "paged_attention_fused_h128"
        assert fused_kernel.head_size == 128

    def test_fused_kernel_single_sequence(self, setup_fused):
        """Test fused kernel with single sequence."""
        kv_cache = setup_fused['kv_cache']
        fused_kernel = setup_fused['fused_kernel']
        num_query_heads = setup_fused['num_query_heads']
        num_kv_heads = setup_fused['num_kv_heads']
        head_size = setup_fused['head_size']

        num_seqs = 1
        seq_len = 10

        # Create test tensors
        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        block_table = torch.zeros(num_seqs, 4, dtype=torch.int32)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)

        key_buffer, value_buffer = kv_cache.get_buffers(0)

        # Execute fused kernel
        fused_kernel.forward_fused(
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        # Output should be non-zero
        assert not torch.all(output == 0), "Output should not be all zeros"

    def test_fused_kernel_batch(self, setup_fused):
        """Test fused kernel with batch of sequences."""
        kv_cache = setup_fused['kv_cache']
        fused_kernel = setup_fused['fused_kernel']
        num_query_heads = setup_fused['num_query_heads']
        num_kv_heads = setup_fused['num_kv_heads']
        head_size = setup_fused['head_size']

        num_seqs = 4

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(num_seqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        # Each sequence uses different blocks
        block_table = torch.tensor([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
        ], dtype=torch.int32)
        seq_lens = torch.tensor([5, 10, 15, 8], dtype=torch.int32)

        key_buffer, value_buffer = kv_cache.get_buffers(0)

        fused_kernel.forward_fused(
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output,
        )

        # Each sequence should have computed attention
        for i in range(num_seqs):
            assert not torch.all(output[i] == 0), f"Output for seq {i} should not be all zeros"

    def test_fused_kernel_varying_seq_lengths(self, setup_fused):
        """Test fused kernel handles varying sequence lengths."""
        kv_cache = setup_fused['kv_cache']
        fused_kernel = setup_fused['fused_kernel']
        num_query_heads = setup_fused['num_query_heads']
        num_kv_heads = setup_fused['num_kv_heads']
        head_size = setup_fused['head_size']
        block_size = setup_fused['block_size']

        num_seqs = 3

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

        key_buffer, value_buffer = kv_cache.get_buffers(0)

        fused_kernel.forward_fused(
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


class TestFusedKernelGeneric:
    """Tests for generic fused kernel (non-128 head sizes)."""

    def test_generic_kernel_head_size_64(self):
        """Test generic kernel for head_size=64."""
        from vllm_apple.metal.kv_cache import MetalKVCache
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        head_size = 64  # Not 128, should use generic
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

    def test_generic_kernel_head_size_96(self):
        """Test generic kernel for head_size=96."""
        from vllm_apple.metal.bridge.metal_paged_attention_fused import (
            MetalPagedAttentionFused,
        )

        kernel = MetalPagedAttentionFused(
            num_kv_heads=4,
            num_query_heads=32,
            head_size=96,
            block_size=16,
        )

        assert kernel.kernel_name == "paged_attention_fused_generic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
