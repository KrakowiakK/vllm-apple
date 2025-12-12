#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Metal attention kernel performance.

This example benchmarks the Metal PagedAttention kernel across
different batch sizes to measure throughput and scaling.

Usage:
    python examples/benchmark_metal_attention.py

Requirements:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - vllm-apple plugin installed
"""

import time
import torch

# Check Metal availability
try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("ERROR: Metal not available. Install pyobjc-framework-Metal.")
    exit(1)


def benchmark_fused_attention(batch_sizes=[1, 2, 4, 8], iterations=50):
    """Benchmark fused attention kernel."""
    from vllm_apple.metal.kv_cache import MetalKVCache
    from vllm_apple.metal.bridge.metal_paged_attention_fused import (
        MetalPagedAttentionFused,
    )

    # Configuration (typical for LLaMA/Qwen models)
    num_kv_heads = 4
    num_query_heads = 32
    head_size = 128
    block_size = 16
    seq_len = 128

    print("=" * 60)
    print("Metal Fused Attention Benchmark")
    print("=" * 60)
    print(f"Config: num_kv_heads={num_kv_heads}, num_query_heads={num_query_heads}")
    print(f"        head_size={head_size}, block_size={block_size}")
    print(f"        seq_len={seq_len}, iterations={iterations}")
    print("=" * 60)
    print()

    results = []
    baseline_tps = None

    for batch_size in batch_sizes:
        # Create KV cache
        num_blocks = batch_size * ((seq_len + block_size - 1) // block_size) + 10
        cache = MetalKVCache(
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_size=head_size,
            num_layers=1,
        )

        # Create fused kernel
        kernel = MetalPagedAttentionFused(
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            head_size=head_size,
            block_size=block_size,
        )

        # Create test tensors
        query = torch.randn(batch_size, num_query_heads, head_size, dtype=torch.float16)
        new_keys = torch.randn(batch_size, num_kv_heads, head_size, dtype=torch.float16)
        new_values = torch.randn(batch_size, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.zeros_like(query)

        # Block table
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        block_table = torch.zeros(batch_size, 8, dtype=torch.int32)
        for i in range(batch_size):
            for j in range(min(blocks_per_seq, 8)):
                block_table[i, j] = i * blocks_per_seq + j

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
        key_buffer, value_buffer = cache.get_buffers(0)

        # Warmup
        for _ in range(10):
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

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
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
        elapsed = time.perf_counter() - start

        tokens_per_sec = (batch_size * iterations) / elapsed
        ms_per_iter = (elapsed * 1000) / iterations

        if baseline_tps is None:
            baseline_tps = tokens_per_sec
            scaling = 1.0
        else:
            scaling = tokens_per_sec / baseline_tps

        results.append({
            'batch': batch_size,
            'tps': tokens_per_sec,
            'ms': ms_per_iter,
            'scaling': scaling,
        })

        print(f"batch={batch_size:2d}: {tokens_per_sec:8.1f} tok/s, "
              f"{ms_per_iter:.3f} ms/iter, {scaling:.2f}x scaling")

    print()
    print(f"Kernel: {kernel.kernel_name}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    benchmark_fused_attention()
