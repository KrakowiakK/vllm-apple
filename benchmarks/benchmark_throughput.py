#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throughput benchmark for Metal attention backend.

Benchmarks:
- Batch sizes: 1, 2, 4, 8
- Reports tok/s for each batch size
- Verifies speedup ~1.4-3.5x for batch≥4
"""

import time
import argparse
import torch
import numpy as np

# Check Metal availability
try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def run_fused_attention_benchmark(
    batch_size: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict:
    """Run fused attention benchmark.

    Args:
        batch_size: Number of sequences
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    from vllm_apple.metal.kv_cache import MetalKVCache
    from vllm_apple.metal.bridge.metal_paged_attention_fused import (
        MetalPagedAttentionFused,
    )

    # Configuration
    num_kv_heads = 4
    num_query_heads = 32
    head_size = 128
    block_size = 16
    seq_len = 128  # Tokens per sequence

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

    # Block table (each sequence uses consecutive blocks)
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    max_blocks = 8
    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    for i in range(batch_size):
        for j in range(min(blocks_per_seq, max_blocks)):
            block_table[i, j] = i * blocks_per_seq + j

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

    key_buffer, value_buffer = cache.get_buffers(0)

    # Warmup
    for _ in range(warmup_iterations):
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
    start_time = time.perf_counter()
    for _ in range(num_iterations):
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
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    tokens_processed = batch_size * num_iterations
    tok_per_sec = tokens_processed / elapsed

    return {
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'elapsed_s': elapsed,
        'tok_s': tok_per_sec,
        'ms_per_iter': (elapsed * 1000) / num_iterations,
        'kernel': kernel.kernel_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Metal attention throughput benchmark")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations"
    )
    args = parser.parse_args()

    if not METAL_AVAILABLE:
        print("ERROR: Metal not available")
        return

    print("=" * 70)
    print("METAL V1.5 THROUGHPUT BENCHMARK")
    print("=" * 70)
    print()

    results = []
    baseline_tok_s = None

    for batch_size in args.batch_sizes:
        print(f"Running batch_size={batch_size}...")
        result = run_fused_attention_benchmark(
            batch_size=batch_size,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
        )
        results.append(result)

        if baseline_tok_s is None:
            baseline_tok_s = result['tok_s']
            scaling = 1.0
        else:
            scaling = result['tok_s'] / baseline_tok_s

        print(f"  tok/s: {result['tok_s']:.1f}")
        print(f"  ms/iter: {result['ms_per_iter']:.3f}")
        print(f"  scaling: {scaling:.2f}x vs batch=1")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch':<8} {'tok/s':<12} {'ms/iter':<12} {'Scaling':<12}")
    print("-" * 44)

    for result in results:
        scaling = result['tok_s'] / baseline_tok_s
        print(f"{result['batch_size']:<8} {result['tok_s']:<12.1f} {result['ms_per_iter']:<12.3f} {scaling:<12.2f}x")

    print()
    print(f"Kernel: {results[0]['kernel']}")
    print()

    # Verify scaling expectations
    print("=" * 70)
    print("SCALING VERIFICATION")
    print("=" * 70)

    expected_min_scaling = {
        1: 1.0,
        2: 1.4,
        4: 2.0,
        8: 2.5,
    }

    all_passed = True
    for result in results:
        batch = result['batch_size']
        actual_scaling = result['tok_s'] / baseline_tok_s
        min_expected = expected_min_scaling.get(batch, 1.0)

        if actual_scaling >= min_expected:
            status = "PASS"
        else:
            status = "WARN"
            all_passed = False

        print(f"batch={batch}: {actual_scaling:.2f}x (expected >= {min_expected:.1f}x) [{status}]")

    print()
    if all_passed:
        print("All scaling expectations met!")
    else:
        print("Some scaling expectations not met (may vary by hardware)")


if __name__ == "__main__":
    main()
