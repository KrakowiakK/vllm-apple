#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete Engine Benchmark for vLLM-Apple Metal Engine v2.0.

Benchmarks the Metal engine kernels directly without requiring model downloads.
Tests both prefill and decode performance across different configurations.

Metrics:
- Prefill throughput (tok/s) - token-parallel attention
- Decode throughput (tok/s) - fused KV write + attention
- Latency (ms/step)
- Scaling efficiency

Configurations:
- Llama architecture (head_size=128, 32 query heads, 4 KV heads)
- Qwen architecture (head_size=128, 28 query heads, 4 KV heads)
"""

import os
import time
import argparse
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

# Try to import Metal
try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("WARNING: Metal not available")


@dataclass
class BenchmarkConfig:
    """Model architecture configuration."""
    name: str
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int = 16


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    batch_size: int
    seq_len: int
    prefill_tok_s: float
    decode_tok_s: float
    prefill_ms: float
    decode_ms: float
    kernel_type: str


# Model configurations to test
CONFIGS = {
    "llama": BenchmarkConfig(
        name="Llama-style",
        num_query_heads=32,
        num_kv_heads=4,  # GQA with 8:1 ratio
        head_size=128,
    ),
    "qwen": BenchmarkConfig(
        name="Qwen-style",
        num_query_heads=28,
        num_kv_heads=4,  # GQA with 7:1 ratio
        head_size=128,
    ),
}


def run_decode_benchmark(
    config: BenchmarkConfig,
    batch_size: int,
    seq_len: int,
    num_iterations: int = 200,
    warmup_iterations: int = 20,
) -> Dict:
    """Run decode (fused) kernel benchmark.

    Args:
        config: Model configuration
        batch_size: Number of sequences
        seq_len: Sequence length (context)
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    from vllm_apple.metal.kv_cache import MetalKVCache
    from vllm_apple.metal.bridge.metal_paged_attention_fused import (
        MetalPagedAttentionFused,
    )

    # Create KV cache
    blocks_per_seq = (seq_len + config.block_size - 1) // config.block_size
    num_blocks = batch_size * blocks_per_seq + 10
    cache = MetalKVCache(
        num_blocks=num_blocks,
        num_kv_heads=config.num_kv_heads,
        block_size=config.block_size,
        head_size=config.head_size,
        num_layers=1,
    )

    # Create fused kernel
    kernel = MetalPagedAttentionFused(
        num_kv_heads=config.num_kv_heads,
        num_query_heads=config.num_query_heads,
        head_size=config.head_size,
        block_size=config.block_size,
    )

    # Create test tensors (decode = 1 new token per sequence)
    query = torch.randn(batch_size, config.num_query_heads, config.head_size, dtype=torch.float16)
    new_keys = torch.randn(batch_size, config.num_kv_heads, config.head_size, dtype=torch.float16)
    new_values = torch.randn(batch_size, config.num_kv_heads, config.head_size, dtype=torch.float16)
    output = torch.zeros_like(query)

    # Block table
    max_blocks = min(blocks_per_seq + 2, 64)
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
    tokens_processed = batch_size * num_iterations  # 1 token per seq per iteration
    tok_per_sec = tokens_processed / elapsed
    ms_per_iter = (elapsed * 1000) / num_iterations

    return {
        'tok_s': tok_per_sec,
        'ms_per_iter': ms_per_iter,
        'kernel': kernel.kernel_name,
    }


def run_prefill_benchmark(
    config: BenchmarkConfig,
    batch_size: int,
    prompt_len: int,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> Dict:
    """Run prefill (token-parallel) kernel benchmark.

    Args:
        config: Model configuration
        batch_size: Number of sequences
        prompt_len: Prompt length per sequence
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    from vllm_apple.metal.kv_cache import MetalKVCache
    from vllm_apple.metal.bridge.metal_paged_attention_v2 import (
        MetalPagedAttentionV2,
    )

    # Total tokens in batch
    num_tokens = batch_size * prompt_len

    # Create KV cache
    blocks_per_seq = (prompt_len + config.block_size - 1) // config.block_size
    num_blocks = batch_size * blocks_per_seq + 10
    cache = MetalKVCache(
        num_blocks=num_blocks,
        num_kv_heads=config.num_kv_heads,
        block_size=config.block_size,
        head_size=config.head_size,
        num_layers=1,
    )

    # Create V2 kernel (token-parallel prefill)
    kernel = MetalPagedAttentionV2(
        num_kv_heads=config.num_kv_heads,
        num_query_heads=config.num_query_heads,
        head_size=config.head_size,
        block_size=config.block_size,
        scale=1.0 / (config.head_size ** 0.5),
        layer_idx=0,
    )

    # Create test tensors (prefill = all tokens at once)
    query = torch.randn(num_tokens, config.num_query_heads, config.head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, config.num_kv_heads, config.head_size, dtype=torch.float16)
    value = torch.randn(num_tokens, config.num_kv_heads, config.head_size, dtype=torch.float16)
    output = torch.zeros_like(query)

    # Query start locations (cumulative token counts)
    query_start_locs = torch.arange(0, batch_size + 1, dtype=torch.int32) * prompt_len

    # Block table
    max_blocks = min(blocks_per_seq + 2, 64)
    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    for i in range(batch_size):
        for j in range(min(blocks_per_seq, max_blocks)):
            block_table[i, j] = i * blocks_per_seq + j

    # Context lens (full prompt for each)
    context_lens = torch.full((batch_size,), prompt_len, dtype=torch.int32)

    key_buffer, value_buffer = cache.get_buffers(0)

    # Warmup
    for _ in range(warmup_iterations):
        kernel.forward(
            query=query,
            key=key,
            value=value,
            key_cache=key_buffer,
            value_cache=value_buffer,
            block_table=block_table,
            context_lens=context_lens,
            query_start_locs=query_start_locs,
            output=output,
        )

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        kernel.forward(
            query=query,
            key=key,
            value=value,
            key_cache=key_buffer,
            value_cache=value_buffer,
            block_table=block_table,
            context_lens=context_lens,
            query_start_locs=query_start_locs,
            output=output,
        )
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    tokens_processed = num_tokens * num_iterations
    tok_per_sec = tokens_processed / elapsed
    ms_per_iter = (elapsed * 1000) / num_iterations

    return {
        'tok_s': tok_per_sec,
        'ms_per_iter': ms_per_iter,
        'kernel': 'paged_attention_v2_prefill',
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM-Apple Engine Complete Benchmark")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length for decode benchmark"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=128,
        help="Prompt length for prefill benchmark"
    )
    parser.add_argument(
        "--configs", type=str, nargs="+",
        default=["llama", "qwen"],
        help="Model configurations to test"
    )
    parser.add_argument(
        "--decode-iterations", type=int, default=200,
        help="Number of decode benchmark iterations"
    )
    parser.add_argument(
        "--prefill-iterations", type=int, default=50,
        help="Number of prefill benchmark iterations"
    )
    args = parser.parse_args()

    if not METAL_AVAILABLE:
        print("ERROR: Metal not available")
        return

    print("=" * 80)
    print("vLLM-Apple Metal Engine v2.0 - Complete Benchmark")
    print("=" * 80)
    print()

    all_results = []

    for config_name in args.configs:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}")
            continue

        config = CONFIGS[config_name]
        print(f"\n{'='*70}")
        print(f"Configuration: {config.name}")
        print(f"  Query heads: {config.num_query_heads}")
        print(f"  KV heads: {config.num_kv_heads}")
        print(f"  Head size: {config.head_size}")
        print(f"  GQA ratio: {config.num_query_heads // config.num_kv_heads}:1")
        print(f"{'='*70}\n")

        print("DECODE BENCHMARK (Fused KV-write + Attention)")
        print("-" * 60)

        decode_baseline = None
        for batch_size in args.batch_sizes:
            result = run_decode_benchmark(
                config=config,
                batch_size=batch_size,
                seq_len=args.seq_len,
                num_iterations=args.decode_iterations,
            )
            if decode_baseline is None:
                decode_baseline = result['tok_s']
                scaling = 1.0
            else:
                scaling = result['tok_s'] / decode_baseline

            print(f"  Batch {batch_size:2d}: {result['tok_s']:10.1f} tok/s  {result['ms_per_iter']:6.3f} ms/iter  {scaling:5.2f}x scaling")

            all_results.append(BenchmarkResult(
                config_name=config_name,
                batch_size=batch_size,
                seq_len=args.seq_len,
                prefill_tok_s=0,
                decode_tok_s=result['tok_s'],
                prefill_ms=0,
                decode_ms=result['ms_per_iter'],
                kernel_type=result['kernel'],
            ))

        print()
        print("PREFILL BENCHMARK (Token-parallel Attention)")
        print("-" * 60)

        prefill_baseline = None
        for batch_size in args.batch_sizes:
            try:
                result = run_prefill_benchmark(
                    config=config,
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    num_iterations=args.prefill_iterations,
                )
                if prefill_baseline is None:
                    prefill_baseline = result['tok_s']
                    scaling = 1.0
                else:
                    scaling = result['tok_s'] / prefill_baseline

                print(f"  Batch {batch_size:2d}: {result['tok_s']:10.1f} tok/s  {result['ms_per_iter']:6.3f} ms/iter  {scaling:5.2f}x scaling")

                # Update results with prefill data
                for r in all_results:
                    if r.config_name == config_name and r.batch_size == batch_size:
                        r.prefill_tok_s = result['tok_s']
                        r.prefill_ms = result['ms_per_iter']
            except Exception as e:
                print(f"  Batch {batch_size:2d}: ERROR - {e}")

    # Summary table
    print("\n")
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Config':<12} {'Batch':<6} {'Decode tok/s':<15} {'Decode ms':<12} {'Prefill tok/s':<15} {'Prefill ms':<12}")
    print("-" * 100)

    for r in all_results:
        print(f"{r.config_name:<12} {r.batch_size:<6} {r.decode_tok_s:<15.1f} {r.decode_ms:<12.3f} {r.prefill_tok_s:<15.1f} {r.prefill_ms:<12.3f}")

    # Final summary
    print("\n")
    print("=" * 100)
    print("KEY METRICS")
    print("=" * 100)

    for config_name in args.configs:
        config_results = [r for r in all_results if r.config_name == config_name]
        if not config_results:
            continue

        decode_b1 = next((r for r in config_results if r.batch_size == 1), None)
        decode_b16 = next((r for r in config_results if r.batch_size == 16), None)

        if decode_b1 and decode_b16:
            decode_scaling = decode_b16.decode_tok_s / decode_b1.decode_tok_s

            print(f"\n{config_name.upper()}:")
            print(f"  Decode scaling (batch 1→16): {decode_scaling:.2f}x")
            print(f"  Peak decode throughput: {decode_b16.decode_tok_s:.1f} tok/s")

            if decode_b1.prefill_tok_s > 0 and decode_b16.prefill_tok_s > 0:
                prefill_scaling = decode_b16.prefill_tok_s / decode_b1.prefill_tok_s
                print(f"  Prefill scaling (batch 1→16): {prefill_scaling:.2f}x")
                print(f"  Peak prefill throughput: {decode_b16.prefill_tok_s:.1f} tok/s")

    print("\n" + "=" * 100)
    print("Benchmark complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
