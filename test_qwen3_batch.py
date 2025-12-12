#!/usr/bin/env python3
"""End-to-end test for vLLM Apple with Qwen3-30B-A3B.

Tests prefill + decode for batch sizes 1, 2, 4, 8, 16.
"""

import os
import time

# Enable Metal backend and serialization fallback
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_METAL_FUSED_KV"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "32"  # 32GB for KV cache

import torch
from vllm import LLM, SamplingParams


def test_batch_inference(batch_sizes=[1, 2, 4, 8, 16]):
    """Test inference across different batch sizes."""

    model_name = "Qwen/Qwen3-30B-A3B"

    print("=" * 70)
    print("vLLM Apple Silicon - Qwen3-30B-A3B Batch Test")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print("=" * 70)

    # Initialize LLM
    print("\nLoading model...")
    start = time.time()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Test prompts
    base_prompts = [
        "What is 2+2?",
        "Name the capital of France.",
        "Write a haiku about coding.",
        "Explain AI in one sentence.",
        "What color is the sky?",
        "Count from 1 to 5.",
        "Say hello in Spanish.",
        "What is Python?",
        "Name a prime number.",
        "What is the sun?",
        "Define gravity briefly.",
        "What is water made of?",
        "Name a continent.",
        "What is 10 times 10?",
        "Say goodbye.",
        "What day comes after Monday?",
    ]

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )

    results = []

    for batch_size in batch_sizes:
        prompts = base_prompts[:batch_size]

        print(f"\n{'='*70}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"{'='*70}")

        # Warmup for first batch
        if batch_size == batch_sizes[0]:
            print("Warmup run...")
            _ = llm.generate(prompts[:1], sampling_params)

        # Timed run
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        elapsed = time.time() - start

        # Calculate metrics
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)

        tok_per_sec = total_output_tokens / elapsed

        print(f"\nMetrics:")
        print(f"  Input tokens:  {total_input_tokens}")
        print(f"  Output tokens: {total_output_tokens}")
        print(f"  Time:          {elapsed:.3f}s")
        print(f"  Throughput:    {tok_per_sec:.1f} tok/s")

        print(f"\nOutputs:")
        for i, output in enumerate(outputs):
            prompt = output.prompt[:40] + "..." if len(output.prompt) > 40 else output.prompt
            response = output.outputs[0].text.strip()[:60]
            response = response.replace('\n', ' ')
            print(f"  [{i+1}] Q: {prompt}")
            print(f"      A: {response}")

        results.append({
            'batch': batch_size,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'time': elapsed,
            'tok_s': tok_per_sec,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch':<8} {'In Tok':<10} {'Out Tok':<10} {'Time':<10} {'tok/s':<10}")
    print("-" * 48)

    baseline_tps = results[0]['tok_s']
    for r in results:
        scaling = r['tok_s'] / baseline_tps
        print(f"{r['batch']:<8} {r['input_tokens']:<10} {r['output_tokens']:<10} "
              f"{r['time']:<10.3f} {r['tok_s']:<10.1f} ({scaling:.2f}x)")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    test_batch_inference()
