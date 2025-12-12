#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scaling and throughput tests for vLLM Apple.

Tests prefill + decode throughput across various batch sizes.
Consolidates: test_scaling.py, test_adaptive_moe.py

Usage:
    # Full scaling test (1-256)
    python tests/e2e/test_scaling.py

    # MoE-focused test (16-256)
    python tests/e2e/test_scaling.py --moe

    # Custom batch sizes
    python tests/e2e/test_scaling.py --batch-sizes 1 4 16 64
"""

import argparse
import os
import time

# Enable Metal backend
os.environ.setdefault("VLLM_METAL_ATTENTION", "1")
os.environ.setdefault("VLLM_METAL_FUSED_KV", "1")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "32")

import torch


# Real conversation prompt for meaningful benchmarking
REAL_PROMPT = (
    "You are a helpful AI assistant. Please provide a detailed and informative response.\n\n"
    "User: I'm planning to learn programming. I have no prior experience but I'm very motivated.\n"
    "Can you give me a comprehensive guide on how to start? Include:\n"
    "1. Which programming language should I learn first and why?\n"
    "2. What resources (books, websites, courses) do you recommend?\n"
    "3. What projects should I build as a beginner?\n"
    "4. How long will it take to become proficient?\n"
    "5. What are common mistakes beginners make and how to avoid them?\n\n"
    "Please be thorough in your response."
)


def run_scaling_test(
    batch_sizes: list = None,
    max_tokens: int = 128,
    show_moe_analysis: bool = False,
    model_name: str = "Qwen/Qwen3-30B-A3B",
):
    """Run scaling test with specified batch sizes.

    Args:
        batch_sizes: List of batch sizes to test
        max_tokens: Maximum tokens to generate
        show_moe_analysis: Show MoE-specific analysis
        model_name: Model to use for testing

    Returns:
        List of result dicts with metrics
    """
    from vllm import LLM, SamplingParams

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print("=" * 80)
    print("VLLM APPLE SCALING TEST")
    if show_moe_analysis:
        print("(with MoE Analysis)")
    print("=" * 80)

    # Load model
    print(f"\nLoading {model_name}...")
    start = time.time()

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    print(f"Model loaded in {time.time()-start:.1f}s")

    # Sampling params
    params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    # Warmup
    print("\nWarmup run...")
    _ = llm.generate([REAL_PROMPT], params)
    print("Warmup complete.\n")

    results = []

    for batch_size in batch_sizes:
        prompts = [REAL_PROMPT] * batch_size

        print(f"\n{'='*80}")
        print(f"BATCH SIZE: {batch_size}")
        if show_moe_analysis:
            print(f"Expert tokens: {batch_size} * 8 (top_k) = {batch_size * 8}")
        print(f"{'='*80}")

        # Run inference
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        start = time.time()
        outputs = llm.generate(prompts, params)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        prefill_tok_s = total_input_tokens / elapsed
        decode_tok_s = total_output_tokens / elapsed
        total_tok_s = (total_input_tokens + total_output_tokens) / elapsed

        avg_input_per_req = total_input_tokens / batch_size
        avg_output_per_req = total_output_tokens / batch_size

        print(f"\nMetrics:")
        print(f"  Requests:       {batch_size}")
        print(f"  Input tokens:   {total_input_tokens} (avg {avg_input_per_req:.1f}/req)")
        print(f"  Output tokens:  {total_output_tokens} (avg {avg_output_per_req:.1f}/req)")
        print(f"  Total time:     {elapsed:.2f}s")
        print(f"  Prefill:        {prefill_tok_s:.1f} tok/s")
        print(f"  Decode:         {decode_tok_s:.1f} tok/s")
        print(f"  Total:          {total_tok_s:.1f} tok/s")

        # Show sample output
        print(f"\nSample output (request 1):")
        response = outputs[0].outputs[0].text.strip()[:200].replace('\n', ' ')
        print(f"  {response}...")

        result = {
            'batch': batch_size,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'time': elapsed,
            'prefill_tok_s': prefill_tok_s,
            'decode_tok_s': decode_tok_s,
            'total_tok_s': total_tok_s,
        }
        if show_moe_analysis:
            result['expert_tokens'] = batch_size * 8

        results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SCALING SUMMARY")
    print(f"{'='*80}")

    if show_moe_analysis:
        print(f"{'Batch':<8} {'Exp.Tok':<10} {'In Tok':<10} {'Out Tok':<10} {'Time(s)':<10} {'Prefill':<12} {'Decode':<12} {'Total':<12}")
    else:
        print(f"{'Batch':<8} {'In Tok':<10} {'Out Tok':<10} {'Time(s)':<10} {'Prefill':<12} {'Decode':<12} {'Total':<12} {'Scaling':<10}")
    print("-" * 94)

    baseline_decode = results[0]['decode_tok_s']
    for r in results:
        scaling = r['decode_tok_s'] / baseline_decode
        if show_moe_analysis:
            print(f"{r['batch']:<8} {r['expert_tokens']:<10} {r['input_tokens']:<10} {r['output_tokens']:<10} "
                  f"{r['time']:<10.2f} {r['prefill_tok_s']:<12.1f} {r['decode_tok_s']:<12.1f} "
                  f"{r['total_tok_s']:<12.1f} ({scaling:.2f}x)")
        else:
            print(f"{r['batch']:<8} {r['input_tokens']:<10} {r['output_tokens']:<10} "
                  f"{r['time']:<10.2f} {r['prefill_tok_s']:<12.1f} {r['decode_tok_s']:<12.1f} "
                  f"{r['total_tok_s']:<12.1f} {scaling:<10.2f}x")

    # Efficiency analysis
    print(f"\n{'='*80}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*80}")

    print("\nDecode throughput scaling:")
    for r in results:
        expected_linear = baseline_decode * r['batch']
        actual = r['decode_tok_s']
        efficiency = (actual / expected_linear) * 100 if expected_linear > 0 else 0
        bar_len = min(int(efficiency / 5), 20)
        bar = "█" * bar_len
        print(f"  Batch {r['batch']:>3}: {actual:>6.1f} tok/s  (efficiency: {efficiency:>5.1f}%) {bar}")

    # MoE-specific analysis
    if show_moe_analysis:
        print(f"\n{'='*80}")
        print("MOE ADAPTIVE LOGIC ANALYSIS")
        print(f"{'='*80}")
        print("\nExpected behavior with adaptive logic:")
        print("  Batch 16:  128 expert tokens <= 256  -> COMBINED (small_batch)")
        print("  Batch 32:  256 expert tokens <= 256  -> COMBINED (small_batch)")
        print("  Batch 64:  512 expert tokens <= 4096 -> COMBINED if unique <= 16 (medium_batch)")
        print("  Batch 128: 1024 expert tokens <= 4096 -> COMBINED if unique <= 16 (medium_batch)")
        print("  Batch 256: 2048 expert tokens <= 4096 -> COMBINED if unique <= 16 (medium_batch)")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Scaling test for vLLM Apple")
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Run MoE-focused test with batch sizes 16-256"
    )
    parser.add_argument(
        "--batch-sizes", "-b",
        type=int,
        nargs="+",
        default=None,
        help="Custom batch sizes to test"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=128,
        help="Max tokens to generate (default: 128)"
    )

    args = parser.parse_args()

    if args.batch_sizes:
        batch_sizes = args.batch_sizes
        show_moe = args.moe
    elif args.moe:
        batch_sizes = [16, 32, 64, 128, 256]
        show_moe = True
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        show_moe = False

    run_scaling_test(
        batch_sizes=batch_sizes,
        max_tokens=args.max_tokens,
        show_moe_analysis=show_moe,
    )


if __name__ == "__main__":
    main()
