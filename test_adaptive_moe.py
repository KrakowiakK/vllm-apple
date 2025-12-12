#!/usr/bin/env python3
"""
Adaptive MoE Combine Logic Test
================================

Test batch sizes: 16, 32, 64, 128, 256
Report: prefill tok/s, decode tok/s, total tok/s, forward time, MoE entropy
"""

import os
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_METAL_FUSED_KV"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "32"

import time
import torch
from vllm import LLM, SamplingParams


def main():
    print("=" * 80)
    print("ADAPTIVE MoE COMBINE LOGIC TEST")
    print("=" * 80)

    # Load model
    print("\nLoading Qwen3-30B-A3B...")
    start = time.time()

    llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    print(f"Model loaded in {time.time()-start:.1f}s")

    # Real conversation prompt
    real_prompt = (
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

    # Sampling params - generate 128 tokens for good measurement
    params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
    )

    # Warmup
    print("\nWarmup run...")
    _ = llm.generate([real_prompt], params)
    print("Warmup complete.\n")

    # Test batch sizes
    batch_sizes = [16, 32, 64, 128, 256]
    results = []

    for batch_size in batch_sizes:
        # Create batch of identical prompts
        prompts = [real_prompt] * batch_size

        print(f"\n{'='*80}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"Expert tokens: {batch_size} * 8 (top_k) = {batch_size * 8}")
        print(f"{'='*80}")

        # Run inference
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        start = time.time()
        outputs = llm.generate(prompts, params)
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        elapsed = time.time() - start

        # Calculate metrics
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        prefill_tok_s = total_input_tokens / elapsed
        decode_tok_s = total_output_tokens / elapsed
        total_tok_s = (total_input_tokens + total_output_tokens) / elapsed

        # Time per token estimates
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

        results.append({
            'batch': batch_size,
            'expert_tokens': batch_size * 8,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'time': elapsed,
            'prefill_tok_s': prefill_tok_s,
            'decode_tok_s': decode_tok_s,
            'total_tok_s': total_tok_s,
        })

    # Summary table
    print(f"\n{'='*80}")
    print("ADAPTIVE MoE SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Batch':<8} {'Exp.Tok':<10} {'In Tok':<10} {'Out Tok':<10} {'Time(s)':<10} {'Prefill':<12} {'Decode':<12} {'Total':<12}")
    print("-" * 94)

    baseline_decode = results[0]['decode_tok_s']
    for r in results:
        scaling = r['decode_tok_s'] / baseline_decode
        print(f"{r['batch']:<8} {r['expert_tokens']:<10} {r['input_tokens']:<10} {r['output_tokens']:<10} "
              f"{r['time']:<10.2f} {r['prefill_tok_s']:<12.1f} {r['decode_tok_s']:<12.1f} "
              f"{r['total_tok_s']:<12.1f} ({scaling:.2f}x)")

    # Adaptive logic analysis
    print(f"\n{'='*80}")
    print("ADAPTIVE LOGIC ANALYSIS")
    print(f"{'='*80}")
    print("\nExpected behavior with adaptive logic:")
    print("  Batch 16:  128 expert tokens <= 256  → COMBINED (small_batch)")
    print("  Batch 32:  256 expert tokens <= 256  → COMBINED (small_batch)")
    print("  Batch 64:  512 expert tokens <= 4096 → COMBINED if unique <= 16 (medium_batch)")
    print("  Batch 128: 1024 expert tokens <= 4096 → COMBINED if unique <= 16 (medium_batch)")
    print("  Batch 256: 2048 expert tokens <= 4096 → COMBINED if unique <= 16 (medium_batch)")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    main()
