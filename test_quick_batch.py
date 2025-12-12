#!/usr/bin/env python3
"""Quick batch test without chunked prefill."""

import os
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_METAL_FUSED_KV"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "32"

import time
import torch
from vllm import LLM, SamplingParams


def main():
    print("Loading Qwen3-30B-A3B (no chunked prefill)...")
    start = time.time()

    llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
        enforce_eager=True,
        enable_chunked_prefill=False,  # Disable chunked prefill
    )

    print(f"Model loaded in {time.time()-start:.1f}s")

    # Test prompts
    prompts = [
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

    params = SamplingParams(temperature=0.0, max_tokens=32)

    # Warmup
    print("\nWarmup...")
    _ = llm.generate(prompts[:1], params)

    results = []
    for batch_size in [1, 2, 4, 8, 16]:
        batch_prompts = prompts[:batch_size]

        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        start = time.time()
        outputs = llm.generate(batch_prompts, params)
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        elapsed = time.time() - start

        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tok_per_sec = total_output_tokens / elapsed

        print(f"Batch {batch_size:2d}: {total_output_tokens:4d} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s")
        results.append((batch_size, tok_per_sec))

    print("\n=== Summary ===")
    baseline = results[0][1]
    for batch, tps in results:
        print(f"Batch {batch:2d}: {tps:5.1f} tok/s ({tps/baseline:.2f}x)")


if __name__ == "__main__":
    main()
