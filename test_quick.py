#!/usr/bin/env python3
"""Quick test for vLLM Apple with Qwen3-30B-A3B."""

import os
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_METAL_FUSED_KV"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "32"

import time
from vllm import LLM, SamplingParams


def main():
    print("Loading Qwen3-30B-A3B...")
    start = time.time()

    llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
        enforce_eager=True,
    )

    print(f"Model loaded in {time.time()-start:.1f}s")

    # Simple test
    prompts = ["What is 2+2?"]
    params = SamplingParams(temperature=0.0, max_tokens=32)

    print("\nGenerating...")
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s")
    print(f"Output: {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    main()
