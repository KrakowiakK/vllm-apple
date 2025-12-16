#!/usr/bin/env python3
"""Debug test for engine output comparison."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

import sys
# Check for engine mode arg
use_engine = len(sys.argv) > 1 and sys.argv[1] == '--engine'

if use_engine:
    os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
    os.environ['VLLM_GEMM_BACKEND'] = 'mps'  # Use MPS to rule out Metal GEMM issues
    print("Testing WITH engine (MPS backends)...")
else:
    os.environ['VLLM_APPLE_USE_ENGINE'] = '0'
    print("Testing WITHOUT engine...")

def main():
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='mistralai/Devstral-Small-2505',
        max_model_len=128,
        dtype='float16',
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Simple test prompt
    prompts = ["def add(a, b):"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    outputs = llm.generate(prompts, sampling_params)
    text = outputs[0].outputs[0].text
    print(f"Output: {text!r}")

    # Check if it looks reasonable
    if "return" in text.lower() or "+" in text or "a + b" in text:
        print("✓ Output looks like valid code continuation")
    else:
        print("✗ Output might be garbage")

if __name__ == "__main__":
    main()
