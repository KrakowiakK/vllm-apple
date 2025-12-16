#!/usr/bin/env python3
"""Debug logits from engine vs non-engine."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

import sys
use_engine = len(sys.argv) > 1 and sys.argv[1] == '--engine'

if use_engine:
    os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
    os.environ['VLLM_GEMM_BACKEND'] = 'mps'
    print("Testing WITH engine...")
else:
    os.environ['VLLM_APPLE_USE_ENGINE'] = '0'
    print("Testing WITHOUT engine...")

import torch
import numpy as np

def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='mistralai/Devstral-Small-2505',
        max_model_len=128,
        dtype='float16',
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Single prompt
    prompts = ["def add(a, b):"]
    # Use temperature=0 and max_tokens=1 to get deterministic output
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    # We can't easily get logits from vLLM's API
    # But we can check what token is selected
    outputs = llm.generate(prompts, sampling_params)

    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids
    print(f"Generated text: {text!r}")
    print(f"Generated token IDs: {token_ids}")

    # Decode individual tokens to understand what we got
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('mistralai/Devstral-Small-2505', trust_remote_code=True)
    for tid in token_ids:
        decoded = tok.decode([tid])
        print(f"  Token {tid}: {decoded!r}")

    # Show expected continuation
    print(f"\nExpected: Token for ' return' or similar code")

if __name__ == "__main__":
    main()
