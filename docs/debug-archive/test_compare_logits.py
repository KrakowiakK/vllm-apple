#!/usr/bin/env python3
"""Compare logits from engine vs non-engine."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'  # Non-engine

import torch
import numpy as np

def main():
    from vllm import LLM, SamplingParams

    print("Testing WITHOUT engine (reference)...")

    llm = LLM(
        model='mistralai/Devstral-Small-2505',
        max_model_len=128,
        dtype='float16',
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Generate multiple tokens to see what the model outputs
    prompts = ["def add(a, b):"]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    outputs = llm.generate(prompts, sampling_params)

    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids
    print(f"Generated text: {text!r}")
    print(f"Generated token IDs: {token_ids}")

    # Decode individual tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('mistralai/Devstral-Small-2505', trust_remote_code=True)
    for tid in token_ids:
        decoded = tok.decode([tid])
        print(f"  Token {tid}: {decoded!r}")

if __name__ == "__main__":
    main()
