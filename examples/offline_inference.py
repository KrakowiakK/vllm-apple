#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for vLLM Apple Silicon.

This example demonstrates how to run offline inference using vLLM
with the Apple Silicon Metal backend.

Usage:
    python examples/offline_inference.py

Requirements:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - vllm-apple plugin installed
    - Model downloaded (default: Qwen/Qwen2.5-0.5B-Instruct)
"""

import os

# Enable Metal attention backend
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_METAL_FUSED_KV"] = "1"

from vllm import LLM, SamplingParams


def main():
    # Model configuration
    # Use a small model for demonstration
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Initialize vLLM with Apple Silicon settings
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",  # Use FP16 for Metal
        device="mps",     # Use MPS device
    )

    # Define prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    # Generate responses
    print("=" * 60)
    print("vLLM Apple Silicon - Offline Inference")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Response: {generated_text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
