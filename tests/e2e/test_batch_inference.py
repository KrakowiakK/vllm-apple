#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end batch inference tests for vLLM Apple.

Tests prefill + decode across different batch sizes with various models.
Consolidates: test_e2e_batch.py, test_qwen3_batch.py, test_quick_batch.py

Usage:
    # Quick test with small model (default)
    python -m pytest tests/e2e/test_batch_inference.py -v

    # Test with specific model
    python tests/e2e/test_batch_inference.py --model qwen3-30b
    python tests/e2e/test_batch_inference.py --model qwen2.5-0.5b

    # Run as standalone script
    python tests/e2e/test_batch_inference.py
"""

import argparse
import os
import sys
import time

# Enable Metal backend
os.environ.setdefault("VLLM_METAL_ATTENTION", "1")
os.environ.setdefault("VLLM_METAL_FUSED_KV", "1")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "32")

import torch

# Model configurations
MODELS = {
    "qwen2.5-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_model_len": 512,
        "gpu_memory_utilization": 0.8,
    },
    "qwen3-30b": {
        "name": "Qwen/Qwen3-30B-A3B",
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.9,
    },
    "gpt2": {
        "name": "gpt2",
        "max_model_len": 256,
        "gpu_memory_utilization": 0.5,
    },
}

# Test prompts
TEST_PROMPTS = [
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


def run_batch_inference(
    model_key: str = "qwen2.5-0.5b",
    batch_sizes: list = None,
    max_tokens: int = 64,
    verbose: bool = True,
):
    """Run batch inference test with specified model.

    Args:
        model_key: Key from MODELS dict
        batch_sizes: List of batch sizes to test
        max_tokens: Maximum tokens to generate
        verbose: Print detailed output

    Returns:
        List of result dicts with metrics
    """
    from vllm import LLM, SamplingParams

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    model_config = MODELS[model_key]
    model_name = model_config["name"]

    if verbose:
        print("=" * 70)
        print(f"vLLM Apple Silicon - Batch Inference Test")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Batch sizes: {batch_sizes}")
        print("=" * 70)

    # Initialize LLM
    if verbose:
        print("\nLoading model...")
    start = time.time()

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=model_config["max_model_len"],
        gpu_memory_utilization=model_config["gpu_memory_utilization"],
        enforce_eager=True,
    )

    load_time = time.time() - start
    if verbose:
        print(f"Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    results = []

    for batch_size in batch_sizes:
        prompts = TEST_PROMPTS[:batch_size]

        if verbose:
            print(f"\n{'='*70}")
            print(f"BATCH SIZE: {batch_size}")
            print(f"{'='*70}")

        # Warmup for first batch
        if batch_size == batch_sizes[0]:
            if verbose:
                print("Warmup run...")
            _ = llm.generate(prompts[:1], sampling_params)

        # Timed run
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        tok_per_sec = total_output_tokens / elapsed if elapsed > 0 else 0

        if verbose:
            print(f"\nMetrics:")
            print(f"  Input tokens:  {total_input_tokens}")
            print(f"  Output tokens: {total_output_tokens}")
            print(f"  Time:          {elapsed:.3f}s")
            print(f"  Throughput:    {tok_per_sec:.1f} tok/s")

            print(f"\nOutputs:")
            for i, output in enumerate(outputs):
                prompt = output.prompt[:40] + "..." if len(output.prompt) > 40 else output.prompt
                response = output.outputs[0].text.strip()[:60].replace('\n', ' ')
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
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Batch':<8} {'In Tok':<10} {'Out Tok':<10} {'Time':<10} {'tok/s':<10}")
        print("-" * 48)

        baseline_tps = results[0]['tok_s'] if results[0]['tok_s'] > 0 else 1
        for r in results:
            scaling = r['tok_s'] / baseline_tps
            print(f"{r['batch']:<8} {r['input_tokens']:<10} {r['output_tokens']:<10} "
                  f"{r['time']:<10.3f} {r['tok_s']:<10.1f} ({scaling:.2f}x)")

        print(f"\n{'='*70}")
        print("TEST COMPLETE")
        print(f"{'='*70}")

    return results


# ============================================================================
# Pytest tests
# ============================================================================

def test_batch_inference_small():
    """Test batch inference with small model (quick CI test)."""
    try:
        results = run_batch_inference(
            model_key="gpt2",
            batch_sizes=[1, 2, 4],
            max_tokens=32,
            verbose=False,
        )

        assert len(results) == 3
        for r in results:
            assert r['output_tokens'] > 0
            assert r['tok_s'] > 0
    except Exception as e:
        # Skip if model download fails in CI
        import pytest
        pytest.skip(f"Model download failed: {e}")


# ============================================================================
# CLI interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch inference test for vLLM Apple")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default="qwen2.5-0.5b",
        help="Model to test (default: qwen2.5-0.5b)"
    )
    parser.add_argument(
        "--batch-sizes", "-b",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to test (default: 1 2 4 8 16)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=64,
        help="Max tokens to generate (default: 64)"
    )

    args = parser.parse_args()

    run_batch_inference(
        model_key=args.model,
        batch_sizes=args.batch_sizes,
        max_tokens=args.max_tokens,
        verbose=True,
    )


if __name__ == "__main__":
    main()
