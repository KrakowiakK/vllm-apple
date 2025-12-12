#!/usr/bin/env python3
"""End-to-end test for vllm-apple with Qwen3-30B-A3B.

This test measures actual inference throughput (tokens/second) using
the vllm-apple plugin on Apple Silicon MPS.
"""

import os
import sys
import time
from typing import Optional

# Set environment before any imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["VLLM_PLUGINS"] = "apple"

import torch


def test_moe_benchmark():
    """Test MoE performance in isolation."""
    print("\n" + "=" * 70)
    print("TEST 1: MoE Benchmark (isolated layer)")
    print("=" * 70)

    # Add path to vllm-apple
    sys.path.insert(0, "/Users/pimpc181/Desktop/BATCH/vllm/vllm-apple")

    from vllm_apple.ops.apple_fused_moe import AppleMoEOp

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16

    # Qwen3-30B-A3B dimensions
    num_experts = 128
    hidden_size = 2048
    intermediate_size = 8192
    top_k = 8
    num_tokens = 1

    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Experts: {num_experts}, Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    print(f"Top-K: {top_k}, Tokens: {num_tokens}")

    # Create weights
    w13 = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=dtype, device=device) * 0.01
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device) * 0.01

    # Create input
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)

    # Initialize Apple MoE Op
    apple_op = AppleMoEOp(num_experts, hidden_size, intermediate_size)
    apple_op.set_weights(w13, w2)

    # Warmup
    for _ in range(5):
        _ = apple_op(x, router_logits, top_k)
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    num_iterations = 50
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = apple_op(x, router_logits, top_k)
        if device == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    tps = (num_tokens * num_iterations) / elapsed
    print(f"\nMoE Throughput: {tps:.1f} tokens/second")

    if tps >= 50:
        print("✓ MoE layer meets target (≥50 tps)")
    else:
        print(f"✗ MoE layer below target ({tps:.1f} < 50 tps)")

    return tps


def test_full_model_inference():
    """Test full model inference with vLLM."""
    print("\n" + "=" * 70)
    print("TEST 2: Full Model Inference (Qwen3-30B-A3B)")
    print("=" * 70)

    try:
        from vllm import LLM, SamplingParams

        print("\nInitializing vLLM with Apple platform...")

        # Check available memory
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        total_mem_gb = int(result.stdout.strip()) / (1024**3)
        print(f"Total system memory: {total_mem_gb:.1f} GB")

        # Model path (local or HuggingFace)
        model_name = "Qwen/Qwen3-30B-A3B"

        # Check for local model
        local_paths = [
            "/Users/pimpc181/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B",
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"),
        ]

        for path in local_paths:
            if os.path.exists(path):
                print(f"Found local model at: {path}")
                break
        else:
            print(f"Model will be downloaded from HuggingFace: {model_name}")

        # Initialize LLM
        # Note: vllm-apple plugin should be auto-detected
        llm = LLM(
            model=model_name,
            dtype="float16",
            max_model_len=2048,
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs (not supported on MPS)
        )

        print("Model loaded successfully!")

        # Test prompts
        prompts = [
            "Explain quantum computing in simple terms:",
            "Write a Python function that calculates fibonacci numbers:",
            "What is the capital of France and why is it famous?",
        ]

        sampling_params = SamplingParams(
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
        )

        # Warmup
        print("\nWarming up...")
        _ = llm.generate(["Hello"], SamplingParams(max_tokens=8))
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # Benchmark
        print("\nRunning benchmark...")
        total_tokens = 0
        start = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params)

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        # Count tokens
        for output in outputs:
            total_tokens += len(output.outputs[0].token_ids)

        tps = total_tokens / elapsed

        print(f"\nResults:")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {tps:.1f} tokens/second")

        if tps >= 50:
            print("\n✓ TARGET ACHIEVED: {tps:.1f} tps ≥ 50 tps")
        else:
            print(f"\n✗ Below target: {tps:.1f} tps < 50 tps")
            print(f"   Gap: need {50/tps:.1f}x improvement")

        # Print sample outputs
        print("\n--- Sample Outputs ---")
        for i, output in enumerate(outputs):
            print(f"\nPrompt {i+1}: {prompts[i][:50]}...")
            print(f"Output: {output.outputs[0].text[:100]}...")

        return tps

    except ImportError as e:
        print(f"Error importing vLLM: {e}")
        return 0
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 0


def test_offline_benchmark():
    """Run vLLM's offline benchmark."""
    print("\n" + "=" * 70)
    print("TEST 3: Offline Benchmark")
    print("=" * 70)

    try:
        from vllm import LLM, SamplingParams

        model_name = "Qwen/Qwen3-30B-A3B"

        llm = LLM(
            model=model_name,
            dtype="float16",
            max_model_len=512,  # Smaller for quick test
            trust_remote_code=True,
            enforce_eager=True,
        )

        # Generate many short sequences
        prompts = ["Hello, my name is"] * 10
        sampling_params = SamplingParams(max_tokens=32, temperature=0.8)

        # Benchmark multiple batches
        total_tokens = 0
        total_time = 0

        for batch in range(3):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start

            batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            total_tokens += batch_tokens
            total_time += elapsed

            print(f"Batch {batch+1}: {batch_tokens} tokens in {elapsed:.2f}s = {batch_tokens/elapsed:.1f} tps")

        overall_tps = total_tokens / total_time
        print(f"\nOverall: {total_tokens} tokens in {total_time:.2f}s = {overall_tps:.1f} tps")

        return overall_tps

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print("=" * 70)
    print("vllm-apple End-to-End Test Suite")
    print("=" * 70)
    print(f"Platform: Apple Silicon MPS")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")

    results = {}

    # Test 1: MoE benchmark
    results["moe_tps"] = test_moe_benchmark()

    # Test 2: Full model inference (if resources available)
    try:
        results["full_model_tps"] = test_full_model_inference()
    except Exception as e:
        print(f"Full model test failed: {e}")
        results["full_model_tps"] = 0

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"MoE Layer: {results['moe_tps']:.1f} tps", "✓" if results['moe_tps'] >= 50 else "✗")
    if results.get("full_model_tps", 0) > 0:
        print(f"Full Model: {results['full_model_tps']:.1f} tps", "✓" if results['full_model_tps'] >= 50 else "✗")

    target_met = results.get("full_model_tps", results["moe_tps"]) >= 50
    print(f"\nTarget (50 tps): {'ACHIEVED' if target_met else 'NOT YET'}")

    return 0 if target_met else 1


if __name__ == "__main__":
    sys.exit(main())
