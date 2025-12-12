# SPDX-License-Identifier: Apache-2.0
"""Integration test for vllm-apple plugin with vLLM LLM API."""

import os
import sys
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply MPS patches before importing vLLM
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

from vllm_apple.patch_mps_empty import apply_mps_empty_tensor_patches
apply_mps_empty_tensor_patches()

# Model for testing and benchmarks (MoE model with 3B active params)
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
BENCHMARK_MODEL = "Qwen/Qwen3-30B-A3B"


def test_llm_generate(model_name: str = DEFAULT_MODEL):
    """Test basic text generation with vLLM using Apple plugin."""
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("vLLM Apple Plugin Integration Test")
    print("=" * 60)

    print(f"\nLoading model: {model_name}")
    start = time.perf_counter()

    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,  # Disable torch.compile for testing
    )

    load_time = time.perf_counter() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Test generation
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    print("\nGenerating text...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.perf_counter() - start

    print(f"Generation completed in {gen_time:.2f}s")
    print()

    total_tokens = 0
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        total_tokens += num_tokens

        print(f"Prompt {i+1}: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print(f"Tokens: {num_tokens}")
        print()

    tps = total_tokens / gen_time
    print("=" * 60)
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {gen_time:.2f}s")
    print(f"TPS: {tps:.1f}")
    print("=" * 60)

    # Assert reasonable performance
    assert len(outputs) == len(prompts), "Should generate output for all prompts"
    assert all(len(o.outputs[0].token_ids) > 0 for o in outputs), "Should generate tokens"

    print("\n[OK] Integration test PASSED")
    return tps


def test_tps_benchmark(model_name: str = BENCHMARK_MODEL):
    """Benchmark TPS with single request, multiple tokens."""
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("TPS Benchmark")
    print("=" * 60)

    print(f"\nLoading model: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
    )

    prompts = ["What is the capital of France?"]

    # Warmup
    print("\nWarmup...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    _ = llm.generate(prompts, sampling_params)

    # Benchmark with different token counts
    for num_tokens in [32, 64, 128]:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=num_tokens,
            min_tokens=num_tokens,
        )

        iterations = 3
        total_tokens = 0
        start = time.perf_counter()

        for _ in range(iterations):
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                total_tokens += len(output.outputs[0].token_ids)

        elapsed = time.perf_counter() - start
        tps = total_tokens / elapsed

        status = "PASS" if tps > 7 else "WARN"  # Current baseline ~7.8 TPS
        print(f"  {num_tokens} tokens: {tps:.1f} TPS [{status}]")

    print()
    print("Target: >50 TPS")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Apple Plugin Integration Test")
    parser.add_argument(
        "mode",
        nargs="?",
        default="test",
        choices=["test", "benchmark"],
        help="test=quick test with small model, benchmark=TPS benchmark with large model"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=f"Model to use (default: {DEFAULT_MODEL} for test, {BENCHMARK_MODEL} for benchmark)"
    )

    args = parser.parse_args()

    if args.mode == "benchmark":
        model = args.model or BENCHMARK_MODEL
        test_tps_benchmark(model)
    else:
        model = args.model or DEFAULT_MODEL
        test_llm_generate(model)
