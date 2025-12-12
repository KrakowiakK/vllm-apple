#!/usr/bin/env python3
"""Simple inference test for vllm-apple.

Tests the full vLLM pipeline with Apple platform.
"""

import os
import sys
import time

# Set environment before any imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["VLLM_PLUGINS"] = "apple"

import torch


def test_platform_detection():
    """Test that Apple platform is correctly detected."""
    print("\n" + "=" * 60)
    print("TEST 1: Platform Detection")
    print("=" * 60)

    from vllm.platforms import current_platform

    platform_name = type(current_platform).__name__
    print(f"Platform: {platform_name}")
    print(f"Device name: {current_platform.device_name}")
    print(f"Device type: {current_platform.device_type}")

    assert platform_name == "ApplePlatform", f"Expected ApplePlatform, got {platform_name}"
    assert current_platform.device_name == "mps"
    print("PASS: Apple platform detected correctly")
    return True


def test_moe_operation():
    """Test MoE operation in isolation."""
    print("\n" + "=" * 60)
    print("TEST 2: MoE Operation")
    print("=" * 60)

    sys.path.insert(0, "/Users/pimpc181/Desktop/BATCH/vllm/vllm-apple")
    from vllm_apple.ops.apple_fused_moe import AppleMoEOp

    device = "mps"
    dtype = torch.float16

    # Small test dimensions
    num_experts = 8
    hidden_size = 256
    intermediate_size = 512
    top_k = 2
    num_tokens = 4

    # Create test data
    w13 = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=dtype, device=device) * 0.01
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device) * 0.01
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)

    # Initialize and run
    apple_op = AppleMoEOp(num_experts, hidden_size, intermediate_size)
    apple_op.set_weights(w13, w2)

    output = apple_op(x, router_logits, top_k)
    torch.mps.synchronize()

    # Basic checks
    assert output.shape == (num_tokens, hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Benchmark
    num_iterations = 100
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = apple_op(x, router_logits, top_k)
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    tps = (num_tokens * num_iterations) / elapsed
    print(f"MoE throughput: {tps:.1f} tokens/second")
    print("PASS: MoE operation works correctly")
    return True


def test_vllm_import():
    """Test vLLM import with Apple platform."""
    print("\n" + "=" * 60)
    print("TEST 3: vLLM Import")
    print("=" * 60)

    from vllm import LLM, SamplingParams
    print("vLLM imported successfully")
    print(f"LLM class: {LLM}")
    print(f"SamplingParams: {SamplingParams}")
    print("PASS: vLLM imports work")
    return True


def test_small_model_inference():
    """Test inference with a small model."""
    print("\n" + "=" * 60)
    print("TEST 4: Small Model Inference")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams

        # Use a very small model for testing
        model_name = "facebook/opt-125m"

        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            dtype="float16",
            max_model_len=256,
            enforce_eager=True,
            trust_remote_code=True,
        )

        print("Model loaded successfully!")

        # Test generation
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(max_tokens=16, temperature=0.8)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        for output in outputs:
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            tps = num_tokens / elapsed
            print(f"Generated: {generated_text}")
            print(f"Tokens: {num_tokens}, Time: {elapsed:.2f}s, TPS: {tps:.1f}")

        print("PASS: Small model inference works")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("vllm-apple Integration Test Suite")
    print("=" * 60)
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")

    results = {}

    # Run tests
    results["platform"] = test_platform_detection()
    results["moe"] = test_moe_operation()
    results["import"] = test_vllm_import()
    # results["inference"] = test_small_model_inference()  # Uncomment when ready

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
