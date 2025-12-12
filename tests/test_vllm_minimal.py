# SPDX-License-Identifier: Apache-2.0
"""Minimal vLLM test to diagnose memory issues.

Tests vLLM with the smallest possible model.
"""

import os
import sys
import gc
import time

# Apply MPS patches before importing anything else
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "2"  # 2GB KV cache for testing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import psutil


def print_memory(label=""):
    """Print current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()

    print(f"  [{label}] Process RSS: {mem_info.rss / (1024**3):.2f} GiB, System used: {vm.used / (1024**3):.1f} GiB")


def test_vllm_gpt2():
    """Test vLLM with GPT-2 (smallest reasonable model)."""
    print("\n" + "="*60)
    print("  Test: vLLM with GPT-2")
    print("="*60)

    print_memory("Start")

    print("\n1. Importing vLLM...")
    from vllm import LLM, SamplingParams

    print_memory("After import")

    print("\n2. Creating LLM instance with GPT-2...")
    print("   Model: gpt2 (~500MB)")
    print("   max_model_len: 512")
    print("   enforce_eager: True")

    try:
        llm = LLM(
            model="gpt2",
            dtype="float16",
            max_model_len=512,  # Very short context
            trust_remote_code=False,
            enforce_eager=True,
            gpu_memory_utilization=0.5,  # Use only 50% of GPU memory
        )
        print_memory("After LLM init")

    except Exception as e:
        print(f"\n   [FAIL] LLM creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. Testing generation...")
    prompts = ["Hello, world!"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    try:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        for output in outputs:
            print(f"   Prompt: {output.prompt!r}")
            print(f"   Output: {output.outputs[0].text!r}")
            print(f"   Tokens: {len(output.outputs[0].token_ids)}")

        print(f"   Time: {elapsed:.2f}s")
        print_memory("After generation")

    except Exception as e:
        print(f"\n   [FAIL] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Cleanup...")
    del llm
    gc.collect()
    torch.mps.empty_cache()

    print_memory("After cleanup")

    print("\n   [PASS] GPT-2 test completed")
    return True


def test_vllm_tiny_llama():
    """Test vLLM with TinyLlama (1.1B but fits in memory)."""
    print("\n" + "="*60)
    print("  Test: vLLM with TinyLlama-1.1B")
    print("="*60)

    print_memory("Start")

    print("\n1. Importing vLLM...")
    from vllm import LLM, SamplingParams

    print_memory("After import")

    print("\n2. Creating LLM instance with TinyLlama...")
    print("   Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2GB)")
    print("   max_model_len: 1024")
    print("   enforce_eager: True")

    try:
        llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16",
            max_model_len=1024,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
        )
        print_memory("After LLM init")

    except Exception as e:
        print(f"\n   [FAIL] LLM creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. Testing generation...")
    prompts = ["<|user|>\nHello!\n<|assistant|>\n"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    try:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        for output in outputs:
            generated = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            tps = tokens / elapsed if elapsed > 0 else 0

            print(f"   Output: {generated[:60]!r}...")
            print(f"   Tokens: {tokens}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   TPS: {tps:.1f}")

        print_memory("After generation")

    except Exception as e:
        print(f"\n   [FAIL] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Cleanup...")
    del llm
    gc.collect()
    torch.mps.empty_cache()

    print_memory("After cleanup")

    print("\n   [PASS] TinyLlama test completed")
    return True


def main():
    """Run minimal vLLM tests."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2", "tinyllama", "all"], default="gpt2")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  vLLM Minimal Tests")
    print("="*60)

    results = []

    if args.model in ["gpt2", "all"]:
        try:
            result = test_vllm_gpt2()
            results.append(("GPT-2", result))
        except Exception as e:
            print(f"\n  [ERROR] Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(("GPT-2", False))

    if args.model in ["tinyllama", "all"]:
        try:
            result = test_vllm_tiny_llama()
            results.append(("TinyLlama", result))
        except Exception as e:
            print(f"\n  [ERROR] Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(("TinyLlama", False))

    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
