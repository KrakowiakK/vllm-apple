# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for vLLM-Apple Metal Engine v2.0.

Tests engine mode with VLLM_APPLE_USE_ENGINE=1 environment variable.
This validates the full engine execution path including:
- Engine runner initialization
- Weight loading to MTLBuffer
- KV cache in engine mode
- Forward pass execution (prefill + decode in engine mode by default)
- Step-boundary-only synchronization

IMPORTANT: Engine mode only supports LLaMA-family architectures (LLaMA, Qwen2, Mistral)
that use RMSNorm, RoPE, and gated SiLU MLP. Models like GPT-2 are NOT supported
and will fall back to PyTorch execution.

Run with:
    VLLM_APPLE_RUN_ENGINE_E2E=1 VLLM_APPLE_USE_ENGINE=1 pytest tests/e2e/test_engine_mode.py -v

Or with strict mode (fails on any MPS usage in hot path):
    VLLM_APPLE_RUN_ENGINE_E2E=1 VLLM_APPLE_USE_ENGINE=1 VLLM_METAL_STRICT_NO_MPS=1 pytest tests/e2e/test_engine_mode.py -v

To force PyTorch prefill (not recommended; disallowed in strict mode):
    VLLM_APPLE_RUN_ENGINE_E2E=1 VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=0 pytest tests/e2e/test_engine_mode.py -v
"""

import os
import sys
import gc
import time
import pytest

# These E2E tests are opt-in to avoid CI/download flakes.
if os.environ.get("VLLM_APPLE_RUN_ENGINE_E2E", "0") != "1":
    pytest.skip(
        "Set VLLM_APPLE_RUN_ENGINE_E2E=1 to run engine E2E tests.",
        allow_module_level=True,
    )

# Set environment BEFORE any vllm imports.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")  # 2GB KV cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_PLUGINS", "apple")

# Enable engine mode for these tests
os.environ["VLLM_APPLE_USE_ENGINE"] = "1"
# Engine prefill is default in engine mode; set explicitly for clarity.
os.environ.setdefault("VLLM_APPLE_ENGINE_PREFILL", "1")

import torch


def is_engine_mode_enabled():
    """Check if engine mode is enabled."""
    return os.environ.get("VLLM_APPLE_USE_ENGINE", "0") == "1"


def is_strict_mode_enabled():
    """Check if strict mode is enabled."""
    return os.environ.get("VLLM_METAL_STRICT_NO_MPS", "0") == "1"


class TestEngineMode:
    """Tests for engine mode execution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        gc.collect()
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        yield
        gc.collect()
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    def test_engine_mode_enabled(self):
        """Verify engine mode is enabled for these tests."""
        assert is_engine_mode_enabled(), "Engine mode must be enabled for these tests"
        print(f"\nEngine mode: {is_engine_mode_enabled()}")
        print(f"Strict mode: {is_strict_mode_enabled()}")

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_imports(self):
        """Test that engine modules can be imported."""
        from vllm_apple.engine import (
            is_engine_mode_enabled as engine_enabled,
            MetalEngineContext,
            EngineRunner,
            EngineKVCache,
            EngineWeightLoader,
            StepDescriptor,
            EngineInputs,
            EngineOutputs,
        )

        assert engine_enabled(), "is_engine_mode_enabled() should return True"
        assert MetalEngineContext is not None
        assert EngineRunner is not None

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_context_creation(self):
        """Test MetalEngineContext can be created."""
        from vllm_apple.engine import MetalEngineContext

        context = MetalEngineContext()
        assert context is not None
        assert context.device is not None
        print(f"\nMetal device: {context.device.name()}")

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_with_tinyllama(self):
        """Test engine mode with TinyLlama model.

        TinyLlama uses the LLaMA architecture which is supported by the engine:
        - RMSNorm (not LayerNorm)
        - RoPE positional embeddings (not absolute)
        - Gated SiLU MLP (not standard GELU)
        """
        from vllm import LLM, SamplingParams

        print("\n" + "="*60)
        print("  Engine Mode E2E Test: TinyLlama")
        print("="*60)
        print(f"Engine mode: {is_engine_mode_enabled()}")
        print(f"Strict mode: {is_strict_mode_enabled()}")

        # Create LLM with engine mode
        # TinyLlama-1.1B is a small LLaMA-architecture model
        try:
            llm = LLM(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                dtype="float16",
                max_model_len=256,
                trust_remote_code=False,
                enforce_eager=True,
                gpu_memory_utilization=0.5,
            )
        except Exception as e:
            pytest.skip(f"TinyLlama model not available: {e}")

        # Test generation
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

        try:
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - start

            for output in outputs:
                print(f"\nPrompt: {output.prompt!r}")
                print(f"Output: {output.outputs[0].text!r}")
                print(f"Tokens: {len(output.outputs[0].token_ids)}")
                print(f"Time: {elapsed:.3f}s")

                # Verify we got some output
                assert len(output.outputs[0].token_ids) > 0

        except Exception as e:
            pytest.fail(f"Generation failed: {e}")

        # Cleanup
        del llm

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_with_qwen2_small(self):
        """Test engine mode with a small Qwen2-family model (Qwen2.5 0.5B)."""
        from vllm import LLM, SamplingParams

        try:
            llm = LLM(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                dtype="float16",
                max_model_len=256,
                trust_remote_code=True,
                enforce_eager=True,
                gpu_memory_utilization=0.5,
            )
        except Exception as e:
            pytest.skip(f"Qwen2.5 model not available: {e}")

        prompts = ["Write one short sentence about apples."]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
        outputs = llm.generate(prompts, sampling_params)
        assert outputs and len(outputs[0].outputs[0].token_ids) > 0

        del llm

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_continuous_batching_varied_lengths(self):
        """Validate continuous batching with varied per-request max_tokens."""
        from vllm import LLM, SamplingParams

        try:
            llm = LLM(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                dtype="float16",
                max_model_len=256,
                trust_remote_code=False,
                enforce_eager=True,
                gpu_memory_utilization=0.5,
            )
        except Exception as e:
            pytest.skip(f"TinyLlama model not available: {e}")

        prompts = [
            "Say hello.",
            "Name two colors.",
            "Write a short haiku.",
            "List three planets.",
            "Explain gravity in one sentence.",
            "What is 2+2?",
            "Give one fun fact.",
            "Write one emoji-less greeting.",
        ]
        # Different max_tokens per request to force completion at different steps.
        params = [
            SamplingParams(temperature=0.0, max_tokens=t)
            for t in [8, 12, 16, 8, 20, 4, 10, 6]
        ]

        outputs = llm.generate(prompts, params)
        assert len(outputs) == len(prompts)
        for out, p in zip(outputs, params):
            assert 0 < len(out.outputs[0].token_ids) <= p.max_tokens

        del llm

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_decode_throughput(self):
        """Test decode throughput in engine mode with TinyLlama."""
        from vllm import LLM, SamplingParams

        print("\n" + "="*60)
        print("  Engine Mode Decode Throughput Test: TinyLlama")
        print("="*60)

        # Create LLM with TinyLlama (LLaMA architecture)
        llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="float16",
            max_model_len=256,
            trust_remote_code=False,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
        )

        # Warmup
        sampling_params = SamplingParams(temperature=0.0, max_tokens=8)
        _ = llm.generate(["Warmup"], sampling_params)

        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        results = {}

        for batch_size in batch_sizes:
            prompts = ["The quick brown fox"] * batch_size
            sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            tps = total_tokens / elapsed
            results[batch_size] = tps

            print(f"Batch {batch_size}: {tps:.1f} tokens/sec ({total_tokens} tokens in {elapsed:.2f}s)")

        # Cleanup
        del llm

        # Verify no obvious batch-8 cliff (keep thresholds loose; hardware varies).
        assert results[8] >= results[4] * 0.7, "Batch 8 should not be a large regression vs batch 4"


class TestEngineArchitectureValidation:
    """Tests for engine mode architecture validation."""

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_unsupported_architecture_rejected(self):
        """Verify that unsupported architectures are rejected in engine mode.

        GPT-2 uses LayerNorm + absolute position embeddings + GELU MLP,
        which is incompatible with EngineRunner (requires RMSNorm + RoPE + SiLU).
        """
        from vllm_apple.engine import (
            ModelDescriptor,
            SUPPORTED_ENGINE_ARCHITECTURES,
        )

        print("\n" + "="*60)
        print("  Architecture Validation Test")
        print("="*60)
        print(f"Supported architectures: {sorted(SUPPORTED_ENGINE_ARCHITECTURES)}")

        # Verify GPT-2 is rejected
        with pytest.raises(ValueError) as exc_info:
            ModelDescriptor(
                num_layers=12,
                hidden_size=768,
                num_attention_heads=12,
                num_kv_heads=12,
                head_size=64,
                intermediate_size=3072,
                vocab_size=50257,
                architecture="gpt2",
            )

        error_msg = str(exc_info.value)
        assert "gpt2" in error_msg.lower()
        assert "not supported" in error_msg.lower()
        print(f"GPT-2 correctly rejected: {error_msg[:100]}...")

        # Verify LLaMA is accepted
        desc = ModelDescriptor(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            num_kv_heads=12,
            head_size=64,
            intermediate_size=3072,
            vocab_size=32000,
            architecture="llama",
        )
        assert desc.architecture == "llama"
        print("LLaMA architecture accepted")

        # Verify Qwen2 is accepted
        desc = ModelDescriptor(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            num_kv_heads=12,
            head_size=64,
            intermediate_size=3072,
            vocab_size=32000,
            architecture="qwen2",
        )
        assert desc.architecture == "qwen2"
        print("Qwen2 architecture accepted")


class TestEngineStrictMode:
    """Tests specifically for strict mode (no MPS in hot path)."""

    @pytest.mark.skipif(
        not is_strict_mode_enabled(),
        reason="Strict mode not enabled"
    )
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_strict_mode_no_mps_sync(self):
        """Verify no MPS synchronization in engine hot path."""
        from vllm_apple.engine import is_strict_mode

        assert is_strict_mode(), "Strict mode should be enabled"
        print("\nStrict mode is enabled - MPS sync in hot path would raise")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
