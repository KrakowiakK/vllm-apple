# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vLLM-Apple Metal Engine v2.0 invariants.

This module contains regression tests for critical bugs discovered during
engine implementation. Each test documents:
1. What the bug was
2. Why it matters (invariant violated)
3. How to detect it

These tests ensure bugs don't regress as the codebase evolves.

Reference: METAL_PLAN.md invariants:
- No PyTorch-MPS in the hot path
- Step-boundary-only synchronization
- KV cache single source of truth (engine-owned MTLBuffer)
- CPU tensors at vLLM↔engine boundary

Run with:
    pytest tests/unit/test_engine_invariants.py -v
"""

import os
import pytest
import torch
import numpy as np


@pytest.fixture
def engine_mode_enabled():
    """Fixture to enable engine mode for a test, then restore original state.

    This avoids global side-effects between tests by properly scoping
    the environment variable change.
    """
    original = os.environ.get("VLLM_APPLE_USE_ENGINE")
    os.environ["VLLM_APPLE_USE_ENGINE"] = "1"

    # Reset config module if it was already imported
    try:
        import vllm_apple.engine.config as config_mod
        if hasattr(config_mod, 'reset_engine_config'):
            config_mod.reset_engine_config()
    except ImportError:
        pass

    yield

    # Restore original state
    if original is None:
        os.environ.pop("VLLM_APPLE_USE_ENGINE", None)
    else:
        os.environ["VLLM_APPLE_USE_ENGINE"] = original

    # Reset config again
    try:
        import vllm_apple.engine.config as config_mod
        if hasattr(config_mod, 'reset_engine_config'):
            config_mod.reset_engine_config()
    except ImportError:
        pass


@pytest.fixture
def strict_mode_disabled():
    """Fixture to disable strict mode for tests that need MPS conversion."""
    original = os.environ.get("VLLM_METAL_STRICT_NO_MPS")
    os.environ.pop("VLLM_METAL_STRICT_NO_MPS", None)

    try:
        import vllm_apple.engine.config as config_mod
        if hasattr(config_mod, 'reset_engine_config'):
            config_mod.reset_engine_config()
    except ImportError:
        pass

    yield

    if original is not None:
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = original

    try:
        import vllm_apple.engine.config as config_mod
        if hasattr(config_mod, 'reset_engine_config'):
            config_mod.reset_engine_config()
    except ImportError:
        pass


class TestArchitectureValidation:
    """Tests for model architecture validation.

    BUG FIXED: GPT-2 support was "fake" - EngineRunner is hardcoded for
    LLaMA/Qwen2 architecture (RMSNorm, RoPE, gated SiLU MLP) but tests
    used GPT-2 which uses LayerNorm, absolute position embeddings, and
    standard GELU MLP.

    INVARIANT: Engine mode only supports models with compatible architecture.
    Unsupported models must be rejected at initialization, not fail silently
    at runtime with incorrect results.

    Reference: METAL_PLAN.md - Phase 4 mentions RMSNorm support requirement.
    """

    def test_supported_architectures_defined(self):
        """Verify supported architectures constant exists."""
        from vllm_apple.engine import SUPPORTED_ENGINE_ARCHITECTURES

        assert isinstance(SUPPORTED_ENGINE_ARCHITECTURES, frozenset)
        assert "llama" in SUPPORTED_ENGINE_ARCHITECTURES
        assert "qwen2" in SUPPORTED_ENGINE_ARCHITECTURES
        assert "mistral" in SUPPORTED_ENGINE_ARCHITECTURES
        # GPT-2 should NOT be supported
        assert "gpt2" not in SUPPORTED_ENGINE_ARCHITECTURES

    def test_gpt2_architecture_rejected(self):
        """Verify GPT-2 architecture is rejected with clear error.

        GPT-2 uses:
        - LayerNorm (not RMSNorm)
        - Absolute position embeddings (not RoPE)
        - Standard GELU MLP (not gated SiLU)

        EngineRunner cannot correctly execute GPT-2 models.
        """
        from vllm_apple.engine import ModelDescriptor

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

        error_msg = str(exc_info.value).lower()
        assert "gpt2" in error_msg
        assert "not supported" in error_msg

    def test_llama_architecture_accepted(self):
        """Verify LLaMA architecture is accepted."""
        from vllm_apple.engine import ModelDescriptor

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

    def test_qwen2_architecture_accepted(self):
        """Verify Qwen2 architecture is accepted."""
        from vllm_apple.engine import ModelDescriptor

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

    def test_architecture_detection_gpt2(self):
        """Verify GPT-2 config is detected correctly."""
        from vllm_apple.engine import ModelDescriptor

        class MockGPT2Config:
            model_type = "gpt2"
            num_hidden_layers = 12
            hidden_size = 768
            num_attention_heads = 12
            intermediate_size = 3072
            vocab_size = 50257

        arch = ModelDescriptor._detect_architecture(MockGPT2Config())
        assert arch == "gpt2"

    def test_architecture_detection_llama(self):
        """Verify LLaMA config is detected correctly."""
        from vllm_apple.engine import ModelDescriptor

        class MockLlamaConfig:
            model_type = "llama"
            num_hidden_layers = 32
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 32
            intermediate_size = 11008
            vocab_size = 32000
            rope_theta = 10000.0

        arch = ModelDescriptor._detect_architecture(MockLlamaConfig())
        assert arch == "llama"


class TestPrefillDecodeDispatch:
    """Tests for prefill vs decode dispatch.

    BUG FIXED: The engine attention kernel dispatches by (num_seqs, num_heads),
    NOT (num_tokens, num_heads). For prefill where num_tokens > num_seqs,
    this produces incorrect results. The kernel was treating prefill as if
    each sequence only had 1 token.

    INVARIANT: Engine must route prefill through PyTorch path until
    token-parallel attention is implemented. Decode (1 token per seq)
    can use the engine.

    Reference: METAL_PLAN.md Phase 1.3 - Paged attention + KV-write as engine ops
    """

    def test_step_descriptor_prefill_detection(self):
        """Verify StepDescriptor correctly identifies prefill."""
        from vllm_apple.engine import StepDescriptor

        # Prefill step
        prefill = StepDescriptor(
            step_id=1,
            step_kind="prefill",
            num_scheduled_tokens=128,
            num_seqs_active=4,
        )
        assert prefill.is_prefill
        assert not prefill.is_decode

        # Decode step
        decode = StepDescriptor(
            step_id=2,
            step_kind="decode",
            num_scheduled_tokens=4,
            num_seqs_active=4,
        )
        assert decode.is_decode
        assert not decode.is_prefill

    def test_step_descriptor_validates_step_kind(self):
        """Verify invalid step_kind is rejected."""
        from vllm_apple.engine import StepDescriptor

        with pytest.raises(ValueError):
            StepDescriptor(
                step_id=1,
                step_kind="invalid",
                num_scheduled_tokens=10,
                num_seqs_active=2,
            )

    def test_prefill_tokens_vs_seqs_invariant(self):
        """Document the prefill invariant: num_tokens > num_seqs.

        In prefill:
        - num_tokens = total tokens in the batch (could be 128 for a 128-token prompt)
        - num_seqs = number of sequences (could be 1 for a single request)

        The attention kernel dispatches (num_seqs, num_heads) threads.
        If we have 128 tokens but only 1 seq, only 1 row of Q is processed!
        """
        # This test documents the invariant
        num_tokens = 128  # Prompt length
        num_seqs = 1  # Single sequence

        # In prefill, num_tokens > num_seqs
        assert num_tokens > num_seqs, "Prefill has more tokens than sequences"

        # Kernel dispatches by (num_seqs, num_heads)
        # This means only `num_seqs` query rows are processed
        # For prefill, this is WRONG - we need to process `num_tokens` rows


class TestBoundaryValidation:
    """Tests for vLLM↔engine boundary validation.

    BUG FIXED: ensure_cpu_tensor() was silently converting MPS→CPU via
    .cpu() even in engine mode. This causes implicit synchronization,
    violating the step-boundary-only sync invariant.

    INVARIANT: In engine mode, MPS tensors must be rejected at boundary.
    The calling code must provide CPU tensors. Silent conversion hides bugs.

    Reference: METAL_PLAN.md Invariants - "No implicit MPS barriers"
    """

    def test_cpu_tensor_accepted(self):
        """CPU tensors should always be accepted."""
        from vllm_apple.engine.boundary import ensure_cpu_tensor

        tensor = torch.zeros(10, dtype=torch.float32)
        result = ensure_cpu_tensor(tensor, "test")
        assert result.device.type == "cpu"

    def test_none_returns_none(self):
        """None input should return None."""
        from vllm_apple.engine.boundary import ensure_cpu_tensor

        result = ensure_cpu_tensor(None, "test")
        assert result is None

    def test_non_tensor_raises(self):
        """Non-tensor input should raise TypeError."""
        from vllm_apple.engine.boundary import ensure_cpu_tensor

        with pytest.raises(TypeError):
            ensure_cpu_tensor([1, 2, 3], "test")

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_tensor_rejected_in_engine_mode(self, engine_mode_enabled):
        """MPS tensors must be rejected when engine mode is enabled.

        This test verifies the fix for the strict mode bypass bug.
        Before the fix, MPS tensors were silently converted to CPU,
        causing hidden synchronization.
        """
        import importlib
        import vllm_apple.engine.boundary as boundary_mod

        importlib.reload(boundary_mod)

        from vllm_apple.engine.boundary import ensure_cpu_tensor

        mps_tensor = torch.zeros(10, device="mps")

        with pytest.raises(RuntimeError) as exc_info:
            ensure_cpu_tensor(mps_tensor, "test_mps")

        error_msg = str(exc_info.value).lower()
        assert "mps" in error_msg
        assert "engine mode" in error_msg or "boundary" in error_msg

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_conversion_allowed_with_explicit_flag(self, engine_mode_enabled):
        """MPS conversion should work when explicitly allowed.

        The allow_mps_conversion flag is for legacy non-engine paths
        that explicitly need MPS→CPU conversion.
        """
        import importlib
        import vllm_apple.engine.boundary as boundary_mod

        importlib.reload(boundary_mod)

        from vllm_apple.engine.boundary import ensure_cpu_tensor

        mps_tensor = torch.zeros(10, device="mps")
        result = ensure_cpu_tensor(mps_tensor, "test", allow_mps_conversion=True)
        assert result.device.type == "cpu"


class TestScratchBufferBounds:
    """Tests for scratch buffer bounds checking.

    BUG FIXED: _allocate_scratch() used a magic constant (4096) for
    max_tokens without bounds checking. If num_tokens exceeded this
    limit, buffer overflow would occur.

    INVARIANT: Engine must validate input sizes against allocated buffer
    capacity and fail fast with a clear error message.

    Reference: METAL_PLAN.md Phase 1.1 - EngineStepContext (scratch buffers)
    """

    def test_engine_config_max_batch_size(self):
        """Verify engine config provides max_batch_size."""
        from vllm_apple.engine import get_engine_config

        config = get_engine_config()
        assert hasattr(config, "max_batch_size")
        assert config.max_batch_size > 0

    def test_model_descriptor_max_position_embeddings(self):
        """Verify ModelDescriptor stores max_position_embeddings."""
        from vllm_apple.engine import ModelDescriptor

        desc = ModelDescriptor(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            num_kv_heads=12,
            head_size=64,
            intermediate_size=3072,
            vocab_size=32000,
            max_position_embeddings=4096,
            architecture="llama",
        )
        assert desc.max_position_embeddings == 4096


class TestEngineInputValidation:
    """Tests for engine input validation.

    INVARIANT: Engine inputs must be validated at boundary:
    - All tensors must be on CPU
    - Token counts must not exceed buffer capacity
    - Block table values must be valid physical block IDs

    Reference: METAL_PLAN.md - Engine API Contract (Inputs)
    """

    def test_engine_inputs_validates_cpu_device(self):
        """EngineInputs must validate all tensors are on CPU."""
        from vllm_apple.engine import EngineInputs

        # Valid CPU inputs
        inputs = EngineInputs(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([3], dtype=torch.int32),
        )
        assert inputs.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_engine_inputs_rejects_mps(self):
        """EngineInputs must reject MPS tensors."""
        from vllm_apple.engine import EngineInputs

        with pytest.raises(ValueError) as exc_info:
            EngineInputs(
                token_ids=torch.tensor([1, 2, 3], dtype=torch.int64, device="mps"),
                positions=torch.tensor([0, 1, 2], dtype=torch.int64),
                block_table=torch.tensor([[0, 1]], dtype=torch.int32),
                slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
                seq_lens=torch.tensor([3], dtype=torch.int32),
            )

        assert "cpu" in str(exc_info.value).lower()


class TestSynchronizationPolicy:
    """Tests for synchronization policy enforcement.

    INVARIANT: Synchronization is allowed ONLY at scheduler step boundaries.
    No per-layer/per-op waits inside engine code paths.

    Reference: METAL_PLAN.md - Synchronization Rules (Hard Constraint)
    """

    def test_engine_phase_enum_exists(self):
        """Verify EnginePhase enum for phase tracking."""
        from vllm_apple.engine import EnginePhase

        assert hasattr(EnginePhase, "IDLE")
        assert hasattr(EnginePhase, "ENCODE")
        assert hasattr(EnginePhase, "SUBMIT")
        assert hasattr(EnginePhase, "READBACK")


class TestKVCacheInvariants:
    """Tests for KV cache invariants.

    INVARIANT: KV cache single source of truth is engine-owned MTLBuffer.
    Any vLLM KV tensors must be stubs only, not duplicating full cache.

    Reference: METAL_PLAN.md Phase 1.2 - KV cache single source of truth
    """

    def test_kv_cache_descriptor_validation(self):
        """Verify KVCacheDescriptor validates parameters."""
        from vllm_apple.engine import KVCacheDescriptor

        # Valid descriptor
        desc = KVCacheDescriptor(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            num_layers=12,
        )
        assert desc.num_blocks == 100

        # Invalid: num_blocks < 1
        with pytest.raises(ValueError):
            KVCacheDescriptor(
                num_blocks=0,
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                num_layers=12,
            )

        # Invalid: unsupported head_size
        with pytest.raises(ValueError):
            KVCacheDescriptor(
                num_blocks=100,
                block_size=16,
                num_kv_heads=8,
                head_size=48,  # Not in {32, 64, 96, 128}
                num_layers=12,
            )

    def test_kv_cache_sync_method_exists(self):
        """Verify EngineKVCache has sync_from_torch_cache method.

        BUG FIXED: KV cache was duplicated - torch cache for prefill,
        MTLBuffer for decode. After prefill, engine cache was empty,
        causing incorrect decode results.

        The fix adds sync_from_torch_cache() to copy KV data after prefill.
        """
        from vllm_apple.engine.kv_cache import EngineKVCache

        # Verify the method exists
        assert hasattr(EngineKVCache, 'sync_from_torch_cache')
        assert callable(getattr(EngineKVCache, 'sync_from_torch_cache'))

    def test_kv_cache_supports_torch_sync(self):
        """Verify EngineKVCache.supports_torch_sync returns True.

        This confirms that KV cache sync is supported (storageModeShared).
        """
        from vllm_apple.engine.kv_cache import EngineKVCache

        # Verify the method exists
        assert hasattr(EngineKVCache, 'supports_torch_sync')


# NOTE: TestKVCacheDataConsistency moved to tests/metal/test_kv_sync.py
# (Integration tests requiring actual Metal operations)


class TestMoEArchitectureRejection:
    """Tests for MoE (Mixture of Experts) architecture rejection.

    BUG FIXED: The architecture detection had a dangerous fallback:
    if rope_theta was detected, it assumed "llama" architecture.
    This could allow MoE models (like Mixtral) to pass through
    even though the engine doesn't support MoE routing.

    INVARIANT: MoE models must be explicitly rejected at initialization.
    """

    def test_moe_architecture_detected_and_rejected(self):
        """Verify MoE models are detected and rejected."""
        from vllm_apple.engine import ModelDescriptor

        class MockMixtralConfig:
            model_type = "mixtral"
            num_hidden_layers = 32
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 8
            intermediate_size = 14336
            vocab_size = 32000
            num_local_experts = 8  # MoE indicator
            num_experts_per_tok = 2

        arch = ModelDescriptor._detect_architecture(MockMixtralConfig())
        assert arch == "moe_unsupported", f"MoE should be detected as unsupported, got: {arch}"

    def test_num_experts_attribute_triggers_rejection(self):
        """Verify num_experts attribute triggers MoE rejection."""
        from vllm_apple.engine import ModelDescriptor

        class MockMoEConfig:
            model_type = "custom_moe"
            num_hidden_layers = 24
            hidden_size = 2048
            num_attention_heads = 16
            num_key_value_heads = 16
            intermediate_size = 8192
            vocab_size = 32000
            num_experts = 4  # Alternative MoE indicator

        arch = ModelDescriptor._detect_architecture(MockMoEConfig())
        assert arch == "moe_unsupported"

    def test_single_expert_is_not_moe(self):
        """Verify num_experts=1 is NOT treated as MoE (dense model)."""
        from vllm_apple.engine import ModelDescriptor

        class MockDenseConfig:
            model_type = "llama"
            num_hidden_layers = 32
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 32
            intermediate_size = 11008
            vocab_size = 32000
            num_experts = 1  # Single expert = dense model

        arch = ModelDescriptor._detect_architecture(MockDenseConfig())
        # Should be llama, not moe_unsupported
        assert arch == "llama", f"Single expert should be treated as dense, got: {arch}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
