# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm-apple engine v2.0 components.

Tests engine components in isolation without requiring full model loading.

Run with: pytest tests/unit/test_engine.py -v
"""

import os
import sys
import pytest
import numpy as np

# Apply MPS patches before importing anything else
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


# Note: Most tests here don't require MPS - they test pure Python logic.
# Only add @pytest.mark.skipif(not torch.backends.mps.is_available(), ...)
# to individual tests that actually need Metal/MPS.


class TestEngineImports:
    """Test that engine modules can be imported."""

    def test_import_engine_config(self):
        """Test engine config imports."""
        from vllm_apple.engine import (
            EngineConfig,
            is_engine_mode_enabled,
            get_engine_config,
        )
        assert EngineConfig is not None
        # Engine mode disabled by default
        assert not is_engine_mode_enabled()

    def test_import_engine_guards(self):
        """Test engine guards imports."""
        from vllm_apple.engine import (
            EnginePhase,
            EngineHotPathGuard,
            StepPhaseController,
        )
        assert EnginePhase.IDLE is not None
        assert EnginePhase.ENCODE is not None
        assert EnginePhase.SUBMIT is not None
        assert EnginePhase.READBACK is not None

    def test_import_engine_descriptors(self):
        """Test engine descriptors imports."""
        from vllm_apple.engine import (
            StepDescriptor,
            BatchDescriptor,
            EngineInputs,
            EngineOutputs,
            KVCacheDescriptor,
            ModelDescriptor,
        )
        assert StepDescriptor is not None
        assert EngineInputs is not None

    def test_import_engine_ops(self):
        """Test engine ops imports."""
        from vllm_apple.engine.ops import (
            EngineEmbedding,
            EngineRMSNorm,
            EngineQKVProjection,
            EngineOProjection,
            EngineRoPE,
            EngineElementwiseOps,
            EngineMLP,
            EngineLMHead,
            EngineGEMM,
            PagedAttentionOp,
            KVWriteOp,
        )
        assert EngineEmbedding is not None
        assert EngineGEMM is not None


class TestEnginePhases:
    """Test engine phase state machine."""

    def test_phase_transitions(self):
        """Test phase transitions are tracked correctly."""
        from vllm_apple.engine import EngineHotPathGuard, EnginePhase

        # Initial state
        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE

        # Transition to ENCODE
        EngineHotPathGuard.set_phase(EnginePhase.ENCODE)
        assert EngineHotPathGuard.get_phase() == EnginePhase.ENCODE

        # Transition to SUBMIT
        EngineHotPathGuard.set_phase(EnginePhase.SUBMIT)
        assert EngineHotPathGuard.get_phase() == EnginePhase.SUBMIT

        # Transition to READBACK
        EngineHotPathGuard.set_phase(EnginePhase.READBACK)
        assert EngineHotPathGuard.get_phase() == EnginePhase.READBACK

        # Back to IDLE
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE


class TestEngineTensor:
    """Test EngineTensor abstraction."""

    def test_tensor_creation(self):
        """Test EngineTensor can wrap MTLBuffer."""
        from vllm_apple.engine.tensor import EngineTensor, EngineDType

        # Create dummy buffer (None for test without Metal context)
        tensor = EngineTensor(
            buffer=None,  # Would be MTLBuffer in real usage
            shape=(32, 128),
            dtype=EngineDType.FLOAT16,
            offset=0,
        )

        assert tensor.shape == (32, 128)
        assert tensor.dtype == EngineDType.FLOAT16
        assert tensor.numel == 32 * 128  # numel is a property, not a method
        assert tensor.nbytes == 32 * 128 * 2  # float16 = 2 bytes

    def test_tensor_with_offset(self):
        """Test EngineTensor with byte offset (for buffer views)."""
        from vllm_apple.engine.tensor import EngineTensor, EngineDType

        # K tensor at offset in QKV buffer
        k_offset = 4096 * 2  # After Q section (in bytes)
        tensor = EngineTensor(
            buffer=None,
            shape=(32, 8, 128),  # [num_tokens, num_kv_heads, head_size]
            dtype=EngineDType.FLOAT16,
            offset=k_offset,
        )

        assert tensor.offset == k_offset
        assert tensor.shape == (32, 8, 128)


class TestRMSNormPowerOfTwo:
    """Test RMSNorm power-of-2 thread count."""

    def test_power_of_two_rounding(self):
        """Test that threads_per_group is rounded to power of 2."""
        # This tests the logic, not the actual kernel

        def compute_threads_per_group(hidden_size: int) -> int:
            threads_per_group = min(256, hidden_size)
            threads_per_group = max(32, threads_per_group)
            # Round up to next power of 2
            threads_per_group = 1 << (threads_per_group - 1).bit_length()
            return threads_per_group

        # Test cases
        assert compute_threads_per_group(100) == 128  # 100 -> 128
        assert compute_threads_per_group(128) == 128  # Already power of 2
        assert compute_threads_per_group(200) == 256  # 200 -> 256
        assert compute_threads_per_group(256) == 256  # Already power of 2
        assert compute_threads_per_group(4096) == 256  # Capped at 256
        assert compute_threads_per_group(16) == 32  # Min 32


class TestRoPEBoundsCheck:
    """Test RoPE position bounds checking."""

    def test_bounds_check_logic(self):
        """Test RoPE bounds check raises on invalid position."""
        max_position = 8192

        # Valid positions should not raise
        for max_pos_in_batch in [0, 100, 4096, 8191]:
            if max_pos_in_batch >= max_position:
                with pytest.raises(ValueError):
                    raise ValueError(
                        f"Position ID {max_pos_in_batch} exceeds max_position {max_position}"
                    )
            # No exception for valid positions

        # Invalid position should raise
        with pytest.raises(ValueError, match="exceeds max_position"):
            max_pos_in_batch = 8192  # Equal to max
            if max_pos_in_batch >= max_position:
                raise ValueError(
                    f"Position ID {max_pos_in_batch} exceeds max_position {max_position}"
                )


class TestDtypeCasting:
    """Test dtype casting for Metal kernels."""

    def test_int32_casting(self):
        """Test int64 tensors are cast to int32."""
        # Simulate the runner logic
        token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        # Cast to int32 (as runner does)
        token_ids_int32 = token_ids.to(torch.int32) if token_ids.dtype != torch.int32 else token_ids
        positions_int32 = positions.to(torch.int32) if positions.dtype != torch.int32 else positions

        assert token_ids_int32.dtype == torch.int32
        assert positions_int32.dtype == torch.int32

        # Values should be preserved
        assert torch.equal(token_ids_int32, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))

    def test_already_int32(self):
        """Test tensors already int32 are not converted."""
        token_ids = torch.tensor([1, 2, 3], dtype=torch.int32)

        # Should return same tensor (no copy)
        result = token_ids.to(torch.int32) if token_ids.dtype != torch.int32 else token_ids
        assert result is token_ids  # Same object


class TestGEMMTranspose:
    """Test GEMM transpose logic for PyTorch weights."""

    def test_weight_shapes(self):
        """Test that PyTorch weight shapes are [out, in]."""
        # PyTorch Linear stores weight as [out_features, in_features]
        hidden_size = 4096
        intermediate_size = 11008

        # MLP weights in PyTorch format
        gate_proj_shape = (intermediate_size, hidden_size)  # [I, H]
        up_proj_shape = (intermediate_size, hidden_size)    # [I, H]
        down_proj_shape = (hidden_size, intermediate_size)  # [H, I]

        # For GEMM: C = A @ B
        # If A is [M, K] and B is [K, N], result is [M, N]
        # But our weight is [N, K], so we need B^T

        # gate_proj: x @ W^T = [num_tokens, H] @ [I, H]^T = [num_tokens, I]
        # This requires transpose_B=True

        # Verify the shapes match what we expect
        assert gate_proj_shape == (intermediate_size, hidden_size)
        assert down_proj_shape == (hidden_size, intermediate_size)


class TestKVCacheDescriptor:
    """Test KV cache descriptor validation."""

    def test_descriptor_validation(self):
        """Test KVCacheDescriptor validates parameters."""
        from vllm_apple.engine.descriptors import KVCacheDescriptor

        # Valid descriptor
        desc = KVCacheDescriptor(
            num_blocks=1000,
            block_size=16,
            num_kv_heads=32,
            head_size=128,
            num_layers=32,
        )
        desc.__post_init__()  # Should not raise

        assert desc.num_blocks == 1000
        assert desc.block_size == 16
        # Verify total cache size is computed correctly
        assert desc.total_cache_mb > 0

    def test_invalid_head_size(self):
        """Test KVCacheDescriptor rejects invalid head sizes."""
        from vllm_apple.engine.descriptors import KVCacheDescriptor

        with pytest.raises(ValueError, match="head_size"):
            desc = KVCacheDescriptor(
                num_blocks=1000,
                block_size=16,
                num_kv_heads=32,
                head_size=100,  # Invalid - not in [32, 64, 96, 128]
                num_layers=32,
            )
            desc.__post_init__()


class TestStepDescriptor:
    """Test step descriptor."""

    def test_step_kind(self):
        """Test step kind detection."""
        from vllm_apple.engine.descriptors import StepDescriptor

        # Decode step
        decode_step = StepDescriptor(
            step_id=0,
            step_kind="decode",
            num_scheduled_tokens=4,
            num_seqs_active=4,
            max_num_blocks_per_seq=64,
        )
        assert decode_step.is_decode
        assert not decode_step.is_prefill

        # Prefill step
        prefill_step = StepDescriptor(
            step_id=1,
            step_kind="prefill",
            num_scheduled_tokens=512,
            num_seqs_active=2,
            max_num_blocks_per_seq=64,
        )
        assert prefill_step.is_prefill
        assert not prefill_step.is_decode


class TestKVWriteParams:
    """Test KV write parameter calculation."""

    def test_stride_calculation(self):
        """Test KV cache stride calculations."""
        num_kv_heads = 8
        head_size = 128
        block_size = 16

        # KV cache layout: [num_blocks, num_kv_heads, block_size, head_size]
        kv_stride_block = num_kv_heads * block_size * head_size
        kv_stride_head = block_size * head_size
        kv_stride_token = head_size

        assert kv_stride_block == 8 * 16 * 128  # 16384
        assert kv_stride_head == 16 * 128       # 2048
        assert kv_stride_token == 128

        # New K/V layout: [num_tokens, num_kv_heads, head_size]
        new_kv_stride_token = num_kv_heads * head_size
        new_kv_stride_head = head_size

        assert new_kv_stride_token == 8 * 128  # 1024
        assert new_kv_stride_head == 128


class TestSlotMappingValidation:
    """Test slot_mapping validation for prefill KV write.

    These tests call the actual validate_slot_mapping() method using a mock
    KV cache object. The validation is pure Python/numpy and doesn't require MPS.
    """

    def _create_mock_kv_cache(self, num_blocks: int, block_size: int):
        """Create a mock object with minimal _desc for validation."""
        from dataclasses import dataclass

        @dataclass
        class MockDesc:
            num_blocks: int
            block_size: int

        class MockKVCache:
            def __init__(self, num_blocks, block_size):
                self._desc = MockDesc(num_blocks=num_blocks, block_size=block_size)

        # Import and bind the actual method
        from vllm_apple.engine.kv_cache import EngineKVCache
        mock = MockKVCache(num_blocks, block_size)
        # Bind the actual validation method to our mock
        mock.validate_slot_mapping = lambda slot_mapping, context="": \
            EngineKVCache.validate_slot_mapping(mock, slot_mapping, context)
        return mock

    def test_valid_slot_mapping(self):
        """Test valid slot_mapping passes validation."""
        mock_cache = self._create_mock_kv_cache(num_blocks=100, block_size=16)
        max_valid_slot = 100 * 16 - 1  # 1599

        # Valid slot_mapping - should not raise
        slot_mapping = np.array([0, 1, 15, 100, max_valid_slot], dtype=np.int32)
        mock_cache.validate_slot_mapping(slot_mapping)  # No exception

    def test_invalid_slot_oob(self):
        """Test OOB slot_mapping raises ValueError."""
        mock_cache = self._create_mock_kv_cache(num_blocks=100, block_size=16)

        # Invalid: slot 2000 is out of bounds (max is 1599)
        slot_mapping = np.array([0, 1, 2000], dtype=np.int32)

        with pytest.raises(ValueError, match="Slot mapping contains index 2000"):
            mock_cache.validate_slot_mapping(slot_mapping)

    def test_negative_one_allowed(self):
        """Test -1 is allowed in slot_mapping (padding)."""
        mock_cache = self._create_mock_kv_cache(num_blocks=100, block_size=16)

        # -1 should be allowed for padding
        slot_mapping = np.array([0, -1, 10, -1, 20], dtype=np.int32)
        mock_cache.validate_slot_mapping(slot_mapping)  # No exception

    def test_invalid_negative(self):
        """Test invalid negative values raise ValueError."""
        mock_cache = self._create_mock_kv_cache(num_blocks=100, block_size=16)

        # -2 is invalid (only -1 allowed)
        slot_mapping = np.array([0, -2, 10], dtype=np.int32)

        with pytest.raises(ValueError, match="invalid index -2"):
            mock_cache.validate_slot_mapping(slot_mapping)

    def test_torch_tensor_input(self):
        """Test validation works with torch.Tensor on CPU."""
        mock_cache = self._create_mock_kv_cache(num_blocks=100, block_size=16)

        # Valid torch tensor
        slot_mapping = torch.tensor([0, 10, 100], dtype=torch.int32)
        mock_cache.validate_slot_mapping(slot_mapping)  # No exception

    def test_empty_slot_mapping(self):
        """Test empty slot_mapping passes validation."""
        mock_cache = self._create_mock_kv_cache(num_blocks=100, block_size=16)

        # Empty array should pass (no slots to validate)
        slot_mapping = np.array([], dtype=np.int32)
        mock_cache.validate_slot_mapping(slot_mapping)  # No exception


class TestEngineRunnerAPI:
    """Ensure runner API stays consistent for ops."""

    def test_encode_transformer_layer_accepts_max_blocks_per_seq(self):
        """Regression test: max_blocks_per_seq must be passed into layer encode."""
        import inspect
        from vllm_apple.engine.runner import EngineRunner

        sig = inspect.signature(EngineRunner._encode_transformer_layer)
        assert "max_blocks_per_seq" in sig.parameters


class TestSeqLensSemantics:
    """Test seq_lens semantic correctness for engine path.

    CRITICAL: Engine expects FULL context lengths (computed + new tokens),
    NOT query lengths (new tokens per step).
    """

    def test_seq_lens_source_in_engine_inputs(self):
        """Verify _execute_forward_engine uses attn_metadata.seq_lens, not model_input.seq_lens.

        This is a regression test for the critical bug where query lengths were
        passed instead of full context lengths, causing:
        - Wrong max_seq_len for attention
        - Wrong KV-write positions (seq_len - 1)
        - Wrong context lengths in paged attention
        """
        import inspect
        import ast

        # Read the source of _execute_forward_engine
        from vllm_apple.v1.worker.apple_model_runner import AppleModelRunner

        source = inspect.getsource(AppleModelRunner._execute_forward_engine)

        # Parse the source to find EngineInputs construction
        # The seq_lens= argument should reference attn_metadata.seq_lens, NOT model_input.seq_lens
        assert "attn_metadata.seq_lens" in source, \
            "EngineInputs.seq_lens should use attn_metadata.seq_lens (full context lengths)"

        # Verify model_input.seq_lens is NOT used for EngineInputs.seq_lens
        # (it may appear elsewhere for other purposes like logits extraction)
        lines = source.split('\n')
        for line in lines:
            if 'seq_lens=' in line and 'EngineInputs' in source[:source.find(line)]:
                assert 'model_input.seq_lens' not in line, \
                    f"EngineInputs.seq_lens must NOT use model_input.seq_lens (query lengths): {line}"

    def test_full_vs_query_seq_lens_semantics(self):
        """Document the semantic difference between full and query seq_lens."""
        # Scenario: decode step with 4 sequences, each generating 1 new token
        # Query lengths (model_input.seq_lens): [1, 1, 1, 1] - new tokens per step
        query_seq_lens = [1, 1, 1, 1]

        # Full context lengths (attn_metadata.seq_lens): actual context sizes
        # e.g., sequences with 512, 256, 128, 64 computed tokens + 1 new each
        full_seq_lens = [513, 257, 129, 65]

        # Engine MUST use full_seq_lens for:
        # 1. max_seq_len for attention (determines KV range to attend to)
        max_seq_len = max(full_seq_lens)
        assert max_seq_len == 513, "max_seq_len should be 513 (largest context)"
        assert max(query_seq_lens) == 1, "query max is only 1 (wrong for attention)"

        # 2. KV-write position (seq_len - 1 gives slot for new token)
        # For sequence 0: position should be 512 (0-indexed), not 0
        kv_write_positions = [s - 1 for s in full_seq_lens]
        assert kv_write_positions[0] == 512, "KV write position should be 512"

        wrong_positions = [s - 1 for s in query_seq_lens]
        assert wrong_positions[0] == 0, "Using query lens gives wrong position 0"


class TestMaxSeqLenCalculation:
    """Test max_seq_len calculation from seq_lens."""

    def test_max_seq_len_from_tensor(self):
        """Test computing max_seq_len from seq_lens tensor."""
        seq_lens = torch.tensor([100, 200, 150, 300], dtype=torch.int32)

        # This is what runner does
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0

        assert max_seq_len == 300

    def test_empty_seq_lens(self):
        """Test max_seq_len with empty tensor."""
        seq_lens = torch.tensor([], dtype=torch.int32)

        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0

        assert max_seq_len == 0


class TestFusedWeightLayouts:
    """Regression tests for fused-weight buffer layouts.

    These tests validate encode-time buffer slicing and offsets without requiring
    Metal/MPS runtime availability.
    """

    def test_qkv_fused_multi_token_writes_stacked_layout(self, monkeypatch):
        import vllm_apple.engine.ops.qkv as qkv_mod
        from vllm_apple.engine.tensor import EngineTensor

        gemm_calls: list[dict] = []

        class StubGEMM:

            def __init__(self, context):
                pass

            def encode(self, **kwargs):
                gemm_calls.append(kwargs)

        monkeypatch.setattr(qkv_mod, "EngineGEMM", StubGEMM)

        class DummyBuffer:

            def __init__(self, nbytes: int):
                self._nbytes = nbytes

            def length(self) -> int:
                return self._nbytes

        class StepCtx:
            is_encoding = True

            def memory_barrier(self) -> None:
                pass

        hidden_size = 16
        num_heads = 2
        num_kv_heads = 1
        head_size = 4
        num_tokens = 2

        q_size = num_heads * head_size
        k_size = num_kv_heads * head_size
        v_size = num_kv_heads * head_size
        qkv_size = q_size + k_size + v_size

        proj = qkv_mod.EngineQKVProjection(
            context=object(),
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            fused=True,
        )

        weight = DummyBuffer(qkv_size * hidden_size * 2)
        proj.set_weights(qkv_weight=weight)

        hidden_states = DummyBuffer(num_tokens * hidden_size * 2)
        qkv_output = DummyBuffer(num_tokens * qkv_size * 2)

        q_info, k_info, v_info = proj.encode(
            step_ctx=StepCtx(),
            hidden_states=hidden_states,
            qkv_output=qkv_output,
            num_tokens=num_tokens,
        )

        assert len(gemm_calls) == 3
        q_call, k_call, v_call = gemm_calls

        # Output slices are stacked by projection (Q then K then V).
        assert isinstance(q_call["C"], EngineTensor)
        assert isinstance(k_call["C"], EngineTensor)
        assert isinstance(v_call["C"], EngineTensor)

        assert q_call["C"].offset == 0
        assert k_call["C"].offset == num_tokens * q_size * 2
        assert v_call["C"].offset == num_tokens * (q_size + k_size) * 2

        assert q_call["N"] == q_size
        assert k_call["N"] == k_size
        assert v_call["N"] == v_size
        assert q_call["transpose_B"] is True
        assert k_call["transpose_B"] is True
        assert v_call["transpose_B"] is True

        # Fused weight buffer is sliced by rows: [q][k][v].
        assert isinstance(q_call["B"], EngineTensor)
        assert isinstance(k_call["B"], EngineTensor)
        assert isinstance(v_call["B"], EngineTensor)

        assert q_call["B"].offset == 0
        assert k_call["B"].offset == q_size * hidden_size * 2
        assert v_call["B"].offset == (q_size + k_size) * hidden_size * 2

        # Returned offsets match the stacked layout contract.
        assert q_info["offset"] == 0
        assert k_info["offset"] == num_tokens * q_size * 2
        assert v_info["offset"] == num_tokens * (q_size + k_size) * 2

    def test_mlp_fused_gate_up_slices_weights(self, monkeypatch):
        import vllm_apple.engine.ops.mlp as mlp_mod
        from vllm_apple.engine.tensor import EngineTensor

        gemm_calls: list[dict] = []
        silu_mul_calls: list[dict] = []

        class StubGEMM:

            def __init__(self, context):
                pass

            def encode(self, **kwargs):
                gemm_calls.append(kwargs)

        class StubElementwiseOps:

            def __init__(self, context):
                pass

            def encode_silu_mul(self, **kwargs):
                silu_mul_calls.append(kwargs)

        monkeypatch.setattr(mlp_mod, "EngineGEMM", StubGEMM)
        monkeypatch.setattr(mlp_mod, "EngineElementwiseOps", StubElementwiseOps)

        class DummyBuffer:

            def __init__(self, nbytes: int):
                self._nbytes = nbytes

            def length(self) -> int:
                return self._nbytes

        class StepCtx:
            is_encoding = True

            def __init__(self):
                self.allocs: list[DummyBuffer] = []

            def allocate_scratch(self, size: int):
                buf = DummyBuffer(size)
                self.allocs.append(buf)
                return buf

            def memory_barrier(self) -> None:
                pass

        hidden_size = 16
        intermediate_size = 8
        num_tokens = 3

        gate_up_weight = DummyBuffer(2 * intermediate_size * hidden_size * 2)
        down_weight = DummyBuffer(hidden_size * intermediate_size * 2)

        mlp = mlp_mod.EngineMLP(
            context=object(),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gated=True,
            activation="silu",
        )
        mlp.set_weights(gate_up_proj=gate_up_weight, down_proj=down_weight)

        step_ctx = StepCtx()
        hidden_states = DummyBuffer(num_tokens * hidden_size * 2)
        output = DummyBuffer(num_tokens * hidden_size * 2)

        mlp.encode(
            step_ctx=step_ctx,
            hidden_states=hidden_states,
            output_buffer=output,
            num_tokens=num_tokens,
        )

        # gate, up, down
        assert len(gemm_calls) == 3
        gate_call, up_call, down_call = gemm_calls

        assert gate_call["N"] == intermediate_size
        assert up_call["N"] == intermediate_size
        assert gate_call["transpose_B"] is True
        assert up_call["transpose_B"] is True

        assert isinstance(gate_call["B"], EngineTensor)
        assert isinstance(up_call["B"], EngineTensor)
        assert gate_call["B"].offset == 0
        assert up_call["B"].offset == intermediate_size * hidden_size * 2

        gate_intermediate, up_intermediate, fused_intermediate = step_ctx.allocs
        assert gate_call["C"] is gate_intermediate
        assert up_call["C"] is up_intermediate

        assert len(silu_mul_calls) == 1
        assert silu_mul_calls[0]["gate"] is gate_intermediate
        assert silu_mul_calls[0]["up"] is up_intermediate
        assert silu_mul_calls[0]["output"] is fused_intermediate

        assert down_call["A"] is fused_intermediate
        assert down_call["B"] is down_weight
        assert down_call["C"] is output


class TestPagedAttentionOpDispatch:
    """Regression tests for engine attention dispatch wiring."""

    def test_generic_kernel_sets_new_kv_buffers(self, monkeypatch):
        import vllm_apple.engine.ops.attention as attn_mod

        # Allow testing without Metal bindings by stubbing MTLSize.
        monkeypatch.setattr(attn_mod, "MTLSize", lambda x, y, z: (x, y, z))

        class DummyContext:
            def compile_library(self, *args, **kwargs):
                return None

            def get_pipeline(self, library_name, function_name, function_constants=None):
                return f"pipeline:{function_name}"

            def create_buffer_from_bytes(self, data: bytes):
                return object()

        class Encoder:
            def __init__(self):
                self.calls: list[tuple] = []

            def setComputePipelineState_(self, pipeline):
                self.calls.append(("pipeline", pipeline))

            def setBuffer_offset_atIndex_(self, buf, offset, index):
                self.calls.append(("buffer", index, buf, offset))

            def dispatchThreadgroups_threadsPerThreadgroup_(self, threadgroups, threads_per_group):
                self.calls.append(("dispatch", threadgroups, threads_per_group))

        class StepCtx:
            is_encoding = True

            def __init__(self):
                self.encoder = Encoder()
                self.barriers = 0

            def get_compute_encoder(self):
                return self.encoder

            def memory_barrier(self):
                self.barriers += 1

        ctx = DummyContext()
        attn = attn_mod.PagedAttentionOp(
            context=ctx,
            num_kv_heads=2,
            num_query_heads=4,
            head_size=64,
            block_size=16,
        )
        assert attn._kernel_name == "paged_attention_fused_generic"

        step_ctx = StepCtx()

        attn.encode_decode_fused(
            step_ctx=step_ctx,
            query_buffer="Q",
            new_keys_buffer="NEWK",
            new_values_buffer="NEWV",
            key_buffer="KC",
            value_buffer="VC",
            block_table_buffer="BT",
            seq_lens_buffer="SL",
            output_buffer="OUT",
            num_seqs=3,
            max_seq_len=32,
            max_blocks_per_seq=4,
            query_offset=10,
            new_keys_offset=20,
            new_values_offset=30,
            output_offset=40,
        )

        assert step_ctx.barriers == 0

        buffers = [c for c in step_ctx.encoder.calls if c[0] == "buffer"]
        buffers_by_index = {idx: (buf, off) for _, idx, buf, off in buffers}

        # Generic fused kernel must bind new K/V at buffer(7) and buffer(8).
        assert buffers_by_index[0] == ("Q", 10)
        assert buffers_by_index[5] == ("OUT", 40)
        assert buffers_by_index[7] == ("NEWK", 20)
        assert buffers_by_index[8] == ("NEWV", 30)

        dispatches = [c for c in step_ctx.encoder.calls if c[0] == "dispatch"]
        assert len(dispatches) == 1
        _, threadgroups, threads = dispatches[0]
        assert threadgroups == (3, 4, 1)
        assert threads == (32, 1, 1)

    def test_h128_path_dispatches_kv_write_then_attention(self, monkeypatch):
        import vllm_apple.engine.ops.attention as attn_mod

        monkeypatch.setattr(attn_mod, "MTLSize", lambda x, y, z: (x, y, z))

        class DummyContext:
            def compile_library(self, *args, **kwargs):
                return None

            def get_pipeline(self, library_name, function_name, function_constants=None):
                return f"pipeline:{function_name}"

            def create_buffer_from_bytes(self, data: bytes):
                return object()

        class Encoder:
            def __init__(self):
                self.calls: list[tuple] = []

            def setComputePipelineState_(self, pipeline):
                self.calls.append(("pipeline", pipeline))

            def setBuffer_offset_atIndex_(self, buf, offset, index):
                self.calls.append(("buffer", index, buf, offset))

            def dispatchThreadgroups_threadsPerThreadgroup_(self, threadgroups, threads_per_group):
                self.calls.append(("dispatch", threadgroups, threads_per_group))

        class StepCtx:
            is_encoding = True

            def __init__(self):
                self.encoder = Encoder()
                self.barriers = 0

            def get_compute_encoder(self):
                return self.encoder

            def memory_barrier(self):
                self.barriers += 1

        ctx = DummyContext()
        attn = attn_mod.PagedAttentionOp(
            context=ctx,
            num_kv_heads=2,
            num_query_heads=4,
            head_size=128,
            block_size=16,
        )
        assert attn._kernel_name == "paged_attention_fused_h128"

        step_ctx = StepCtx()

        attn.encode_decode_fused(
            step_ctx=step_ctx,
            query_buffer="Q",
            new_keys_buffer="NEWK",
            new_values_buffer="NEWV",
            key_buffer="KC",
            value_buffer="VC",
            block_table_buffer="BT",
            seq_lens_buffer="SL",
            output_buffer="OUT",
            num_seqs=3,
            max_seq_len=32,
            max_blocks_per_seq=4,
            query_offset=10,
            new_keys_offset=20,
            new_values_offset=30,
            output_offset=40,
        )

        assert step_ctx.barriers == 1

        pipelines = [c for c in step_ctx.encoder.calls if c[0] == "pipeline"]
        assert pipelines[0][1] == "pipeline:kv_write_decode"
        assert pipelines[1][1] == "pipeline:paged_attention_fused_h128"

        dispatches = [c for c in step_ctx.encoder.calls if c[0] == "dispatch"]
        assert len(dispatches) == 2
        _, kv_tg, kv_threads = dispatches[0]
        _, attn_tg, attn_threads = dispatches[1]

        assert kv_tg == (3, 2, 1)
        assert kv_threads == (32, 1, 1)
        assert attn_tg == (3, 4, 1)
        assert attn_threads == (32, 1, 1)

    def test_prefill_kernel_sets_token_to_seq_and_positions(self, monkeypatch):
        import vllm_apple.engine.ops.attention as attn_mod

        monkeypatch.setattr(attn_mod, "MTLSize", lambda x, y, z: (x, y, z))

        class DummyContext:
            def compile_library(self, *args, **kwargs):
                return None

            def get_pipeline(self, library_name, function_name, function_constants=None):
                return f"pipeline:{function_name}"

            def create_buffer_from_bytes(self, data: bytes):
                return object()

        class Encoder:
            def __init__(self):
                self.calls: list[tuple] = []

            def setComputePipelineState_(self, pipeline):
                self.calls.append(("pipeline", pipeline))

            def setBuffer_offset_atIndex_(self, buf, offset, index):
                self.calls.append(("buffer", index, buf, offset))

            def dispatchThreadgroups_threadsPerThreadgroup_(self, threadgroups, threads_per_group):
                self.calls.append(("dispatch", threadgroups, threads_per_group))

        class StepCtx:
            is_encoding = True

            def __init__(self):
                self.encoder = Encoder()
                self.barriers = 0

            def get_compute_encoder(self):
                return self.encoder

            def memory_barrier(self):
                self.barriers += 1

        ctx = DummyContext()
        attn = attn_mod.PagedAttentionOp(
            context=ctx,
            num_kv_heads=2,
            num_query_heads=4,
            head_size=64,
            block_size=16,
        )

        step_ctx = StepCtx()

        attn.encode_prefill(
            step_ctx=step_ctx,
            query_buffer="Q",
            key_buffer="KC",
            value_buffer="VC",
            block_table_buffer="BT",
            token_to_seq_buffer="T2S",
            positions_buffer="POS",
            output_buffer="OUT",
            num_tokens=5,
            num_seqs=2,
            max_seq_len=32,
            max_blocks_per_seq=4,
            query_offset=11,
            output_offset=22,
        )

        assert step_ctx.barriers == 0

        buffers = [c for c in step_ctx.encoder.calls if c[0] == "buffer"]
        buffers_by_index = {idx: (buf, off) for _, idx, buf, off in buffers}

        assert buffers_by_index[0] == ("Q", 11)
        assert buffers_by_index[4] == ("T2S", 0)
        assert buffers_by_index[5] == ("POS", 0)
        assert buffers_by_index[6] == ("OUT", 22)
        assert buffers_by_index[7][1] == 0  # params buffer

        dispatches = [c for c in step_ctx.encoder.calls if c[0] == "dispatch"]
        assert len(dispatches) == 1
        _, threadgroups, threads = dispatches[0]
        assert threadgroups == (5, 4, 1)
        assert threads == (32, 1, 1)


class TestAttentionBackendKVCacheDevice:
    """Regression tests for KV-cache device selection."""

    def test_mps_backend_allocates_kv_cache_on_mps(self):
        from vllm_apple.v1.attention.backends.mps_attn import MPSAttentionBackend

        assert MPSAttentionBackend.get_kv_cache_device() == "mps"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
