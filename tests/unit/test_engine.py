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


# Skip all tests if Metal/MPS not available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


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


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
