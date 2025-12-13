# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for strict no-MPS mode enforcement.

These tests verify that the v2.0 engine invariant "no PyTorch-MPS in hot path"
is properly enforced when VLLM_METAL_STRICT_NO_MPS=1.

Tests cover:
- EngineHotPathGuard phase transitions
- Strict mode monkey-patching
- Boundary validation
- Forbidden operation detection
"""

import os
import pytest
import torch

from vllm_apple.engine.guards import (
    EngineHotPathGuard,
    EnginePhase,
    StepPhaseController,
    check_forbidden_operation,
    FORBIDDEN_HOT_PATH_OPS,
)
from vllm_apple.engine.strict_mode import (
    enable_strict_mode,
    disable_strict_mode,
    is_strict_mode,
    get_strict_mode_enabled,
    StrictModeContext,
    assert_no_mps_tensors,
    assert_cpu_tensors,
)
from vllm_apple.engine.boundary import (
    validate_engine_inputs,
    validate_engine_outputs,
    ensure_cpu_tensor,
    BoundaryValidationResult,
    ValidationError,
)


class TestEngineHotPathGuard:
    """Tests for EngineHotPathGuard phase management."""

    def test_initial_phase_is_idle(self):
        """Phase should be IDLE when not in engine context."""
        # Reset to clean state
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE

    def test_encode_phase_context_manager(self):
        """Test encode_phase context manager sets and restores phase."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

        with EngineHotPathGuard.encode_phase():
            assert EngineHotPathGuard.get_phase() == EnginePhase.ENCODE
            assert EngineHotPathGuard.is_hot_path()

        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE

    def test_submit_phase_context_manager(self):
        """Test submit_phase context manager sets and restores phase."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

        with EngineHotPathGuard.submit_phase():
            assert EngineHotPathGuard.get_phase() == EnginePhase.SUBMIT
            assert EngineHotPathGuard.is_hot_path()

        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE

    def test_readback_phase_context_manager(self):
        """Test readback_phase context manager sets and restores phase."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

        with EngineHotPathGuard.readback_phase():
            assert EngineHotPathGuard.get_phase() == EnginePhase.READBACK
            assert not EngineHotPathGuard.is_hot_path()

        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE

    def test_is_hot_path_encode_submit_only(self):
        """Only ENCODE and SUBMIT phases should be considered hot path."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        assert not EngineHotPathGuard.is_hot_path()

        EngineHotPathGuard.set_phase(EnginePhase.ENCODE)
        assert EngineHotPathGuard.is_hot_path()

        EngineHotPathGuard.set_phase(EnginePhase.SUBMIT)
        assert EngineHotPathGuard.is_hot_path()

        EngineHotPathGuard.set_phase(EnginePhase.READBACK)
        assert not EngineHotPathGuard.is_hot_path()

        # Cleanup
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

    def test_assert_not_hot_path_raises_during_encode(self):
        """assert_not_hot_path should raise during ENCODE phase."""
        with EngineHotPathGuard.encode_phase():
            with pytest.raises(RuntimeError, match="Forbidden operation"):
                EngineHotPathGuard.assert_not_hot_path("test_operation")

    def test_assert_not_hot_path_raises_during_submit(self):
        """assert_not_hot_path should raise during SUBMIT phase."""
        with EngineHotPathGuard.submit_phase():
            with pytest.raises(RuntimeError, match="Forbidden operation"):
                EngineHotPathGuard.assert_not_hot_path("test_operation")

    def test_assert_not_hot_path_allows_during_readback(self):
        """assert_not_hot_path should NOT raise during READBACK phase."""
        with EngineHotPathGuard.readback_phase():
            # Should not raise
            EngineHotPathGuard.assert_not_hot_path("test_operation")

    def test_assert_not_hot_path_allows_during_idle(self):
        """assert_not_hot_path should NOT raise during IDLE phase."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        # Should not raise
        EngineHotPathGuard.assert_not_hot_path("test_operation")

    def test_nested_phases_restore_correctly(self):
        """Nested phase contexts should restore to parent phase."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

        with EngineHotPathGuard.encode_phase():
            assert EngineHotPathGuard.get_phase() == EnginePhase.ENCODE

            with EngineHotPathGuard.submit_phase():
                assert EngineHotPathGuard.get_phase() == EnginePhase.SUBMIT

            assert EngineHotPathGuard.get_phase() == EnginePhase.ENCODE

        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE


class TestStepPhaseController:
    """Tests for StepPhaseController transitions."""

    def test_step_execution_starts_in_encode(self):
        """step_execution context should start in ENCODE phase."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

        with EngineHotPathGuard.step_execution() as step:
            assert EngineHotPathGuard.get_phase() == EnginePhase.ENCODE

    def test_transition_to_submit(self):
        """transition_to_submit should change phase to SUBMIT."""
        with EngineHotPathGuard.step_execution() as step:
            step.transition_to_submit()
            assert EngineHotPathGuard.get_phase() == EnginePhase.SUBMIT

    def test_transition_to_readback(self):
        """transition_to_readback should change phase to READBACK."""
        with EngineHotPathGuard.step_execution() as step:
            step.transition_to_submit()
            step.transition_to_readback()
            assert EngineHotPathGuard.get_phase() == EnginePhase.READBACK

    def test_cannot_skip_submit_phase(self):
        """Cannot transition directly to READBACK from ENCODE."""
        with EngineHotPathGuard.step_execution() as step:
            with pytest.raises(RuntimeError, match="SUBMIT"):
                step.transition_to_readback()

    def test_cannot_go_back_to_submit(self):
        """Cannot transition back to SUBMIT after READBACK."""
        with EngineHotPathGuard.step_execution() as step:
            step.transition_to_submit()
            step.transition_to_readback()
            with pytest.raises(RuntimeError, match="ENCODE"):
                step.transition_to_submit()


class TestForbiddenOperations:
    """Tests for forbidden operation detection."""

    def test_check_forbidden_operation_outside_hot_path(self):
        """check_forbidden_operation should not raise outside hot path."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        # Should not raise
        check_forbidden_operation("torch.mps.synchronize")

    def test_check_forbidden_operation_in_hot_path_strict(self):
        """check_forbidden_operation should raise in hot path with strict mode."""
        with EngineHotPathGuard.encode_phase():
            with pytest.raises(RuntimeError):
                check_forbidden_operation("torch.mps.synchronize", strict_mode=True)

    def test_check_forbidden_operation_in_hot_path_non_strict(self):
        """check_forbidden_operation should not raise with strict_mode=False."""
        with EngineHotPathGuard.encode_phase():
            # Should not raise, just log warning
            check_forbidden_operation("torch.mps.synchronize", strict_mode=False)

    def test_forbidden_ops_list_contains_mps_sync(self):
        """FORBIDDEN_HOT_PATH_OPS should contain MPS sync operations."""
        assert "torch.mps.synchronize" in FORBIDDEN_HOT_PATH_OPS
        assert "torch.mps.commit" in FORBIDDEN_HOT_PATH_OPS
        assert "waitUntilCompleted" in FORBIDDEN_HOT_PATH_OPS


class TestStrictMode:
    """Tests for strict mode enable/disable."""

    def setup_method(self):
        """Ensure strict mode is disabled before each test."""
        disable_strict_mode()
        if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
            del os.environ["VLLM_METAL_STRICT_NO_MPS"]

    def teardown_method(self):
        """Clean up after each test."""
        disable_strict_mode()
        if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
            del os.environ["VLLM_METAL_STRICT_NO_MPS"]
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

    def test_is_strict_mode_default_false(self):
        """is_strict_mode should return False by default."""
        assert not is_strict_mode()

    def test_is_strict_mode_with_env_var(self):
        """is_strict_mode should return True when env var is set."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        assert is_strict_mode()

    def test_enable_strict_mode_without_env_var(self):
        """enable_strict_mode should do nothing without env var."""
        enable_strict_mode()
        assert not get_strict_mode_enabled()

    def test_enable_strict_mode_with_env_var(self):
        """enable_strict_mode should apply patches when env var is set."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()
        assert get_strict_mode_enabled()

    def test_disable_strict_mode(self):
        """disable_strict_mode should restore original functions."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()
        assert get_strict_mode_enabled()

        disable_strict_mode()
        assert not get_strict_mode_enabled()

    def test_strict_mode_context_enable(self):
        """StrictModeContext should enable strict mode temporarily."""
        assert not get_strict_mode_enabled()

        with StrictModeContext(enabled=True):
            assert get_strict_mode_enabled()

        assert not get_strict_mode_enabled()

    def test_strict_mode_context_disable(self):
        """StrictModeContext should disable strict mode temporarily."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()
        assert get_strict_mode_enabled()

        with StrictModeContext(enabled=False):
            assert not get_strict_mode_enabled()

        assert get_strict_mode_enabled()


class TestStrictModePatches:
    """Tests for strict mode monkey-patches on PyTorch MPS APIs."""

    def setup_method(self):
        """Enable strict mode for patch tests."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()

    def teardown_method(self):
        """Disable strict mode after tests."""
        disable_strict_mode()
        if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
            del os.environ["VLLM_METAL_STRICT_NO_MPS"]
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

    @pytest.mark.skipif(
        not hasattr(torch, 'mps') or not hasattr(torch.mps, 'synchronize'),
        reason="torch.mps.synchronize not available"
    )
    def test_mps_synchronize_raises_in_hot_path(self):
        """torch.mps.synchronize should raise during hot path."""
        with EngineHotPathGuard.encode_phase():
            with pytest.raises(RuntimeError, match="torch.mps.synchronize"):
                torch.mps.synchronize()

    @pytest.mark.skipif(
        not hasattr(torch, 'mps') or not hasattr(torch.mps, 'synchronize'),
        reason="torch.mps.synchronize not available"
    )
    def test_mps_synchronize_allowed_outside_hot_path(self):
        """torch.mps.synchronize should work outside hot path."""
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        # Should not raise (though may fail if MPS not available)
        try:
            torch.mps.synchronize()
        except RuntimeError as e:
            # MPS may not be available, but should not be our error
            assert "Forbidden operation" not in str(e)


class TestMPSTensorAssertions:
    """Tests for MPS tensor assertion utilities."""

    def setup_method(self):
        """Enable strict mode for assertion tests."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()

    def teardown_method(self):
        """Disable strict mode after tests."""
        disable_strict_mode()
        if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
            del os.environ["VLLM_METAL_STRICT_NO_MPS"]

    def test_assert_no_mps_tensors_with_cpu_tensor(self):
        """assert_no_mps_tensors should pass with CPU tensors."""
        cpu_tensor = torch.randn(10)
        # Should not raise
        assert_no_mps_tensors(cpu_tensor, context="test")

    def test_assert_no_mps_tensors_with_none(self):
        """assert_no_mps_tensors should handle None gracefully."""
        assert_no_mps_tensors(None, context="test")

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_assert_no_mps_tensors_with_mps_tensor(self):
        """assert_no_mps_tensors should raise with MPS tensor."""
        mps_tensor = torch.randn(10, device="mps")
        with pytest.raises(RuntimeError, match="MPS tensor detected"):
            assert_no_mps_tensors(mps_tensor, context="test")

    def test_assert_cpu_tensors_with_cpu_tensor(self):
        """assert_cpu_tensors should pass with CPU tensors."""
        cpu_tensor = torch.randn(10)
        # Should not raise
        assert_cpu_tensors(query=cpu_tensor)

    def test_assert_cpu_tensors_with_none(self):
        """assert_cpu_tensors should handle None gracefully."""
        assert_cpu_tensors(query=None)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_assert_cpu_tensors_with_mps_tensor(self):
        """assert_cpu_tensors should raise with MPS tensor."""
        mps_tensor = torch.randn(10, device="mps")
        with pytest.raises(RuntimeError, match="Non-CPU tensor"):
            assert_cpu_tensors(query=mps_tensor)


class TestBoundaryValidation:
    """Tests for engine boundary validation."""

    def test_validate_engine_inputs_cpu_tensors(self):
        """validate_engine_inputs should pass for CPU tensors."""
        token_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        positions = torch.tensor([0, 1, 2], dtype=torch.int64)

        result = validate_engine_inputs(
            strict=False,
            token_ids=token_ids,
            positions=positions,
        )
        assert result.is_valid

    def test_validate_engine_inputs_none_values(self):
        """validate_engine_inputs should handle None values."""
        result = validate_engine_inputs(
            strict=False,
            token_ids=None,
            positions=None,
        )
        assert result.is_valid

    def test_validate_engine_inputs_non_tensor(self):
        """validate_engine_inputs should warn for non-tensor values."""
        result = validate_engine_inputs(
            strict=False,
            token_ids=[1, 2, 3],  # List, not tensor
        )
        # Should have warning
        assert len(result.warnings) > 0

    def test_validate_engine_inputs_non_contiguous(self):
        """validate_engine_inputs should warn for non-contiguous tensors."""
        tensor = torch.randn(10, 10)[:, ::2]  # Non-contiguous slice
        assert not tensor.is_contiguous()

        result = validate_engine_inputs(
            strict=False,
            data=tensor,
        )
        # Should have warning about contiguity
        assert any("contiguous" in w.message.lower() for w in result.warnings)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_validate_engine_inputs_mps_tensor_strict(self):
        """validate_engine_inputs should error for MPS tensors in strict mode."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        try:
            mps_tensor = torch.randn(10, device="mps")
            result = validate_engine_inputs(
                strict=False,  # Don't raise, just collect errors
                token_ids=mps_tensor,
            )
            # Should have error about device
            assert not result.is_valid or len(result.warnings) > 0
        finally:
            if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
                del os.environ["VLLM_METAL_STRICT_NO_MPS"]

    def test_validate_engine_outputs_cpu_tensor(self):
        """validate_engine_outputs should pass for CPU tensors."""
        logits = torch.randn(4, 1000)  # (num_seqs, vocab_size)

        result = validate_engine_outputs(
            strict=False,
            logits=logits,
        )
        assert result.is_valid

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_validate_engine_outputs_mps_tensor_error(self):
        """validate_engine_outputs should error for MPS tensors."""
        mps_tensor = torch.randn(10, device="mps")

        result = validate_engine_outputs(
            strict=False,
            logits=mps_tensor,
        )
        assert not result.is_valid
        assert any("MPS" in e.message for e in result.errors)


class TestEnsureCPUTensor:
    """Tests for ensure_cpu_tensor utility."""

    def test_ensure_cpu_tensor_none(self):
        """ensure_cpu_tensor should return None for None input."""
        result = ensure_cpu_tensor(None)
        assert result is None

    def test_ensure_cpu_tensor_already_cpu(self):
        """ensure_cpu_tensor should return same tensor if already CPU."""
        cpu_tensor = torch.randn(10)
        result = ensure_cpu_tensor(cpu_tensor)
        assert result is cpu_tensor  # Same object

    def test_ensure_cpu_tensor_copy_cpu(self):
        """ensure_cpu_tensor with copy=True should clone CPU tensor."""
        cpu_tensor = torch.randn(10)
        result = ensure_cpu_tensor(cpu_tensor, copy=True)
        assert result is not cpu_tensor
        assert torch.equal(result, cpu_tensor)

    def test_ensure_cpu_tensor_invalid_type(self):
        """ensure_cpu_tensor should raise for non-tensor input."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            ensure_cpu_tensor([1, 2, 3], name="test_tensor")

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_ensure_cpu_tensor_converts_mps(self):
        """ensure_cpu_tensor should convert MPS tensor to CPU."""
        mps_tensor = torch.randn(10, device="mps")
        result = ensure_cpu_tensor(mps_tensor)
        assert result.device.type == "cpu"
        # Values should match
        assert torch.allclose(result, mps_tensor.cpu())


class TestBoundaryValidationResult:
    """Tests for BoundaryValidationResult class."""

    def test_empty_result_is_valid(self):
        """Empty result should be valid."""
        result = BoundaryValidationResult()
        assert result.is_valid

    def test_result_with_error_is_invalid(self):
        """Result with error should be invalid."""
        result = BoundaryValidationResult()
        result.add_error("test", "type", "test error")
        assert not result.is_valid

    def test_result_with_warning_is_valid(self):
        """Result with warning only should still be valid."""
        result = BoundaryValidationResult()
        result.add_warning("test", "type", "test warning")
        assert result.is_valid

    def test_raise_if_invalid(self):
        """raise_if_invalid should raise RuntimeError on errors."""
        result = BoundaryValidationResult()
        result.add_error("test_tensor", "device", "wrong device")

        with pytest.raises(RuntimeError, match="validation failed"):
            result.raise_if_invalid()

    def test_raise_if_invalid_noop_when_valid(self):
        """raise_if_invalid should do nothing when valid."""
        result = BoundaryValidationResult()
        # Should not raise
        result.raise_if_invalid()


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_validation_error_creation(self):
        """ValidationError should be created with all fields."""
        error = ValidationError(
            tensor_name="query",
            error_type="device",
            message="Expected CPU, got MPS",
            actual_value="mps:0",
            expected_value="cpu",
        )
        assert error.tensor_name == "query"
        assert error.error_type == "device"
        assert "CPU" in error.message
        assert error.actual_value == "mps:0"
        assert error.expected_value == "cpu"


class TestStrictModeActivation:
    """Tests for strict mode activation flow (simulating worker startup).

    These tests verify that strict mode is properly activated when
    enable_strict_mode() is called, matching the behavior expected
    when AppleWorker.init_device() runs with VLLM_METAL_STRICT_NO_MPS=1.
    """

    def setup_method(self):
        """Ensure clean state before each test."""
        disable_strict_mode()
        if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
            del os.environ["VLLM_METAL_STRICT_NO_MPS"]
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

    def teardown_method(self):
        """Clean up after each test."""
        disable_strict_mode()
        if "VLLM_METAL_STRICT_NO_MPS" in os.environ:
            del os.environ["VLLM_METAL_STRICT_NO_MPS"]
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)

    def test_strict_mode_activation_idempotent(self):
        """enable_strict_mode() should be safe to call multiple times."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"

        # Call multiple times (as might happen in tests or reinitialization)
        enable_strict_mode()
        assert get_strict_mode_enabled()

        enable_strict_mode()
        assert get_strict_mode_enabled()

        enable_strict_mode()
        assert get_strict_mode_enabled()

    def test_strict_mode_activation_sets_patches(self):
        """enable_strict_mode() should apply monkey-patches when env var set."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"

        # Before activation - get_strict_mode_enabled should be False
        assert not get_strict_mode_enabled()

        # Activate
        enable_strict_mode()

        # After activation - patches should be in place
        assert get_strict_mode_enabled()

    @pytest.mark.skipif(
        not hasattr(torch, 'mps') or not hasattr(torch.mps, 'synchronize'),
        reason="torch.mps.synchronize not available"
    )
    def test_patches_functional_after_activation(self):
        """Patches should be functional after enable_strict_mode()."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()

        # In hot path, torch.mps.synchronize should raise
        with EngineHotPathGuard.encode_phase():
            with pytest.raises(RuntimeError, match="torch.mps.synchronize"):
                torch.mps.synchronize()

        # Outside hot path, should not raise our error
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        try:
            torch.mps.synchronize()
        except RuntimeError as e:
            # MPS may not be available, but should not be our forbidden op error
            assert "Forbidden operation" not in str(e)

    def test_strict_mode_deactivation_removes_patches(self):
        """disable_strict_mode() should restore original behavior."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()
        assert get_strict_mode_enabled()

        disable_strict_mode()
        assert not get_strict_mode_enabled()

    def test_env_var_checked_at_enable_time(self):
        """enable_strict_mode() should check env var at call time."""
        # First call without env var - should not enable
        enable_strict_mode()
        assert not get_strict_mode_enabled()

        # Now set env var and call again
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()
        assert get_strict_mode_enabled()

    def test_strict_mode_with_phase_transitions(self):
        """Strict mode should work correctly through phase transitions."""
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = "1"
        enable_strict_mode()

        # Simulate a full step execution
        with EngineHotPathGuard.step_execution() as step:
            # ENCODE phase - should be hot path
            assert EngineHotPathGuard.is_hot_path()

            # Transition to SUBMIT - still hot path
            step.transition_to_submit()
            assert EngineHotPathGuard.is_hot_path()

            # Transition to READBACK - no longer hot path
            step.transition_to_readback()
            assert not EngineHotPathGuard.is_hot_path()

        # After step - should be IDLE
        assert EngineHotPathGuard.get_phase() == EnginePhase.IDLE
