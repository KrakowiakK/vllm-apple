# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Boundary Validation for vLLM-Apple Metal Engine v2.0.

This module provides validation utilities for the vLLM↔engine boundary.
It ensures that:
1. No MPS tensors cross the boundary (inputs must be CPU)
2. Outputs are CPU-addressable (returned at step boundary)
3. Input shapes/dtypes match engine expectations

The boundary validation is the first line of defense for the v2.0 invariant:
"No tensors with device.type == 'mps' crossing the vLLM↔engine boundary"

Usage:
    from vllm_apple.engine.boundary import validate_engine_inputs, validate_engine_outputs

    # Before engine execution
    validate_engine_inputs(
        token_ids=token_ids_tensor,
        positions=positions_tensor,
        block_table=block_table_tensor,
    )

    # After engine execution
    validate_engine_outputs(logits=logits_tensor)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from vllm.logger import init_logger
from .strict_mode import is_strict_mode

logger = init_logger(__name__)


# Expected dtypes for engine inputs
EXPECTED_INPUT_DTYPES: Dict[str, Set[torch.dtype]] = {
    "token_ids": {torch.int32, torch.int64, torch.long},
    "positions": {torch.int32, torch.int64, torch.long},
    "block_table": {torch.int32, torch.int64},
    "slot_mapping": {torch.int32, torch.int64, torch.long},
    "seq_lens": {torch.int32, torch.int64},
    "query": {torch.float16, torch.float32},
    "key": {torch.float16, torch.float32},
    "value": {torch.float16, torch.float32},
}

# Expected devices for engine inputs (must be CPU for v2.0)
ALLOWED_INPUT_DEVICES = {"cpu"}


@dataclass
class ValidationError:
    """Represents a validation error for engine boundary."""
    tensor_name: str
    error_type: str
    message: str
    actual_value: Any
    expected_value: Any


class BoundaryValidationResult:
    """Result of boundary validation."""

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def add_error(self, tensor_name: str, error_type: str, message: str,
                  actual: Any = None, expected: Any = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            tensor_name=tensor_name,
            error_type=error_type,
            message=message,
            actual_value=actual,
            expected_value=expected,
        ))

    def add_warning(self, tensor_name: str, error_type: str, message: str,
                    actual: Any = None, expected: Any = None) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(
            tensor_name=tensor_name,
            error_type=error_type,
            message=message,
            actual_value=actual,
            expected_value=expected,
        ))

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def raise_if_invalid(self) -> None:
        """Raise RuntimeError if validation failed."""
        if not self.is_valid:
            error_msgs = [f"  - {e.tensor_name}: {e.message}" for e in self.errors]
            raise RuntimeError(
                f"Engine boundary validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(error_msgs)
            )

    def log_warnings(self) -> None:
        """Log any warnings."""
        for w in self.warnings:
            logger.warning(f"Engine boundary warning for '{w.tensor_name}': {w.message}")


def validate_tensor_device(
    tensor: torch.Tensor,
    name: str,
    result: BoundaryValidationResult,
    strict: bool = True,
) -> None:
    """Validate tensor is on allowed device (CPU for engine mode).

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages
        result: Validation result to append errors to
        strict: If True, device mismatch is an error. If False, it's a warning.
    """
    device_type = tensor.device.type
    if device_type not in ALLOWED_INPUT_DEVICES:
        msg = (
            f"Expected device in {ALLOWED_INPUT_DEVICES}, got '{tensor.device}'. "
            f"Engine mode requires CPU tensors at the vLLM↔engine boundary."
        )
        if strict:
            result.add_error(name, "device", msg, actual=str(tensor.device),
                           expected=list(ALLOWED_INPUT_DEVICES))
        else:
            result.add_warning(name, "device", msg, actual=str(tensor.device),
                             expected=list(ALLOWED_INPUT_DEVICES))


def validate_tensor_dtype(
    tensor: torch.Tensor,
    name: str,
    result: BoundaryValidationResult,
    expected_dtypes: Optional[Set[torch.dtype]] = None,
) -> None:
    """Validate tensor has expected dtype.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor
        result: Validation result to append errors to
        expected_dtypes: Set of allowed dtypes, or None to skip check
    """
    if expected_dtypes is None:
        expected_dtypes = EXPECTED_INPUT_DTYPES.get(name)

    if expected_dtypes is not None and tensor.dtype not in expected_dtypes:
        result.add_warning(
            name, "dtype",
            f"Expected dtype in {[str(d) for d in expected_dtypes]}, got '{tensor.dtype}'",
            actual=str(tensor.dtype),
            expected=[str(d) for d in expected_dtypes],
        )


def validate_tensor_contiguous(
    tensor: torch.Tensor,
    name: str,
    result: BoundaryValidationResult,
) -> None:
    """Validate tensor is contiguous in memory.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor
        result: Validation result to append errors to
    """
    if not tensor.is_contiguous():
        result.add_warning(
            name, "contiguous",
            "Tensor is not contiguous. Consider calling .contiguous() for optimal performance.",
        )


def validate_engine_inputs(
    strict: bool = True,
    **tensors: Optional[torch.Tensor],
) -> BoundaryValidationResult:
    """Validate all input tensors for engine execution.

    This validates that:
    1. All tensors are on CPU (not MPS)
    2. Tensors have expected dtypes
    3. Tensors are contiguous

    Args:
        strict: If True, raise on validation errors. If False, return result.
        **tensors: Named tensors to validate (e.g., token_ids=tensor)

    Returns:
        BoundaryValidationResult with errors and warnings

    Raises:
        RuntimeError: If strict=True and validation fails
    """
    result = BoundaryValidationResult()

    # Only enforce strict device checking if strict mode is enabled
    strict_device = is_strict_mode()

    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            result.add_warning(name, "type", f"Expected torch.Tensor, got {type(tensor)}")
            continue

        validate_tensor_device(tensor, name, result, strict=strict_device)
        validate_tensor_dtype(tensor, name, result)
        validate_tensor_contiguous(tensor, name, result)

    if strict and not result.is_valid:
        result.raise_if_invalid()

    result.log_warnings()
    return result


def validate_engine_outputs(
    strict: bool = True,
    **tensors: Optional[torch.Tensor],
) -> BoundaryValidationResult:
    """Validate all output tensors from engine execution.

    Outputs must be CPU-addressable at step boundary.

    Args:
        strict: If True, raise on validation errors.
        **tensors: Named tensors to validate (e.g., logits=tensor)

    Returns:
        BoundaryValidationResult with errors and warnings

    Raises:
        RuntimeError: If strict=True and validation fails
    """
    result = BoundaryValidationResult()

    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            result.add_warning(name, "type", f"Expected torch.Tensor, got {type(tensor)}")
            continue

        # Outputs should be CPU-addressable
        # For MTLBuffer with storageModeShared, the memory is CPU-addressable
        # even though it's not a PyTorch CPU tensor per se
        # We check that it's NOT on MPS (which would require sync to read)
        if tensor.device.type == "mps":
            result.add_error(
                name, "device",
                f"Output tensor on MPS device. "
                f"Engine outputs must be CPU-addressable at step boundary. "
                f"Got device: {tensor.device}",
                actual=str(tensor.device),
                expected="cpu",
            )

    if strict and not result.is_valid:
        result.raise_if_invalid()

    result.log_warnings()
    return result


def ensure_cpu_tensor(
    tensor: Optional[torch.Tensor],
    name: str = "tensor",
    copy: bool = False,
    strict_mode: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    """Ensure tensor is on CPU.

    This is a utility for preparing inputs at the engine boundary.

    Args:
        tensor: Input tensor (may be None, CPU, or MPS)
        name: Name for logging
        copy: If True, always make a copy even if already on CPU
        strict_mode: If True, raise on MPS tensor instead of converting.
                     If None, uses is_strict_mode() to determine.

    Returns:
        CPU tensor, or None if input was None

    Raises:
        RuntimeError: If strict_mode and tensor is on MPS device.
                      MPS tensors should be pre-converted BEFORE calling
                      engine boundary functions.
    """
    if tensor is None:
        return None

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for '{name}', got {type(tensor)}")

    if tensor.device.type == "cpu":
        if copy:
            return tensor.clone()
        return tensor

    # Check strict mode
    if strict_mode is None:
        strict_mode = is_strict_mode()

    if strict_mode:
        # In strict mode: fail-fast, don't silently convert
        raise RuntimeError(
            f"MPS tensor '{name}' passed to engine boundary in strict mode. "
            f"Device: {tensor.device}. "
            f"MPS→CPU conversion is not allowed in strict mode. "
            f"Tensors must be on CPU BEFORE reaching engine boundary. "
            f"Convert tensors in vLLM code, not in engine code."
        )

    # Non-strict mode: convert with warning
    logger.warning(
        f"Converting '{name}' from {tensor.device} to CPU at engine boundary. "
        f"This should be done in vLLM code, not here."
    )
    return tensor.cpu()


def prepare_inputs_for_engine(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare all inputs for engine execution.

    Converts tensors to CPU if needed and validates the boundary contract.
    This should be called at the step boundary BEFORE entering engine hot path.

    Args:
        token_ids: Token IDs tensor
        positions: Position IDs tensor
        block_table: Block table for paged attention
        slot_mapping: Slot mapping for KV cache
        seq_lens: Sequence lengths

    Returns:
        Tuple of CPU tensors ready for engine execution

    Raises:
        RuntimeError: If validation fails in strict mode
    """
    # Convert to CPU if needed (must be done before hot path)
    token_ids_cpu = ensure_cpu_tensor(token_ids, "token_ids")
    positions_cpu = ensure_cpu_tensor(positions, "positions")
    block_table_cpu = ensure_cpu_tensor(block_table, "block_table")
    slot_mapping_cpu = ensure_cpu_tensor(slot_mapping, "slot_mapping")
    seq_lens_cpu = ensure_cpu_tensor(seq_lens, "seq_lens")

    # Validate
    validate_engine_inputs(
        strict=True,
        token_ids=token_ids_cpu,
        positions=positions_cpu,
        block_table=block_table_cpu,
        slot_mapping=slot_mapping_cpu,
        seq_lens=seq_lens_cpu,
    )

    return token_ids_cpu, positions_cpu, block_table_cpu, slot_mapping_cpu, seq_lens_cpu
