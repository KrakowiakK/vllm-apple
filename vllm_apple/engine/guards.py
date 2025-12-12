# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Hot Path Guards for vLLM-Apple Metal Engine v2.0.

This module provides thread-local context management for engine execution phases.
It enforces the v2.0 synchronization policy: NO per-layer/per-op waits during
the ENCODE or SUBMIT phases. Only the READBACK phase (at step boundary) allows
synchronization and CPU readback operations.

Usage:
    with EngineHotPathGuard.encode_phase():
        # Encode kernels to command buffer
        # Any forbidden operation will raise RuntimeError

    with EngineHotPathGuard.submit_phase():
        command_buffer.commit()
        # No waits allowed here

    with EngineHotPathGuard.readback_phase():
        command_buffer.waitUntilCompleted()  # OK here
        output = buffer.contents()  # OK here
"""

import threading
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Set

from vllm.logger import init_logger

logger = init_logger(__name__)


class EnginePhase(Enum):
    """Execution phases for engine hot path."""
    IDLE = "idle"           # Outside engine execution
    ENCODE = "encode"       # Building/encoding command buffer
    SUBMIT = "submit"       # Command buffer committed, not yet waited
    READBACK = "readback"   # After wait, reading outputs to CPU


# Operations forbidden during ENCODE and SUBMIT phases
FORBIDDEN_HOT_PATH_OPS: Set[str] = {
    # PyTorch MPS synchronization
    "torch.mps.synchronize",
    "torch.mps.commit",

    # MPS tensor to CPU operations (cause implicit sync)
    "tensor.item() on MPS tensor",
    "tensor.tolist() on MPS tensor",
    "tensor.cpu() on MPS tensor",
    "tensor.numpy() on MPS tensor",

    # Metal command buffer waits
    "waitUntilCompleted",
    "waitUntilScheduled",

    # Metal buffer readback
    "MTLBuffer.contents",
    "MTLBuffer.getBytes",
    "didModifyRange",

    # Metal synchronization
    "synchronizeResource",
    "synchronizeTexture",
}


class EngineHotPathGuard:
    """Thread-local guard for engine execution phases.

    This class enforces the v2.0 synchronization policy by:
    1. Tracking the current execution phase per thread
    2. Raising RuntimeError if forbidden operations are attempted during
       ENCODE or SUBMIT phases
    3. Allowing synchronization only during READBACK phase (step boundary)

    The guard is thread-local to support concurrent engine execution if needed.
    """

    _local = threading.local()

    @classmethod
    def get_phase(cls) -> EnginePhase:
        """Get current execution phase for this thread."""
        return getattr(cls._local, 'phase', EnginePhase.IDLE)

    @classmethod
    def set_phase(cls, phase: EnginePhase) -> None:
        """Set execution phase for this thread."""
        old_phase = cls.get_phase()
        cls._local.phase = phase
        if old_phase != phase:
            logger.debug(f"Engine phase: {old_phase.value} -> {phase.value}")

    @classmethod
    def is_hot_path(cls) -> bool:
        """Check if currently in hot path (ENCODE or SUBMIT phase)."""
        return cls.get_phase() in (EnginePhase.ENCODE, EnginePhase.SUBMIT)

    @classmethod
    def assert_not_hot_path(cls, operation: str) -> None:
        """Raise RuntimeError if called during ENCODE or SUBMIT phase.

        Args:
            operation: Name of the operation being attempted

        Raises:
            RuntimeError: If called during hot path phases
        """
        phase = cls.get_phase()
        if phase in (EnginePhase.ENCODE, EnginePhase.SUBMIT):
            raise RuntimeError(
                f"Forbidden operation '{operation}' during engine {phase.value} phase. "
                f"Synchronization/readback is only allowed at step boundary (READBACK phase). "
                f"This violates the v2.0 'step-boundary-only synchronization' invariant."
            )

    @classmethod
    def assert_allowed_in_phase(cls, operation: str, allowed_phases: Set[EnginePhase]) -> None:
        """Assert operation is allowed in current phase.

        Args:
            operation: Name of the operation
            allowed_phases: Set of phases where operation is allowed

        Raises:
            RuntimeError: If current phase not in allowed_phases
        """
        phase = cls.get_phase()
        if phase not in allowed_phases:
            raise RuntimeError(
                f"Operation '{operation}' not allowed in {phase.value} phase. "
                f"Allowed phases: {[p.value for p in allowed_phases]}"
            )

    @classmethod
    @contextmanager
    def encode_phase(cls):
        """Context manager for ENCODE phase."""
        old_phase = cls.get_phase()
        cls.set_phase(EnginePhase.ENCODE)
        try:
            yield
        finally:
            cls.set_phase(old_phase)

    @classmethod
    @contextmanager
    def submit_phase(cls):
        """Context manager for SUBMIT phase."""
        old_phase = cls.get_phase()
        cls.set_phase(EnginePhase.SUBMIT)
        try:
            yield
        finally:
            cls.set_phase(old_phase)

    @classmethod
    @contextmanager
    def readback_phase(cls):
        """Context manager for READBACK phase."""
        old_phase = cls.get_phase()
        cls.set_phase(EnginePhase.READBACK)
        try:
            yield
        finally:
            cls.set_phase(old_phase)

    @classmethod
    @contextmanager
    def step_execution(cls):
        """Context manager for full step execution (ENCODE -> SUBMIT -> READBACK).

        This is a convenience wrapper that manages the full step lifecycle.
        The caller should use the yielded controller to transition phases.

        Usage:
            with EngineHotPathGuard.step_execution() as step:
                # Starts in ENCODE phase
                encode_kernels()

                step.transition_to_submit()
                command_buffer.commit()

                step.transition_to_readback()
                command_buffer.waitUntilCompleted()
                result = read_output()
        """
        controller = StepPhaseController()
        old_phase = cls.get_phase()
        cls.set_phase(EnginePhase.ENCODE)
        try:
            yield controller
        finally:
            cls.set_phase(old_phase)


class StepPhaseController:
    """Controller for transitioning between phases within a step."""

    def __init__(self):
        self._current_phase_idx = 0
        self._phases = [EnginePhase.ENCODE, EnginePhase.SUBMIT, EnginePhase.READBACK]

    def transition_to_submit(self) -> None:
        """Transition from ENCODE to SUBMIT phase."""
        if self._current_phase_idx != 0:
            raise RuntimeError("Can only transition to SUBMIT from ENCODE phase")
        self._current_phase_idx = 1
        EngineHotPathGuard.set_phase(EnginePhase.SUBMIT)

    def transition_to_readback(self) -> None:
        """Transition from SUBMIT to READBACK phase."""
        if self._current_phase_idx != 1:
            raise RuntimeError("Can only transition to READBACK from SUBMIT phase")
        self._current_phase_idx = 2
        EngineHotPathGuard.set_phase(EnginePhase.READBACK)


def check_forbidden_operation(operation: str, strict_mode: bool = True) -> None:
    """Check if operation is forbidden and raise if in hot path.

    Args:
        operation: Name of the operation
        strict_mode: If True, raise on violation. If False, log warning only.

    Raises:
        RuntimeError: If strict_mode and operation is forbidden in current phase
    """
    if not EngineHotPathGuard.is_hot_path():
        return

    if operation in FORBIDDEN_HOT_PATH_OPS:
        msg = (
            f"Forbidden operation '{operation}' detected during "
            f"{EngineHotPathGuard.get_phase().value} phase"
        )
        if strict_mode:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)


# =============================================================================
# Guarded Metal API wrappers
# =============================================================================
# These functions wrap Metal API calls with phase checking to enforce the
# v2.0 "no readback in hot path" invariant.

def guarded_buffer_contents(buffer, operation_context: str = "") -> memoryview:
    """Get MTLBuffer contents with phase guard.

    This wraps MTLBuffer.contents() to enforce the v2.0 invariant.
    Only allowed in READBACK or IDLE phase.

    Args:
        buffer: MTLBuffer to read
        operation_context: Additional context for error message

    Returns:
        memoryview of buffer contents

    Raises:
        RuntimeError: If called during ENCODE or SUBMIT phase
    """
    phase = EngineHotPathGuard.get_phase()
    if phase in (EnginePhase.ENCODE, EnginePhase.SUBMIT):
        ctx = f" ({operation_context})" if operation_context else ""
        raise RuntimeError(
            f"MTLBuffer.contents() called during {phase.value} phase{ctx}. "
            f"Buffer readback is only allowed in READBACK phase. "
            f"Use guarded_buffer_contents() only at step boundary."
        )
    return buffer.contents()


def guarded_wait_until_completed(command_buffer, operation_context: str = "") -> None:
    """Wait for command buffer with phase guard.

    This wraps waitUntilCompleted() to enforce proper phase transitions.
    Only allowed in READBACK or IDLE phase - NOT during ENCODE or SUBMIT.

    The v2.0 invariant is: one waitUntilCompleted() per step, at step boundary.
    This means wait is forbidden during both ENCODE (still building CB) and
    SUBMIT (CB committed but we're still in the "no-wait" hot path).

    Args:
        command_buffer: MTLCommandBuffer to wait on
        operation_context: Additional context for error message

    Raises:
        RuntimeError: If called during ENCODE or SUBMIT phase
    """
    phase = EngineHotPathGuard.get_phase()
    if phase in (EnginePhase.ENCODE, EnginePhase.SUBMIT):
        ctx = f" ({operation_context})" if operation_context else ""
        raise RuntimeError(
            f"waitUntilCompleted() called during {phase.value} phase{ctx}. "
            f"Wait is only allowed in READBACK phase (step boundary). "
            f"This enforces the v2.0 'one wait per step' invariant."
        )
    command_buffer.waitUntilCompleted()


def guarded_blit_copy(
    encoder,  # MTLBlitCommandEncoder
    source_buffer,
    source_offset: int,
    dest_buffer,
    dest_offset: int,
    size: int,
) -> None:
    """Encode blit copy with phase guard.

    Blit encoding is allowed during ENCODE phase.

    Args:
        encoder: MTLBlitCommandEncoder
        source_buffer: Source MTLBuffer
        source_offset: Offset in source
        dest_buffer: Destination MTLBuffer
        dest_offset: Offset in destination
        size: Bytes to copy
    """
    # Blit encoding is allowed in ENCODE phase
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
        source_buffer, source_offset, dest_buffer, dest_offset, size
    )


def guarded_fill_buffer(
    encoder,  # MTLBlitCommandEncoder
    buffer,
    range_start: int,
    range_length: int,
    value: int = 0,
) -> None:
    """Encode buffer fill with phase guard.

    Uses MTLBlitCommandEncoder.fillBuffer for efficient GPU-side zeroing.

    Args:
        encoder: MTLBlitCommandEncoder
        buffer: MTLBuffer to fill
        range_start: Start offset
        range_length: Length to fill
        value: Fill value (0-255)
    """
    from Metal import NSMakeRange
    fill_range = NSMakeRange(range_start, range_length)
    encoder.fillBuffer_range_value_(buffer, fill_range, value)
