# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Step Context for vLLM-Apple Metal Engine v2.0.

This module provides per-step resource management for engine execution.
Each scheduler step gets an EngineStepContext that manages:
- Command buffer for the step
- Compute encoder
- Scratch buffer allocations
- Step-level profiling markers

The step context enforces the three-phase execution model:
1. ENCODE: Build command buffer (no waits)
2. SUBMIT: Commit command buffer (no waits)
3. READBACK: Wait and read outputs

Usage:
    with EngineStepContext(context, step_desc) as step:
        # ENCODE phase - encode all operations
        step.encode_operation(...)

        # SUBMIT phase - commit command buffer
        step.submit()

        # READBACK phase - wait and read outputs
        results = step.readback()
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import time

from vllm.logger import init_logger
from .guards import EngineHotPathGuard, EnginePhase
from .profiling import get_profiler, PROFILING_ENABLED

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLBarrierScopeBuffers
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    MTLBarrierScopeBuffers = None


@dataclass
class ScratchAllocation:
    """Tracks a scratch buffer allocation for a step."""
    buffer: Any  # MTLBuffer
    size: int
    name: str


class EngineStepContext:
    """Per-step execution context for Metal engine.

    This class manages all resources needed for a single scheduler step:
    - Command buffer and compute encoder
    - Scratch buffer allocations
    - Phase transitions (ENCODE -> SUBMIT -> READBACK)
    - Profiling markers

    The context is used as a context manager to ensure proper cleanup.

    Attributes:
        context: Parent MetalEngineContext
        step_id: Unique step identifier
        step_kind: "prefill" or "decode"
        num_tokens: Number of tokens in this step
        num_seqs: Number of sequences in this step
    """

    def __init__(
        self,
        engine_context: Any,  # MetalEngineContext
        step_id: int,
        step_kind: str = "decode",
        num_tokens: int = 0,
        num_seqs: int = 0,
    ):
        """Initialize step context.

        Args:
            engine_context: Parent MetalEngineContext
            step_id: Unique step identifier
            step_kind: "prefill" or "decode"
            num_tokens: Number of tokens in this step
            num_seqs: Number of sequences in this step
        """
        self._context = engine_context
        self.step_id = step_id
        self.step_kind = step_kind
        self.num_tokens = num_tokens
        self.num_seqs = num_seqs

        # Command buffer state
        self._command_buffer: Optional[Any] = None
        self._encoder: Optional[Any] = None
        self._is_submitted = False
        self._is_completed = False

        # Scratch allocations
        self._scratch_allocations: List[ScratchAllocation] = []

        # Phase tracking
        self._current_phase = EnginePhase.IDLE

        # Profiling
        self._profiler = get_profiler() if PROFILING_ENABLED else None

    def __enter__(self) -> "EngineStepContext":
        """Enter step context - starts ENCODE phase."""
        # Start profiling
        if self._profiler:
            self._profiler.begin_step(
                step_id=self.step_id,
                step_kind=self.step_kind,
                num_tokens=self.num_tokens,
                num_seqs=self.num_seqs,
            )
            self._profiler.mark_encode_start()

        # Create command buffer
        self._command_buffer = self._context.new_command_buffer()

        # Create compute encoder
        self._encoder = self._command_buffer.computeCommandEncoder()
        if self._encoder is None:
            raise RuntimeError("Failed to create compute encoder")

        # Enter ENCODE phase
        self._current_phase = EnginePhase.ENCODE
        EngineHotPathGuard.set_phase(EnginePhase.ENCODE)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit step context - cleanup resources."""
        # Release scratch allocations
        self._release_scratch()

        # Reset phase
        EngineHotPathGuard.set_phase(EnginePhase.IDLE)
        self._current_phase = EnginePhase.IDLE

        # End profiling
        if self._profiler:
            self._profiler.end_step()

        return False  # Don't suppress exceptions

    @property
    def command_buffer(self) -> Any:
        """Get the command buffer for this step."""
        return self._command_buffer

    @property
    def encoder(self) -> Any:
        """Get the compute encoder for this step.

        Note: Prefer get_compute_encoder() for ops that may interleave
        with MPS operations.
        """
        return self._encoder

    def get_compute_encoder(self) -> Any:
        """Get compute encoder, re-opening if needed.

        This method handles the case where the encoder was temporarily
        ended for MPS operations. If the encoder is closed, it re-opens
        a new one.

        Returns:
            MTLComputeCommandEncoder

        Raises:
            RuntimeError: If not in ENCODE phase
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("get_compute_encoder only allowed during ENCODE phase")

        if self._encoder is None:
            # Re-open encoder (was closed for MPS)
            self._encoder = self._command_buffer.computeCommandEncoder()
            if self._encoder is None:
                raise RuntimeError("Failed to re-create compute encoder")

        return self._encoder

    def end_compute_encoder_for_mps(self) -> Any:
        """End compute encoder to allow MPS encoding.

        MPS operations (MPSMatrixMultiplication, etc.) encode directly to
        command buffer, not to a compute encoder. This method ends the
        current encoder and returns the command buffer for MPS use.

        After MPS encoding, call get_compute_encoder() to re-open
        the compute encoder for further kernel dispatch.

        Returns:
            MTLCommandBuffer for MPS encoding

        Raises:
            RuntimeError: If not in ENCODE phase
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("end_compute_encoder_for_mps only allowed during ENCODE phase")

        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None

        return self._command_buffer

    @property
    def is_encoding(self) -> bool:
        """Check if in ENCODE phase."""
        return self._current_phase == EnginePhase.ENCODE

    @property
    def is_submitted(self) -> bool:
        """Check if command buffer has been submitted."""
        return self._is_submitted

    @property
    def is_completed(self) -> bool:
        """Check if execution is complete."""
        return self._is_completed

    def allocate_scratch(self, size: int, name: str = "scratch") -> Any:
        """Allocate scratch buffer for this step.

        Args:
            size: Buffer size in bytes
            name: Name for debugging

        Returns:
            MTLBuffer

        Raises:
            RuntimeError: If not in ENCODE phase
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("Scratch allocation only allowed during ENCODE phase")

        buffer = self._context.allocate_scratch(size, step_id=self.step_id)
        self._scratch_allocations.append(ScratchAllocation(buffer, size, name))
        return buffer

    def _release_scratch(self) -> None:
        """Release all scratch allocations for this step."""
        for alloc in self._scratch_allocations:
            self._context.release_scratch(alloc.buffer)
        self._scratch_allocations.clear()

    def set_pipeline(self, pipeline: Any) -> None:
        """Set compute pipeline on encoder.

        Args:
            pipeline: MTLComputePipelineState
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("set_pipeline only allowed during ENCODE phase")
        self._encoder.setComputePipelineState_(pipeline)

    def set_buffer(self, buffer: Any, offset: int, index: int) -> None:
        """Set buffer at index on encoder.

        Args:
            buffer: MTLBuffer
            offset: Byte offset into buffer
            index: Argument table index
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("set_buffer only allowed during ENCODE phase")
        self._encoder.setBuffer_offset_atIndex_(buffer, offset, index)

    def set_bytes(self, data: bytes, index: int) -> None:
        """Set inline bytes at index on encoder.

        Args:
            data: Bytes to set
            index: Argument table index
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("set_bytes only allowed during ENCODE phase")
        self._encoder.setBytes_length_atIndex_(data, len(data), index)

    def dispatch_threads(
        self,
        grid_size: tuple,
        threadgroup_size: tuple,
    ) -> None:
        """Dispatch compute threads.

        Args:
            grid_size: (width, height, depth) of thread grid
            threadgroup_size: (width, height, depth) of threadgroups
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("dispatch_threads only allowed during ENCODE phase")

        from Metal import MTLSize
        grid = MTLSize(grid_size[0], grid_size[1], grid_size[2])
        threads = MTLSize(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2])
        self._encoder.dispatchThreads_threadsPerThreadgroup_(grid, threads)

    def dispatch_threadgroups(
        self,
        threadgroups: tuple,
        threads_per_threadgroup: tuple,
    ) -> None:
        """Dispatch threadgroups.

        Args:
            threadgroups: (width, height, depth) number of threadgroups
            threads_per_threadgroup: (width, height, depth) threads per group
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("dispatch_threadgroups only allowed during ENCODE phase")

        from Metal import MTLSize
        groups = MTLSize(threadgroups[0], threadgroups[1], threadgroups[2])
        threads = MTLSize(threads_per_threadgroup[0], threads_per_threadgroup[1], threads_per_threadgroup[2])
        self._encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)

    def memory_barrier(self) -> None:
        """Insert memory barrier for buffer coherence.

        Call this between operations that have producer-consumer dependency
        on the same buffers.
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("memory_barrier only allowed during ENCODE phase")

        if HAS_METAL and MTLBarrierScopeBuffers is not None:
            self._encoder.memoryBarrierWithScope_(MTLBarrierScopeBuffers)

    def end_encoding(self) -> None:
        """End command encoding.

        Call this before submit() to finalize the encoder.
        Safe to call even if encoder was already ended for MPS.
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("end_encoding only allowed during ENCODE phase")

        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None

        if self._profiler:
            self._profiler.mark_encode_end()

    def submit(self) -> None:
        """Submit command buffer (transition to SUBMIT phase).

        This commits the command buffer to the GPU. No more encoding
        is allowed after this. The command buffer will execute
        asynchronously until wait_until_completed() is called.
        """
        if self._current_phase != EnginePhase.ENCODE:
            raise RuntimeError("submit only allowed from ENCODE phase")
        if self._encoder is not None:
            raise RuntimeError("Must call end_encoding() before submit()")

        # Transition to SUBMIT phase
        self._current_phase = EnginePhase.SUBMIT
        EngineHotPathGuard.set_phase(EnginePhase.SUBMIT)

        if self._profiler:
            self._profiler.mark_submit()

        # Commit command buffer
        self._command_buffer.commit()
        self._is_submitted = True

    def wait_until_completed(self) -> None:
        """Wait for GPU execution to complete (transition to READBACK phase).

        This blocks until all commands in the buffer have finished.
        After this, output buffers can be read.
        """
        if not self._is_submitted:
            raise RuntimeError("Must call submit() before wait_until_completed()")
        if self._is_completed:
            return

        # Transition to READBACK phase
        self._current_phase = EnginePhase.READBACK
        EngineHotPathGuard.set_phase(EnginePhase.READBACK)

        # Wait for completion
        self._command_buffer.waitUntilCompleted()
        self._is_completed = True

        if self._profiler:
            self._profiler.mark_execute_complete()

        # Check for errors
        error = self._command_buffer.error()
        if error is not None:
            raise RuntimeError(f"Command buffer execution failed: {error}")

    def readback_complete(self) -> None:
        """Mark readback as complete.

        Call this after reading all output buffers.
        """
        if self._current_phase != EnginePhase.READBACK:
            raise RuntimeError("readback_complete only allowed in READBACK phase")

        if self._profiler:
            self._profiler.mark_readback_complete()


@contextmanager
def step_execution(
    engine_context: Any,
    step_id: int,
    step_kind: str = "decode",
    num_tokens: int = 0,
    num_seqs: int = 0,
):
    """Context manager for step execution.

    This is a convenience wrapper that creates an EngineStepContext.

    Usage:
        with step_execution(context, step_id=0) as step:
            # Encode operations
            step.set_pipeline(pipeline)
            step.set_buffer(buffer, 0, 0)
            step.dispatch_threadgroups(...)

            # Submit and wait
            step.end_encoding()
            step.submit()
            step.wait_until_completed()

            # Read outputs
            ...
            step.readback_complete()
    """
    step_ctx = EngineStepContext(
        engine_context,
        step_id=step_id,
        step_kind=step_kind,
        num_tokens=num_tokens,
        num_seqs=num_seqs,
    )
    with step_ctx:
        yield step_ctx
