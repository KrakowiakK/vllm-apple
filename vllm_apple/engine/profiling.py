# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-Level Profiling for vLLM-Apple Metal Engine v2.0.

This module provides profiling utilities for engine step execution.
It tracks timing at the step level (not per-layer/per-op) to maintain
the v2.0 synchronization policy while still providing useful metrics.

Profiling is enabled via VLLM_METAL_PROFILE=1

Features:
- Step-level timing (encode, submit, execute, readback)
- Kernel breakdown (optional, for detailed analysis)
- GPU trace capture via MTLCaptureManager
- Memory tracking

Usage:
    from vllm_apple.engine.profiling import get_profiler, PROFILING_ENABLED

    if PROFILING_ENABLED:
        profiler = get_profiler()
        profiler.begin_step(step_id=0, step_kind="decode", num_tokens=4, num_seqs=4)

        # ... execute step ...

        profiler.end_step()

        # Get summary
        summary = profiler.get_summary()
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from vllm.logger import init_logger

logger = init_logger(__name__)

# Environment variables for profiling
PROFILING_ENABLED = os.environ.get("VLLM_METAL_PROFILE", "0") == "1"
CAPTURE_TRACE = os.environ.get("VLLM_METAL_CAPTURE_NEXT_STEP", "0") == "1"


@dataclass
class StepProfile:
    """Profile data for a single engine step."""

    # Step identification
    step_id: int
    step_kind: str  # "prefill" or "decode"
    num_tokens: int
    num_seqs: int

    # Timing (milliseconds)
    encode_time_ms: float = 0.0
    submit_time_ms: float = 0.0
    execute_time_ms: float = 0.0  # GPU execution (from submit to wait complete)
    readback_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Kernel breakdown (optional, for detailed profiling)
    attention_kernel_ms: float = 0.0
    kv_write_kernel_ms: float = 0.0
    gemm_kernel_ms: float = 0.0
    norm_kernel_ms: float = 0.0
    other_kernel_ms: float = 0.0

    # Memory (bytes)
    peak_memory_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_kind": self.step_kind,
            "num_tokens": self.num_tokens,
            "num_seqs": self.num_seqs,
            "encode_time_ms": self.encode_time_ms,
            "submit_time_ms": self.submit_time_ms,
            "execute_time_ms": self.execute_time_ms,
            "readback_time_ms": self.readback_time_ms,
            "total_time_ms": self.total_time_ms,
            "attention_kernel_ms": self.attention_kernel_ms,
            "kv_write_kernel_ms": self.kv_write_kernel_ms,
            "gemm_kernel_ms": self.gemm_kernel_ms,
            "norm_kernel_ms": self.norm_kernel_ms,
            "other_kernel_ms": self.other_kernel_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass
class ProfileSummary:
    """Summary statistics across multiple steps."""

    num_steps: int = 0
    num_decode_steps: int = 0
    num_prefill_steps: int = 0

    # Total tokens processed
    total_tokens: int = 0
    total_decode_tokens: int = 0
    total_prefill_tokens: int = 0

    # Average timing (ms)
    avg_step_time_ms: float = 0.0
    avg_encode_time_ms: float = 0.0
    avg_execute_time_ms: float = 0.0
    avg_readback_time_ms: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0

    # Percentiles (p50, p95, p99)
    p50_step_time_ms: float = 0.0
    p95_step_time_ms: float = 0.0
    p99_step_time_ms: float = 0.0


class EngineProfiler:
    """Step-level profiler for Metal engine.

    Tracks timing and metrics at the step level, respecting the v2.0
    synchronization policy (no per-layer timing inside hot path).
    """

    def __init__(self, max_history: int = 1000):
        """Initialize profiler.

        Args:
            max_history: Maximum number of step profiles to keep in memory
        """
        self.max_history = max_history
        self.profiles: List[StepProfile] = []
        self._current_step: Optional[StepProfile] = None
        self._timestamps: Dict[str, float] = {}
        self._step_counter: int = 0
        self._enabled = PROFILING_ENABLED

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    def begin_step(
        self,
        step_id: Optional[int] = None,
        step_kind: str = "decode",
        num_tokens: int = 0,
        num_seqs: int = 0,
    ) -> None:
        """Begin profiling a new step.

        Args:
            step_id: Optional step ID (auto-incremented if None)
            step_kind: "prefill" or "decode"
            num_tokens: Number of tokens in this step
            num_seqs: Number of sequences in this step
        """
        if not self._enabled:
            return

        if step_id is None:
            step_id = self._step_counter
            self._step_counter += 1

        self._current_step = StepProfile(
            step_id=step_id,
            step_kind=step_kind,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
        )
        self._timestamps = {"step_start": time.perf_counter()}

    def mark_encode_start(self) -> None:
        """Mark the start of encode phase."""
        if not self._enabled or self._current_step is None:
            return
        self._timestamps["encode_start"] = time.perf_counter()

    def mark_encode_end(self) -> None:
        """Mark the end of encode phase."""
        if not self._enabled or self._current_step is None:
            return
        self._timestamps["encode_end"] = time.perf_counter()

    def mark_submit(self) -> None:
        """Mark command buffer submission."""
        if not self._enabled or self._current_step is None:
            return
        self._timestamps["submit"] = time.perf_counter()

    def mark_execute_complete(self) -> None:
        """Mark GPU execution completion (after waitUntilCompleted)."""
        if not self._enabled or self._current_step is None:
            return
        self._timestamps["execute_complete"] = time.perf_counter()

    def mark_readback_complete(self) -> None:
        """Mark readback completion."""
        if not self._enabled or self._current_step is None:
            return
        self._timestamps["readback_complete"] = time.perf_counter()

    def add_kernel_time(self, kernel_type: str, time_ms: float) -> None:
        """Add time for a specific kernel type.

        Note: This should only be used with GPU-side timing (e.g., from
        command buffer timestamps), not CPU timing during hot path.

        Args:
            kernel_type: One of "attention", "kv_write", "gemm", "norm", "other"
            time_ms: Time in milliseconds
        """
        if not self._enabled or self._current_step is None:
            return

        if kernel_type == "attention":
            self._current_step.attention_kernel_ms += time_ms
        elif kernel_type == "kv_write":
            self._current_step.kv_write_kernel_ms += time_ms
        elif kernel_type == "gemm":
            self._current_step.gemm_kernel_ms += time_ms
        elif kernel_type == "norm":
            self._current_step.norm_kernel_ms += time_ms
        else:
            self._current_step.other_kernel_ms += time_ms

    def end_step(self) -> Optional[StepProfile]:
        """End profiling for current step.

        Returns:
            The completed StepProfile, or None if profiling disabled
        """
        if not self._enabled or self._current_step is None:
            return None

        now = time.perf_counter()
        ts = self._timestamps

        # Calculate timing
        step_start = ts.get("step_start", now)
        encode_start = ts.get("encode_start", step_start)
        encode_end = ts.get("encode_end", encode_start)
        submit = ts.get("submit", encode_end)
        execute_complete = ts.get("execute_complete", submit)
        readback_complete = ts.get("readback_complete", now)

        self._current_step.encode_time_ms = (encode_end - encode_start) * 1000
        self._current_step.submit_time_ms = (submit - encode_end) * 1000
        self._current_step.execute_time_ms = (execute_complete - submit) * 1000
        self._current_step.readback_time_ms = (readback_complete - execute_complete) * 1000
        self._current_step.total_time_ms = (readback_complete - step_start) * 1000

        # Store profile
        self.profiles.append(self._current_step)

        # Trim history if needed
        if len(self.profiles) > self.max_history:
            self.profiles = self.profiles[-self.max_history:]

        result = self._current_step
        self._current_step = None
        self._timestamps = {}

        return result

    @contextmanager
    def profile_step(
        self,
        step_kind: str = "decode",
        num_tokens: int = 0,
        num_seqs: int = 0,
    ):
        """Context manager for profiling a step.

        Usage:
            with profiler.profile_step(step_kind="decode", num_tokens=4):
                # Encode, submit, wait, readback
                pass
        """
        self.begin_step(step_kind=step_kind, num_tokens=num_tokens, num_seqs=num_seqs)
        try:
            yield self
        finally:
            self.end_step()

    def get_last_profile(self) -> Optional[StepProfile]:
        """Get the most recent step profile."""
        return self.profiles[-1] if self.profiles else None

    def get_summary(self, last_n: Optional[int] = None) -> ProfileSummary:
        """Get summary statistics.

        Args:
            last_n: Only consider last N steps, or None for all

        Returns:
            ProfileSummary with aggregated statistics
        """
        profiles = self.profiles[-last_n:] if last_n else self.profiles

        if not profiles:
            return ProfileSummary()

        summary = ProfileSummary()
        summary.num_steps = len(profiles)

        total_time_ms = 0.0
        total_encode_ms = 0.0
        total_execute_ms = 0.0
        total_readback_ms = 0.0
        step_times = []

        for p in profiles:
            if p.step_kind == "decode":
                summary.num_decode_steps += 1
                summary.total_decode_tokens += p.num_tokens
            else:
                summary.num_prefill_steps += 1
                summary.total_prefill_tokens += p.num_tokens

            summary.total_tokens += p.num_tokens
            total_time_ms += p.total_time_ms
            total_encode_ms += p.encode_time_ms
            total_execute_ms += p.execute_time_ms
            total_readback_ms += p.readback_time_ms
            step_times.append(p.total_time_ms)

        # Averages
        summary.avg_step_time_ms = total_time_ms / len(profiles)
        summary.avg_encode_time_ms = total_encode_ms / len(profiles)
        summary.avg_execute_time_ms = total_execute_ms / len(profiles)
        summary.avg_readback_time_ms = total_readback_ms / len(profiles)

        # Throughput
        total_time_s = total_time_ms / 1000.0
        if total_time_s > 0:
            summary.tokens_per_second = summary.total_tokens / total_time_s
            # Decode throughput (for decode steps only)
            decode_time_ms = sum(p.total_time_ms for p in profiles if p.step_kind == "decode")
            if decode_time_ms > 0:
                summary.decode_tokens_per_second = summary.total_decode_tokens / (decode_time_ms / 1000.0)

        # Percentiles
        step_times.sort()
        n = len(step_times)
        if n > 0:
            summary.p50_step_time_ms = step_times[int(n * 0.5)]
            summary.p95_step_time_ms = step_times[int(n * 0.95)] if n >= 20 else step_times[-1]
            summary.p99_step_time_ms = step_times[int(n * 0.99)] if n >= 100 else step_times[-1]

        return summary

    def reset(self) -> None:
        """Reset all profiling data."""
        self.profiles.clear()
        self._current_step = None
        self._timestamps = {}

    def log_summary(self, last_n: Optional[int] = 50) -> None:
        """Log a summary of recent profiling data.

        Args:
            last_n: Number of recent steps to summarize
        """
        if not self._enabled:
            return

        summary = self.get_summary(last_n)

        logger.info(
            f"Engine Profile Summary (last {last_n or 'all'} steps):\n"
            f"  Steps: {summary.num_steps} ({summary.num_decode_steps} decode, {summary.num_prefill_steps} prefill)\n"
            f"  Tokens: {summary.total_tokens} total ({summary.total_decode_tokens} decode)\n"
            f"  Throughput: {summary.tokens_per_second:.1f} tok/s (decode: {summary.decode_tokens_per_second:.1f} tok/s)\n"
            f"  Step time: avg={summary.avg_step_time_ms:.2f}ms, "
            f"p50={summary.p50_step_time_ms:.2f}ms, p95={summary.p95_step_time_ms:.2f}ms\n"
            f"  Breakdown: encode={summary.avg_encode_time_ms:.2f}ms, "
            f"execute={summary.avg_execute_time_ms:.2f}ms, readback={summary.avg_readback_time_ms:.2f}ms"
        )


# Global profiler instance
_profiler: Optional[EngineProfiler] = None


def get_profiler() -> EngineProfiler:
    """Get the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = EngineProfiler()
    return _profiler


def capture_gpu_trace(command_buffer: Any, step_id: int = 0) -> bool:
    """Capture GPU trace for debugging via MTLCaptureManager.

    This captures the next step's GPU execution for analysis in Xcode.
    Set VLLM_METAL_CAPTURE_NEXT_STEP=1 to enable.

    Args:
        command_buffer: MTLCommandBuffer to capture
        step_id: Step ID for the capture file name

    Returns:
        True if capture was started, False otherwise
    """
    global CAPTURE_TRACE

    if not CAPTURE_TRACE:
        return False

    try:
        from Metal import MTLCaptureManager, MTLCaptureDescriptor

        capture_manager = MTLCaptureManager.sharedCaptureManager()

        # Check if capture is supported
        if not capture_manager.supportsDestination_(1):  # GPUTraceDocument
            logger.warning("GPU trace capture not supported")
            return False

        descriptor = MTLCaptureDescriptor.alloc().init()
        descriptor.setCaptureObject_(command_buffer.device())
        descriptor.setDestination_(1)  # GPUTraceDocument

        # Set output URL
        import os
        output_path = os.path.expanduser(f"~/Desktop/vllm_metal_trace_step{step_id}.gputrace")
        from Foundation import NSURL
        descriptor.setOutputURL_(NSURL.fileURLWithPath_(output_path))

        error = None
        success = capture_manager.startCaptureWithDescriptor_error_(descriptor, error)

        if success:
            logger.info(f"Started GPU trace capture to {output_path}")
            # Disable after one capture
            CAPTURE_TRACE = False
            os.environ["VLLM_METAL_CAPTURE_NEXT_STEP"] = "0"
            return True
        else:
            logger.error(f"Failed to start GPU trace: {error}")
            return False

    except ImportError:
        logger.warning("Metal framework not available for GPU trace capture")
        return False
    except Exception as e:
        logger.error(f"GPU trace capture failed: {e}")
        return False


def stop_gpu_trace() -> None:
    """Stop an active GPU trace capture."""
    try:
        from Metal import MTLCaptureManager

        capture_manager = MTLCaptureManager.sharedCaptureManager()
        if capture_manager.isCapturing():
            capture_manager.stopCapture()
            logger.info("Stopped GPU trace capture")
    except Exception as e:
        logger.error(f"Failed to stop GPU trace: {e}")
