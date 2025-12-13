# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-Apple Metal Engine v2.0.

This module provides a first-class Metal execution engine where vLLM remains
the orchestrator (scheduler + sampling) and the engine handles all GPU compute
via MTLBuffer with step-boundary-only synchronization.

Key Invariants:
    1. No PyTorch-MPS in hot path: No torch.device("mps") or torch.mps.synchronize()
    2. Step-boundary-only sync: One command buffer per scheduler step, single wait at end
    3. Engine-owned MTLBuffer KV cache: Single source of truth, no tensor duplication
    4. Three-phase execution: ENCODE → SUBMIT → WAIT/READBACK

Usage:
    from vllm_apple.engine import EngineConfig, is_engine_mode_enabled

    if is_engine_mode_enabled():
        from vllm_apple.engine.context import MetalEngineContext
        from vllm_apple.engine.runner import EngineRunner
        # Use engine mode execution

Environment Variables:
    VLLM_APPLE_USE_ENGINE: Enable v2.0 engine mode (default: 0)
    VLLM_METAL_STRICT_NO_MPS: Strict mode - raise on MPS in hot path (default: 0)
    VLLM_METAL_PROFILE: Enable step-level profiling (default: 0)
    VLLM_METAL_CAPTURE_NEXT_STEP: Capture GPU trace via MTLCaptureManager (default: 0)

Module Structure:
    config.py       - Engine mode flags and configuration
    guards.py       - EngineHotPathGuard, phase enforcement
    strict_mode.py  - Strict mode monkey-patches
    boundary.py     - Input/output boundary validation
    profiling.py    - Step-level profiling
    context.py      - MetalEngineContext (device, queues, pipelines)
    step.py         - EngineStepContext (per-step scratch, command buffer)
    descriptors.py  - StepDescriptor, BatchDescriptor
    tensor.py       - EngineTensor wrapper around MTLBuffer
    kv_cache.py     - EngineKVCache (wraps existing MetalKVCache)
    runner.py       - EngineRunner (main execution)
    ops/            - Operation encoders (attention, gemm, etc.)
    kernels/        - Custom Metal kernels
"""

from .config import (
    EngineConfig,
    is_engine_mode_enabled,
    get_engine_config,
    ENGINE_MODE_ENV,
    STRICT_MODE_ENV,
    PROFILE_ENV,
    CAPTURE_TRACE_ENV,
)

from .guards import (
    EnginePhase,
    EngineHotPathGuard,
    StepPhaseController,
)

from .strict_mode import (
    enable_strict_mode,
    disable_strict_mode,
    is_strict_mode,
    get_strict_mode_enabled,
)

from .boundary import (
    validate_engine_inputs,
    validate_engine_outputs,
    ensure_cpu_tensor,
)

from .profiling import (
    PROFILING_ENABLED,
    get_profiler,
    StepProfile,
    ProfileSummary,
    EngineProfiler,
)

from .descriptors import (
    StepDescriptor,
    BatchDescriptor,
    EngineInputs,
    EngineOutputs,
    KVCacheDescriptor,
    ModelDescriptor,
    SUPPORTED_ENGINE_ARCHITECTURES,
)

# Direct class imports (for type hints and direct instantiation)
from .context import MetalEngineContext
from .step import EngineStepContext
from .kv_cache import EngineKVCache
from .runner import EngineRunner
from .weight_loader import EngineWeightLoader

# Lazy imports for heavy modules (avoid import at module load)
def get_engine_context():
    """Get the global Metal engine context."""
    return MetalEngineContext.get_instance()

def create_step_context(*args, **kwargs):
    """Create a step execution context."""
    return EngineStepContext(*args, **kwargs)

def create_engine_runner(*args, **kwargs):
    """Create an engine runner."""
    return EngineRunner(*args, **kwargs)

def create_weight_loader(*args, **kwargs):
    """Create a weight loader."""
    return EngineWeightLoader(*args, **kwargs)

__all__ = [
    # Config
    "EngineConfig",
    "is_engine_mode_enabled",
    "get_engine_config",
    "ENGINE_MODE_ENV",
    "STRICT_MODE_ENV",
    "PROFILE_ENV",
    "CAPTURE_TRACE_ENV",
    # Guards
    "EnginePhase",
    "EngineHotPathGuard",
    "StepPhaseController",
    # Strict mode
    "enable_strict_mode",
    "disable_strict_mode",
    "is_strict_mode",
    "get_strict_mode_enabled",
    # Boundary validation
    "validate_engine_inputs",
    "validate_engine_outputs",
    "ensure_cpu_tensor",
    # Profiling
    "PROFILING_ENABLED",
    "get_profiler",
    "StepProfile",
    "ProfileSummary",
    "EngineProfiler",
    # Descriptors
    "StepDescriptor",
    "BatchDescriptor",
    "EngineInputs",
    "EngineOutputs",
    "KVCacheDescriptor",
    "ModelDescriptor",
    "SUPPORTED_ENGINE_ARCHITECTURES",
    # Core classes
    "MetalEngineContext",
    "EngineStepContext",
    "EngineKVCache",
    "EngineRunner",
    "EngineWeightLoader",
    # Context helpers
    "get_engine_context",
    "create_step_context",
    # Runner and weight loader
    "create_engine_runner",
    "create_weight_loader",
]
