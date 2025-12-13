# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Configuration for vLLM-Apple Metal Engine v2.0.

This module provides configuration management for the Metal engine mode.
Configuration is primarily controlled through environment variables to
maintain compatibility with vLLM's configuration patterns.

Environment Variables:
    VLLM_APPLE_USE_ENGINE:
        Enable v2.0 engine mode. When set to "1", the Metal engine takes
        over GPU execution from PyTorch-MPS. Default: "0"

    VLLM_APPLE_ENGINE_PREFILL:
        Enable running prefill/mixed steps in the engine. When disabled,
        prefill continues to route through the PyTorch path (and may require
        KV cache sync). Default: "0"

    VLLM_METAL_STRICT_NO_MPS:
        Enable strict mode that raises RuntimeError if any PyTorch-MPS
        operations are detected during the engine hot path. Useful for
        development and debugging. Default: "0"

    VLLM_METAL_PROFILE:
        Enable step-level profiling for performance analysis. Default: "0"

    VLLM_METAL_CAPTURE_NEXT_STEP:
        Trigger GPU trace capture for the next engine step using
        MTLCaptureManager. Automatically resets after capture. Default: "0"

    VLLM_METAL_PATH_RESOURCES:
        Override the path to Metal kernel resources (.metallib files).
        If not set, uses the default path in the package. Default: None

    VLLM_METAL_TOPK_LOGITS:
        Enable top-k logits readback optimization. Set to the k value
        (e.g., "50") to enable. When disabled, full logits are returned.
        Default: None (disabled)

Usage:
    from vllm_apple.engine.config import is_engine_mode_enabled, EngineConfig

    if is_engine_mode_enabled():
        config = EngineConfig.from_env()
        # Use engine mode with config
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# Environment variable names
ENGINE_MODE_ENV = "VLLM_APPLE_USE_ENGINE"
ENGINE_PREFILL_ENV = "VLLM_APPLE_ENGINE_PREFILL"
STRICT_MODE_ENV = "VLLM_METAL_STRICT_NO_MPS"
PROFILE_ENV = "VLLM_METAL_PROFILE"
CAPTURE_TRACE_ENV = "VLLM_METAL_CAPTURE_NEXT_STEP"
RESOURCE_PATH_ENV = "VLLM_METAL_PATH_RESOURCES"
TOPK_LOGITS_ENV = "VLLM_METAL_TOPK_LOGITS"


def is_engine_mode_enabled() -> bool:
    """Check if engine mode is enabled via environment variable.

    Returns:
        True if VLLM_APPLE_USE_ENGINE=1
    """
    return os.environ.get(ENGINE_MODE_ENV, "0") == "1"


def is_engine_prefill_enabled() -> bool:
    """Check if engine prefill execution is enabled via environment variable.

    Returns:
        True if VLLM_APPLE_ENGINE_PREFILL=1
    """
    return os.environ.get(ENGINE_PREFILL_ENV, "0") == "1"


def is_strict_mode_enabled() -> bool:
    """Check if strict mode is enabled via environment variable.

    Returns:
        True if VLLM_METAL_STRICT_NO_MPS=1
    """
    return os.environ.get(STRICT_MODE_ENV, "0") == "1"


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled via environment variable.

    Returns:
        True if VLLM_METAL_PROFILE=1
    """
    return os.environ.get(PROFILE_ENV, "0") == "1"


def get_topk_logits() -> Optional[int]:
    """Get the top-k logits value if enabled.

    Returns:
        Integer k value, or None if disabled
    """
    value = os.environ.get(TOPK_LOGITS_ENV)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid {TOPK_LOGITS_ENV} value: {value}, ignoring")
        return None


def get_resource_path() -> Optional[str]:
    """Get the Metal resource path override.

    Returns:
        Custom resource path, or None for default
    """
    return os.environ.get(RESOURCE_PATH_ENV)


@dataclass
class EngineConfig:
    """Configuration for Metal engine execution.

    This dataclass holds all engine configuration settings. It can be
    created from environment variables using the `from_env()` class method.

    Attributes:
        engine_mode: Whether engine mode is enabled
        strict_mode: Whether strict no-MPS mode is enabled
        profiling: Whether step-level profiling is enabled
        capture_trace: Whether to capture GPU trace for next step
        resource_path: Override path for Metal kernel resources
        topk_logits: Top-k value for logits readback, or None for full logits
        max_batch_size: Maximum batch size for engine execution
        max_seq_len: Maximum sequence length
        scratch_pool_size_mb: Size of scratch memory pool in MB
        command_buffer_pool_size: Number of command buffers to pre-allocate
    """

    # Feature flags
    engine_mode: bool = False
    engine_prefill: bool = False
    strict_mode: bool = False
    profiling: bool = False
    capture_trace: bool = False

    # Resource paths
    resource_path: Optional[str] = None

    # Logits handling
    topk_logits: Optional[int] = None

    # Engine limits (can be overridden from model config)
    max_batch_size: int = 256
    max_seq_len: int = 8192
    scratch_pool_size_mb: int = 512
    command_buffer_pool_size: int = 4

    # Compute settings
    use_fused_attention: bool = True
    use_fused_mlp: bool = True

    # Debug settings
    validate_inputs: bool = True
    validate_outputs: bool = True
    log_step_timing: bool = False

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Create config from environment variables.

        Returns:
            EngineConfig populated from environment
        """
        return cls(
            engine_mode=is_engine_mode_enabled(),
            engine_prefill=is_engine_prefill_enabled(),
            strict_mode=is_strict_mode_enabled(),
            profiling=is_profiling_enabled(),
            capture_trace=os.environ.get(CAPTURE_TRACE_ENV, "0") == "1",
            resource_path=get_resource_path(),
            topk_logits=get_topk_logits(),
        )

    def validate(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.topk_logits is not None:
            if self.topk_logits < 1:
                raise ValueError(f"topk_logits must be >= 1, got {self.topk_logits}")
            if self.topk_logits > 100000:
                raise ValueError(f"topk_logits must be <= 100000, got {self.topk_logits}")

        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")

        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")

        if self.scratch_pool_size_mb < 1:
            raise ValueError(f"scratch_pool_size_mb must be >= 1, got {self.scratch_pool_size_mb}")

    def log_config(self) -> None:
        """Log current configuration."""
        logger.info(
            "Engine Configuration:\n"
            f"  engine_mode: {self.engine_mode}\n"
            f"  engine_prefill: {self.engine_prefill}\n"
            f"  strict_mode: {self.strict_mode}\n"
            f"  profiling: {self.profiling}\n"
            f"  topk_logits: {self.topk_logits}\n"
            f"  max_batch_size: {self.max_batch_size}\n"
            f"  max_seq_len: {self.max_seq_len}\n"
            f"  scratch_pool_size_mb: {self.scratch_pool_size_mb}"
        )


# Global config instance (lazily initialized)
_engine_config: Optional[EngineConfig] = None


def get_engine_config() -> EngineConfig:
    """Get the global engine configuration.

    Returns a cached EngineConfig instance. The config is created from
    environment variables on first call.

    Returns:
        Global EngineConfig instance
    """
    global _engine_config
    if _engine_config is None:
        _engine_config = EngineConfig.from_env()
    return _engine_config


def set_engine_config(config: EngineConfig) -> None:
    """Set the global engine configuration.

    This is primarily useful for testing.

    Args:
        config: EngineConfig to use globally
    """
    global _engine_config
    _engine_config = config


def reset_engine_config() -> None:
    """Reset the global engine configuration.

    The next call to get_engine_config() will re-read from environment.
    """
    global _engine_config
    _engine_config = None
