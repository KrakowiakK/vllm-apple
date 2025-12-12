# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict Mode Enforcement for vLLM-Apple Metal Engine v2.0.

This module provides monkey-patching for PyTorch MPS APIs to enforce the
v2.0 "No PyTorch-MPS in hot path" invariant when VLLM_METAL_STRICT_NO_MPS=1.

When strict mode is enabled:
- torch.mps.synchronize() raises during ENCODE/SUBMIT phases
- tensor.item(), .tolist(), .cpu(), .numpy() on MPS tensors raise during hot path
- Any implicit MPS synchronization is detected and blocked

Usage:
    # Enable at startup (typically in platform.py)
    from vllm_apple.engine.strict_mode import enable_strict_mode
    enable_strict_mode()

    # Now any MPS operations during hot path will raise
    with EngineHotPathGuard.encode_phase():
        torch.mps.synchronize()  # Raises RuntimeError
"""

import os
from functools import wraps
from typing import Callable, TypeVar, Any

import torch

from vllm.logger import init_logger
from .guards import EngineHotPathGuard, EnginePhase

logger = init_logger(__name__)

# Environment variable to enable strict mode
STRICT_NO_MPS_ENV = "VLLM_METAL_STRICT_NO_MPS"

# Global flag tracking if strict mode is enabled
_strict_mode_enabled = False
_patches_applied = False

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])


def is_strict_mode() -> bool:
    """Check if strict mode is enabled."""
    return os.environ.get(STRICT_NO_MPS_ENV, "0") == "1"


def get_strict_mode_enabled() -> bool:
    """Check if strict mode patches have been applied."""
    return _strict_mode_enabled


def _wrap_forbidden_in_hot_path(name: str) -> Callable[[F], F]:
    """Decorator factory to wrap functions that are forbidden in hot path.

    Args:
        name: Human-readable name of the operation for error messages

    Returns:
        Decorator that wraps the function with hot path checking
    """
    def decorator(original_fn: F) -> F:
        @wraps(original_fn)
        def wrapper(*args, **kwargs):
            EngineHotPathGuard.assert_not_hot_path(name)
            return original_fn(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def _wrap_mps_tensor_method(method_name: str, original_method: Callable) -> Callable:
    """Wrap a tensor method to check for MPS tensor + hot path combination.

    Args:
        method_name: Name of the method (e.g., "item", "cpu")
        original_method: The original method to wrap

    Returns:
        Wrapped method that checks for MPS + hot path violations
    """
    @wraps(original_method)
    def wrapper(self, *args, **kwargs):
        # Only check if tensor is on MPS device
        if hasattr(self, 'device') and self.device.type == "mps":
            EngineHotPathGuard.assert_not_hot_path(
                f"tensor.{method_name}() on MPS tensor"
            )
        return original_method(self, *args, **kwargs)
    return wrapper


# Store original functions for potential restoration
_original_functions = {}


def enable_strict_mode() -> None:
    """Enable strict mode by patching torch.mps APIs.

    This function applies monkey-patches to:
    - torch.mps.synchronize()
    - torch.mps.commit() (if exists)
    - torch.Tensor.item() (for MPS tensors)
    - torch.Tensor.tolist() (for MPS tensors)
    - torch.Tensor.cpu() (for MPS tensors)
    - torch.Tensor.numpy() (for MPS tensors)

    The patches only take effect when in engine hot path (ENCODE/SUBMIT phases).
    Outside the hot path, the original behavior is preserved.

    Should be called once at startup if VLLM_METAL_STRICT_NO_MPS=1.
    """
    global _strict_mode_enabled, _patches_applied

    if not is_strict_mode():
        logger.info("Strict mode not enabled (VLLM_METAL_STRICT_NO_MPS != 1)")
        return

    if _patches_applied:
        logger.warning("Strict mode patches already applied")
        return

    logger.info("Enabling strict mode: patching PyTorch MPS APIs")

    # Patch torch.mps.synchronize
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        _original_functions['torch.mps.synchronize'] = torch.mps.synchronize
        torch.mps.synchronize = _wrap_forbidden_in_hot_path(
            "torch.mps.synchronize"
        )(torch.mps.synchronize)
        logger.debug("Patched torch.mps.synchronize")

    # Patch torch.mps.commit if it exists
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'commit'):
        _original_functions['torch.mps.commit'] = torch.mps.commit
        torch.mps.commit = _wrap_forbidden_in_hot_path(
            "torch.mps.commit"
        )(torch.mps.commit)
        logger.debug("Patched torch.mps.commit")

    # Patch tensor methods that cause implicit sync on MPS
    tensor_methods_to_patch = ['item', 'tolist', 'cpu', 'numpy']

    for method_name in tensor_methods_to_patch:
        if hasattr(torch.Tensor, method_name):
            original = getattr(torch.Tensor, method_name)
            _original_functions[f'torch.Tensor.{method_name}'] = original
            setattr(
                torch.Tensor,
                method_name,
                _wrap_mps_tensor_method(method_name, original)
            )
            logger.debug(f"Patched torch.Tensor.{method_name}")

    _patches_applied = True
    _strict_mode_enabled = True
    logger.info("Strict mode enabled: MPS operations will raise in engine hot path")


def disable_strict_mode() -> None:
    """Disable strict mode by restoring original functions.

    This is primarily useful for testing.
    """
    global _strict_mode_enabled, _patches_applied

    if not _patches_applied:
        return

    logger.info("Disabling strict mode: restoring original PyTorch MPS APIs")

    # Restore torch.mps functions
    if 'torch.mps.synchronize' in _original_functions:
        torch.mps.synchronize = _original_functions['torch.mps.synchronize']

    if 'torch.mps.commit' in _original_functions:
        torch.mps.commit = _original_functions['torch.mps.commit']

    # Restore tensor methods
    for method_name in ['item', 'tolist', 'cpu', 'numpy']:
        key = f'torch.Tensor.{method_name}'
        if key in _original_functions:
            setattr(torch.Tensor, method_name, _original_functions[key])

    _original_functions.clear()
    _patches_applied = False
    _strict_mode_enabled = False
    logger.info("Strict mode disabled")


class StrictModeContext:
    """Context manager for temporarily enabling/disabling strict mode patches.

    Useful for testing specific code paths with or without strict mode.

    Usage:
        with StrictModeContext(enabled=True):
            # Code here runs with strict mode
            pass

        with StrictModeContext(enabled=False):
            # Code here runs without strict mode
            pass
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._was_enabled = False

    def __enter__(self):
        self._was_enabled = _strict_mode_enabled
        if self.enabled and not _patches_applied:
            # Temporarily set env var for enable_strict_mode check
            os.environ[STRICT_NO_MPS_ENV] = "1"
            enable_strict_mode()
        elif not self.enabled and _patches_applied:
            disable_strict_mode()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._was_enabled and not _strict_mode_enabled:
            os.environ[STRICT_NO_MPS_ENV] = "1"
            enable_strict_mode()
        elif not self._was_enabled and _strict_mode_enabled:
            disable_strict_mode()
            if STRICT_NO_MPS_ENV in os.environ:
                del os.environ[STRICT_NO_MPS_ENV]
        return False


def assert_no_mps_tensors(*tensors: torch.Tensor, context: str = "") -> None:
    """Assert that none of the provided tensors are on MPS device.

    This is a utility for validating engine boundary inputs.

    Args:
        *tensors: Tensors to check
        context: Description of where this check is being performed

    Raises:
        RuntimeError: If any tensor is on MPS device and strict mode is enabled
    """
    if not _strict_mode_enabled:
        return

    for i, tensor in enumerate(tensors):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type == "mps":
            raise RuntimeError(
                f"MPS tensor detected at engine boundary{' (' + context + ')' if context else ''}. "
                f"Tensor #{i} is on device '{tensor.device}'. "
                f"Engine mode requires CPU tensors at the vLLM↔engine boundary. "
                f"This violates the v2.0 'no MPS in hot path' invariant."
            )


def assert_cpu_tensors(**named_tensors: torch.Tensor) -> None:
    """Assert that all named tensors are on CPU device.

    Args:
        **named_tensors: Named tensors to check (e.g., query=q_tensor)

    Raises:
        RuntimeError: If any tensor is not on CPU and strict mode is enabled
    """
    if not _strict_mode_enabled:
        return

    for name, tensor in named_tensors.items():
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type != "cpu":
            raise RuntimeError(
                f"Non-CPU tensor '{name}' detected at engine boundary. "
                f"Device: {tensor.device}. Expected: cpu. "
                f"Engine mode requires CPU tensors at the vLLM↔engine boundary."
            )
