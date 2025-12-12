# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Apple GPU Plugin - Native Metal backend for Apple Silicon.

This plugin provides optimized inference for vLLM on Apple Silicon using
custom Metal kernels. Following the vllm-gaudi architecture pattern.
"""

from typing import Optional

__version__ = "0.1.0"


def register() -> Optional[str]:
    """Register Apple platform with vLLM.

    Returns the fully-qualified name of the ApplePlatform class
    if MPS is available, otherwise None.

    This follows the vLLM platform plugin convention where plugins
    return a string (qualname) instead of the class itself.
    """
    import torch

    # Check if MPS (Metal Performance Shaders) is available
    if not torch.backends.mps.is_available():
        return None

    # Return the fully-qualified class name
    return "vllm_apple.platform.ApplePlatform"


def register_ops():
    """Register Apple-optimized operations.

    Imports modules that register custom ops via @register_oot decorators.
    """
    # Import to trigger @register_oot decorators
    from vllm_apple.ops import apple_fused_moe  # noqa: F401


def register_models():
    """Register Apple-optimized model implementations.

    Optional: model-specific optimizations can be registered here.
    """
    pass


def pre_register_and_update():
    """Platform patches applied before workers start.

    Enables MPS fallback to CPU for operations not yet supported on Metal.
    """
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
