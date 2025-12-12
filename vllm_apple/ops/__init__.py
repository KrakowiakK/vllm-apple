# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple-optimized operations for vLLM."""

from vllm_apple.ops import apple_fused_moe  # noqa: F401

__all__ = ["apple_fused_moe"]
