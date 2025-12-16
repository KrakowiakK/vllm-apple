# SPDX-License-Identifier: Apache-2.0
"""Debug utilities for vLLM Apple Metal Engine.

This module provides debugging tools for diagnosing numerical divergence
between the Metal engine and PyTorch-MPS reference implementations.

Modules:
    - prefill_checkpoint: Checkpoint capture for engine forward pass
    - pytorch_reference_hooks: Hook system for HuggingFace model comparison

Usage:
    # Enable checkpoint capture
    export VLLM_PREFILL_EQ_DEBUG=1

    # Run inference and compare checkpoints
    from vllm_apple.debug import compare_checkpoints, print_comparison_report
    results, first_div = compare_checkpoints()
    print(print_comparison_report(results, first_div))
"""

from .prefill_checkpoint import (
    CheckpointStore,
    capture_checkpoint,
    get_checkpoint_store,
    compare_checkpoints,
    print_comparison_report,
    reset_stores,
    CHECKPOINT_DEBUG_ENABLED,
)

from .pytorch_reference_hooks import (
    ReferenceHookManager,
    attach_reference_hooks,
    detect_architecture,
    get_num_layers,
)

__all__ = [
    # Checkpoint capture
    "CheckpointStore",
    "capture_checkpoint",
    "get_checkpoint_store",
    "compare_checkpoints",
    "print_comparison_report",
    "reset_stores",
    "CHECKPOINT_DEBUG_ENABLED",
    # PyTorch hooks
    "ReferenceHookManager",
    "attach_reference_hooks",
    "detect_architecture",
    "get_num_layers",
]
