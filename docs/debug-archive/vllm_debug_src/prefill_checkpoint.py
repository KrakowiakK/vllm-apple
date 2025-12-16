# SPDX-License-Identifier: Apache-2.0
"""Checkpoint capture system for debugging prefill divergence.

This module provides infrastructure for capturing intermediate tensors
at specific checkpoints during the engine forward pass, enabling
comparison with PyTorch reference values.

Usage:
    # Enable checkpoint capture
    export VLLM_PREFILL_EQ_DEBUG=1

    # In engine code:
    from vllm_apple.debug import capture_checkpoint
    capture_checkpoint("layer0_input_norm", tensor)

Checkpoints:
    - embed_output: After embedding lookup
    - layer{N}_input_norm: After input LayerNorm
    - layer{N}_qkv_proj: After QKV projection
    - layer{N}_rope_q: After RoPE on Q
    - layer{N}_rope_k: After RoPE on K
    - layer{N}_attn_output: After attention
    - layer{N}_o_proj: After O projection
    - layer{N}_post_attn: After post-attention residual
    - layer{N}_mlp_gate_up: After MLP gate/up projection
    - layer{N}_mlp_act: After MLP activation
    - layer{N}_mlp_down: After MLP down projection
    - final_norm: After final LayerNorm
    - lm_head_logits: Final logits
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import torch

# Check if debug is enabled
CHECKPOINT_DEBUG_ENABLED = True # Hardcoded for debugging
DEBUG_DISK_PATH = "/tmp/vllm_layer_test" # Hardcoded for reliability

# Checkpoint names in execution order
CHECKPOINT_ORDER = [
    "embed_output",
    # Per-layer checkpoints are added dynamically
    "final_norm",
    "lm_head_logits",
]

# Per-layer checkpoint suffixes
LAYER_CHECKPOINT_SUFFIXES = [
    "input_norm",
    "qkv_proj",
    "rope_q",
    "rope_k",
    "attn_output",
    "o_proj",
    "post_attn",
    "mlp_gate_up",
    "mlp_act",
    "mlp_down",
]


@dataclass
class CheckpointData:
    """Data for a single checkpoint."""
    name: str
    tensor: torch.Tensor  # float32 clone on CPU
    shape: Tuple[int, ...]
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    has_nan: bool
    has_inf: bool


@dataclass
class CheckpointStore:
    """Storage for captured checkpoints."""
    checkpoints: Dict[str, CheckpointData] = field(default_factory=dict)
    source: str = "unknown"  # "engine" or "pytorch"
    num_layers: int = 0
    capture_enabled: bool = CHECKPOINT_DEBUG_ENABLED

    # Only capture last token position for efficiency
    last_token_only: bool = True
    last_pos: Optional[int] = None

    def clear(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.num_layers = 0
        self.last_pos = None

    def set_context(self, source: str, num_layers: int, last_pos: int):
        """Set context for checkpoint capture."""
        self.source = source
        self.num_layers = num_layers
        self.last_pos = last_pos

    def capture(self, name: str, tensor: Any, token_idx: Optional[int] = None) -> None:
        """Capture a checkpoint tensor.

        Args:
            name: Checkpoint name
            tensor: Tensor to capture (torch.Tensor, MTLBuffer ptr, or numpy array)
            token_idx: If set, extract only this token position. If None and
                      last_token_only is True, uses self.last_pos.
        """
        if not self.capture_enabled:
            return

        # Determine which token to capture
        if self.last_token_only:
            if token_idx is None:
                token_idx = self.last_pos
            if token_idx is None:
                # Default to last position
                token_idx = -1

        # Convert to torch tensor if needed
        if isinstance(tensor, torch.Tensor):
            t = tensor
        elif isinstance(tensor, np.ndarray):
            t = torch.from_numpy(tensor)
        else:
            # Assume it's an MTLBuffer contents pointer
            # Need to handle this in the calling code
            raise TypeError(f"Cannot capture tensor of type {type(tensor)}. "
                          f"Convert to torch.Tensor or numpy.ndarray first.")

        # Extract last token if applicable
        if token_idx is not None:
             if t.dim() == 2 or t.dim() == 3:
                 # [tokens, dim] or [tokens, heads, dim]
                 # Slice dim 0
                 slice_idx = token_idx % t.shape[0] if t.shape[0] > 0 else 0 
                 t = t[slice_idx:slice_idx+1]
             elif t.dim() == 4:
                 # [batch, heads, seq, dim] - assume batch=1
                 # Slice dim 2
                 # Note: token_idx might be absolute position in batch?
                 # Assuming batch size 1 for now as per prefill test
                 slice_idx = token_idx % t.shape[2] if t.shape[2] > 0 else 0
                 t = t[:, :, slice_idx:slice_idx+1, :]
                 
                 # Reshape [B, H, S, D] -> [B, S, H, D] -> [B*S, H, D]
                 # Permuted shape: [B, S, H, D]. H is dim 2. D is dim 3.
                 # t refers to original tensor here, so H is dim 1.
                 t = t.permute(0, 2, 1, 3).reshape(-1, t.shape[1], t.shape[3])

        # Convert to float32 CPU for comparison
        t_cpu = t.detach().float().cpu().clone()

        # Compute stats
        t_np = t_cpu.numpy()

        if t_np.size == 0:
             # Handle empty array case gracefully
             ckpt = CheckpointData(
                 name=name,
                 tensor=t_cpu,
                 shape=tuple(t_cpu.shape),
                 dtype=str(tensor.dtype) if hasattr(tensor, 'dtype') else "unknown",
                 min_val=0.0,
                 max_val=0.0,
                 mean_val=0.0,
                 std_val=0.0,
                 has_nan=False,
                 has_inf=False,
             )
        else:
             ckpt = CheckpointData(
                 name=name,
                 tensor=t_cpu,
                 shape=tuple(t_cpu.shape),
                 dtype=str(tensor.dtype) if hasattr(tensor, 'dtype') else "unknown",
                 min_val=float(np.nanmin(t_np)),
                 max_val=float(np.nanmax(t_np)),
                 mean_val=float(np.nanmean(t_np)),
                 std_val=float(np.nanstd(t_np)),
                 has_nan=bool(np.isnan(t_np).any()),
                 has_inf=bool(np.isinf(t_np).any()),
             )

        self.checkpoints[name] = ckpt
        
        # Save to disk if configured
        if DEBUG_DISK_PATH and self.source == "engine": # Only save engine to disk (subprocess)
             try:
                 os.makedirs(DEBUG_DISK_PATH, exist_ok=True)
                 path = os.path.join(DEBUG_DISK_PATH, f"{self.source}_{name}.pt")
                 torch.save(t_cpu, path)
             except Exception as e:
                 print(f"Failed to save checkpoint {name}: {e}")

    def get(self, name: str) -> Optional[CheckpointData]:
        """Get a checkpoint by name."""
        return self.checkpoints.get(name)

    def list_checkpoints(self) -> List[str]:
        """List all captured checkpoint names in order."""
        # Build ordered list based on known order
        ordered = []

        # embed_output first
        if "embed_output" in self.checkpoints:
            ordered.append("embed_output")

        # Layer checkpoints
        for layer_idx in range(self.num_layers):
            for suffix in LAYER_CHECKPOINT_SUFFIXES:
                name = f"layer{layer_idx}_{suffix}"
                if name in self.checkpoints:
                    ordered.append(name)

        # Final checkpoints
        for name in ["final_norm", "lm_head_logits"]:
            if name in self.checkpoints:
                ordered.append(name)

        # Add any unlisted checkpoints at the end
        for name in self.checkpoints:
            if name not in ordered:
                ordered.append(name)

        return ordered

    def summary(self) -> str:
        """Generate summary of captured checkpoints."""
        lines = [
            f"CheckpointStore(source={self.source}, num_layers={self.num_layers})",
            f"Captured {len(self.checkpoints)} checkpoints:",
        ]
        for name in self.list_checkpoints():
            ckpt = self.checkpoints[name]
            status = ""
            if ckpt.has_nan:
                status += " [NaN!]"
            if ckpt.has_inf:
                status += " [Inf!]"
            lines.append(
                f"  {name}: shape={ckpt.shape}, "
                f"range=[{ckpt.min_val:.4f}, {ckpt.max_val:.4f}], "
                f"mean={ckpt.mean_val:.4f}{status}"
            )
        return "\n".join(lines)


# Global stores for engine and pytorch checkpoints
_engine_store: Optional[CheckpointStore] = None
_pytorch_store: Optional[CheckpointStore] = None


def get_checkpoint_store(source: str = "engine") -> CheckpointStore:
    """Get or create checkpoint store for given source."""
    global _engine_store, _pytorch_store

    if source == "engine":
        if _engine_store is None:
            _engine_store = CheckpointStore(source="engine")
        return _engine_store
    elif source == "pytorch":
        if _pytorch_store is None:
            _pytorch_store = CheckpointStore(source="pytorch")
        return _pytorch_store
    else:
        raise ValueError(f"Unknown source: {source}")


def load_from_disk(path: str, source: str = "engine"):
    """Load checkpoints from disk into the specified store."""
    if not os.path.exists(path):
        return
        
    store = get_checkpoint_store(source)
    prefix = f"{source}_"
    
    for filename in os.listdir(path):
        if filename.startswith(prefix) and filename.endswith(".pt"):
            name = filename[len(prefix):-3] # Strip prefix and .pt
            try:
                tensor = torch.load(os.path.join(path, filename))
                store.capture(name, tensor)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")


def capture_checkpoint(name: str, tensor: Any, source: str = "engine",
                       token_idx: Optional[int] = None) -> None:
    """Convenience function to capture a checkpoint.

    Args:
        name: Checkpoint name
        tensor: Tensor to capture
        source: "engine" or "pytorch"
        token_idx: Optional specific token index to capture
    """
    if not CHECKPOINT_DEBUG_ENABLED:
        return
    store = get_checkpoint_store(source)
    store.capture(name, tensor, token_idx)


def compare_checkpoints(
    engine_store: Optional[CheckpointStore] = None,
    pytorch_store: Optional[CheckpointStore] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare checkpoints between engine and pytorch stores.

    Returns:
        Dict mapping checkpoint name to comparison metrics:
            - max_diff: Maximum absolute difference
            - mean_diff: Mean absolute difference
            - cosine_sim: Cosine similarity
            - status: "PASS" or "FAIL"
    """
    if engine_store is None:
        engine_store = get_checkpoint_store("engine")
    if pytorch_store is None:
        pytorch_store = get_checkpoint_store("pytorch")

    results = {}
    first_divergence = None

    # Get ordered list of checkpoints
    all_names = set(engine_store.list_checkpoints()) | set(pytorch_store.list_checkpoints())

    for name in engine_store.list_checkpoints():
        if name not in pytorch_store.checkpoints:
            # Try to synthesize qkv_out from separate q/k/v
            if name.endswith("_qkv_out"):
                prefix = name[:-8] # remove _qkv_out
                q_name = f"{prefix}_q_proj"
                k_name = f"{prefix}_k_proj"
                v_name = f"{prefix}_v_proj"
                
                if (q_name in pytorch_store.checkpoints and 
                    k_name in pytorch_store.checkpoints and 
                    v_name in pytorch_store.checkpoints):
                    
                    # Synthesize!
                    q = pytorch_store.get(q_name).tensor
                    k = pytorch_store.get(k_name).tensor
                    v = pytorch_store.get(v_name).tensor
                    
                    # Concat along last dim
                    # Only float16/float32.
                    fused = torch.cat([q, k, v], dim=-1)
                    pytorch_store.capture(name, fused)
                    
            # Re-check existence
            if name not in pytorch_store.checkpoints:
                results[name] = {
                    "max_diff": float('inf'),
                    "mean_diff": float('inf'),
                    "cosine_sim": 0.0,
                    "status": "MISSING_PYTORCH",
                }
                if first_divergence is None:
                    first_divergence = name
                continue

        engine_ckpt = engine_store.get(name)
        pytorch_ckpt = pytorch_store.get(name)

        # Compare shapes
        if engine_ckpt.shape != pytorch_ckpt.shape:
            print(f"DEBUG: SHAPE_MISMATCH detail for {name}: Engine {engine_ckpt.shape} vs PyTorch {pytorch_ckpt.shape}") # Explicit print
            results[name] = {
                "max_diff": float('inf'),
                "mean_diff": float('inf'),
                "cosine_sim": 0.0,
                "status": f"SHAPE_MISMATCH: {engine_ckpt.shape} vs {pytorch_ckpt.shape}",
            }
            if first_divergence is None:
                first_divergence = name
            continue

        # Compute differences
        e_np = engine_ckpt.tensor.numpy().flatten()
        p_np = pytorch_ckpt.tensor.numpy().flatten()

        if "rope_q_in" in name and "layer0" in name:
             print(f"DEBUG: {name} Engine[:20]: {e_np[:20]}")
             print(f"DEBUG: {name} PyTorch[:20]: {p_np[:20]}")
             # Check for common scaling factor?
             if np.sum(np.abs(p_np)) > 0:
                 ratio = np.mean(np.abs(e_np)) / np.mean(np.abs(p_np))
                 print(f"DEBUG: {name} Mean Ratio E/P: {ratio:.4f}")

        abs_diff = np.abs(e_np - p_np)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))

        # Cosine similarity
        e_norm = np.linalg.norm(e_np)
        p_norm = np.linalg.norm(p_np)
        if e_norm > 0 and p_norm > 0:
            cosine_sim = float(np.dot(e_np, p_np) / (e_norm * p_norm))
        else:
            cosine_sim = 0.0

        # Determine status (initial thresholds)
        if max_diff < 1e-3 and mean_diff < 1e-5:
            status = "PASS"
        elif max_diff < 1e-2 and mean_diff < 1e-4:
            status = "WARN"
        else:
            status = "FAIL"
            if first_divergence is None:
                first_divergence = name

        results[name] = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "cosine_sim": cosine_sim,
            "status": status,
        }

    return results, first_divergence


def print_comparison_report(
    results: Dict[str, Dict[str, float]],
    first_divergence: Optional[str],
) -> str:
    """Generate human-readable comparison report."""
    lines = [
        "="*80,
        "CHECKPOINT COMPARISON REPORT",
        "="*80,
        "",
        f"{'Checkpoint':<30} {'max|diff|':>12} {'mean|diff|':>12} {'cos_sim':>10} {'Status':>10}",
        "-"*80,
    ]

    for name, metrics in results.items():
        max_diff = metrics["max_diff"]
        mean_diff = metrics["mean_diff"]
        cos_sim = metrics["cosine_sim"]
        status = metrics["status"]

        # Format values
        if max_diff == float('inf'):
            max_str = "inf"
            mean_str = "inf"
        else:
            max_str = f"{max_diff:.2e}"
            mean_str = f"{mean_diff:.2e}"
        cos_str = f"{cos_sim:.6f}"

        lines.append(f"{name:<30} {max_str:>12} {mean_str:>12} {cos_str:>10} {status:>10}")

    lines.append("-"*80)

    if first_divergence:
        lines.append(f"\nFIRST DIVERGENCE: {first_divergence}")
    else:
        lines.append("\nNo divergence detected - all checkpoints PASS")

    lines.append("="*80)
    return "\n".join(lines)


def reset_stores():
    """Reset both checkpoint stores."""
    global _engine_store, _pytorch_store
    if _engine_store:
        _engine_store.clear()
    if _pytorch_store:
        _pytorch_store.clear()
