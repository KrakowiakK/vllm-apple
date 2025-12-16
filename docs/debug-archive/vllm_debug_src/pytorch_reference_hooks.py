# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference hooks for capturing HuggingFace model activations.

This module provides forward hooks to capture intermediate tensors from
HuggingFace transformer models for comparison with Metal engine outputs.

Usage:
    from vllm_apple.debug.pytorch_reference_hooks import attach_reference_hooks

    # Attach hooks to HF model
    model = AutoModelForCausalLM.from_pretrained(...)
    hooks = attach_reference_hooks(model, num_layers=32)

    # Run forward pass
    outputs = model(input_ids)

    # Checkpoints are now in the pytorch store
    from vllm_apple.debug import get_checkpoint_store
    store = get_checkpoint_store("pytorch")

Note:
    This module supports multiple model architectures:
    - Mistral/Devstral (model.layers[i].self_attn, model.layers[i].mlp)
    - Llama (similar structure)
    - Qwen2 (similar structure)

    The hooks are attached based on module type detection, not hardcoded paths.
"""

import os
from typing import Dict, List, Optional, Any, Callable, Tuple
import torch
import torch.nn as nn

from .prefill_checkpoint import (
    get_checkpoint_store,
    CheckpointStore,
    CHECKPOINT_DEBUG_ENABLED,
)


# Module type patterns for different architectures
# Maps (checkpoint_suffix, module_attr_patterns)
MODULE_PATTERNS = {
    # Mistral/Devstral patterns
    "mistral": {
        "embed": ["model.embed_tokens"],
        "input_norm": ["input_layernorm"],
        "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "rope": ["self_attn.rotary_emb"],
        "attn_output": ["self_attn"],  # Capture full attention output
        "o_proj": ["self_attn.o_proj"],
        "post_attn_norm": ["post_attention_layernorm"],
        "mlp_gate_up": ["mlp.gate_proj", "mlp.up_proj"],
        "mlp_down": ["mlp.down_proj"],
        "final_norm": ["model.norm"],
        "lm_head": ["lm_head"],
    },
    # Llama patterns (very similar to Mistral)
    "llama": {
        "embed": ["model.embed_tokens"],
        "input_norm": ["input_layernorm"],
        "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "rope": ["self_attn.rotary_emb"],
        "attn_output": ["self_attn"],
        "o_proj": ["self_attn.o_proj"],
        "post_attn_norm": ["post_attention_layernorm"],
        "mlp_gate_up": ["mlp.gate_proj", "mlp.up_proj"],
        "mlp_down": ["mlp.down_proj"],
        "final_norm": ["model.norm"],
        "lm_head": ["lm_head"],
    },
    "qwen": {
        "embed": ["model.embed_tokens"],
        "input_norm": ["input_layernorm"],
        "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "rope": ["self_attn.rotary_emb"],
        "attn_output": ["self_attn"],
        "o_proj": ["self_attn.o_proj"],
        "post_attn_norm": ["post_attention_layernorm"],
        "mlp_gate_up": ["mlp.gate_proj", "mlp.up_proj"],
        "mlp_down": ["mlp.down_proj"],
        "final_norm": ["model.norm"],
        "lm_head": ["lm_head"],
    },
}


class ReferenceHookManager:
    """Manages forward hooks for capturing reference activations."""

    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        last_pos: int,
        architecture: str = "mistral",
    ):
        self.model = model
        self.num_layers = num_layers
        self.last_pos = last_pos
        self.architecture = architecture
        self.hooks: List[Any] = []
        self.store = get_checkpoint_store("pytorch")

        # Set up store context
        self.store.set_context("pytorch", num_layers, last_pos)

    def _create_hook(
        self,
        module: nn.Module,
        checkpoint_name: str,
        output_index: Optional[int] = None,
    ) -> Callable:
        """Create a forward hook that captures output to checkpoint store.

        Args:
            module: Module to attach hook to
            checkpoint_name: Name for the checkpoint
            output_index: If output is a tuple, which index to capture (None = all)
        """
        def hook(module, input, output):
            if not CHECKPOINT_DEBUG_ENABLED:
                return

            # Handle tuple outputs (common in transformers)
            if isinstance(output, tuple):
                try:
                    if output_index is not None:
                        tensor = output[output_index]
                    else:
                        tensor = output[0]  # Default to first element
                except Exception as e:
                    print(f"FAILED to capture output for checkpoint '{checkpoint_name}' at index {output_index}: {e}")
                    return # Skip capturing if output access fails
            else:
                tensor = output

            # Capture checkpoint
            if isinstance(tensor, torch.Tensor):
                # Handle batch dimension from HF (usually [1, seq, hidden])
                if tensor.dim() == 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                self.store.capture(checkpoint_name, tensor, token_idx=self.last_pos)

        self.hooks.append(module.register_forward_hook(hook)) # Register hook here
        return hook # Return the hook function itself, not the handle

    def _find_module(self, path: str) -> Optional[nn.Module]:
        """Find module by dot-separated path."""
        parts = path.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _attach_embedding_hook(self) -> None:
        """Attach hook to embedding layer."""
        patterns = MODULE_PATTERNS.get(self.architecture, {}).get("embed", [])
        for pattern in patterns:
            module = self._find_module(pattern)
            if module is not None:
                self._create_hook(module, "embed_output") # Adjusted call
                return

    def _attach_layer_hooks(self, layer_idx: int, layer_module: nn.Module) -> None:
        """Attach hooks to a single transformer layer."""
        prefix = f"layer{layer_idx}"

        # Input norm (captures output of input_layernorm)
        if hasattr(layer_module, "input_layernorm"):
            self._create_hook(layer_module.input_layernorm, f"{prefix}_input_norm")

        # Self-attention
        if hasattr(layer_module, "self_attn"):
            attn = layer_module.self_attn

            # Q, K, V projections (before RoPE)
            if hasattr(attn, "q_proj"):
                self._create_hook(attn.q_proj, f"{prefix}_q_proj")
            if hasattr(attn, "k_proj"):
                self._create_hook(attn.k_proj, f"{prefix}_k_proj")
            if hasattr(attn, "v_proj"):
                self._create_hook(attn.v_proj, f"{prefix}_v_proj")

            # Full attention output (index 0 of tuple check inside hook or assume tensor?)
            # HF attention returns (attn_output, attn_weights, past_key_values)
            self._create_hook(attn, f"{prefix}_attn_output", output_index=0)

            # O projection
            if hasattr(attn, "o_proj"):
                self._create_hook(attn.o_proj, f"{prefix}_o_proj")

        # Post-attention norm
        if hasattr(layer_module, "post_attention_layernorm"):
            self._create_hook(layer_module.post_attention_layernorm, f"{prefix}_post_attn")

        # MLP
        if hasattr(layer_module, "mlp"):
            mlp = layer_module.mlp

            # Gate projection
            if hasattr(mlp, "gate_proj"):
                self._create_hook(mlp.gate_proj, f"{prefix}_mlp_gate")

            # Up projection
            if hasattr(mlp, "up_proj"):
                self._create_hook(mlp.up_proj, f"{prefix}_mlp_up")

            # Down projection (final MLP output before residual)
            if hasattr(mlp, "down_proj"):
                self._create_hook(mlp.down_proj, f"{prefix}_mlp_down")


    def _attach_final_hooks(self) -> None:
        """Attach hooks to final norm and lm_head."""
        # Final norm
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            self._create_hook(self.model.model.norm, "final_norm")
        elif hasattr(self.model, "norm"):
             self._create_hook(self.model.norm, "final_norm")

        # LM head
        if hasattr(self.model, "lm_head"):
            self._create_hook(self.model.lm_head, "lm_head_logits")


    def attach_all(self) -> "ReferenceHookManager":
        """Attach all hooks to the model."""
        # DEBUG: Print all registered modules
        print("\nDEBUG: Registered Modules in HF Model:")
        for name, _ in self.model.named_modules():
             print(f"  {name}")
        print("DEBUG: End Registered Modules\n")

        if not CHECKPOINT_DEBUG_ENABLED:
            return self

        # Clear any existing hooks
        self.remove_all()
        self.store.clear()
        self.store.set_context("pytorch", self.num_layers, self.last_pos)

        # MONKEY PATCH: Capture RoPE outputs
        # This is needed because apply_rotary_pos_emb is functional
        try:
             import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
             original_apply_rope = qwen2_mod.apply_rotary_pos_emb
             
             # Use a closure to capture 'self' (manager)
             # We use a simple counter attached to the function wrapper 
             # to track invocations.
             # Reset counter on attach
             self._rope_call_count = 0
             
             def patched_apply_rope_wrapper(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                 # Run original
                 q_pe, k_pe = original_apply_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
                
                 # Capture logic
                 layer_idx = self._rope_call_count
                 if layer_idx < self.num_layers:
                     self.store.capture(f"layer{layer_idx}_rope_q_in", q.detach().clone(), token_idx=None)
                     self.store.capture(f"layer{layer_idx}_rope_k_in", k.detach().clone(), token_idx=None)
                     self.store.capture(f"layer{layer_idx}_rope_q", q_pe.detach().clone(), token_idx=None)
                     self.store.capture(f"layer{layer_idx}_rope_k", k_pe.detach().clone(), token_idx=None)
                 
                 self._rope_call_count += 1
                 return q_pe, k_pe
             
             # Bind the method to 'self' so we can access store? 
             # No, 'self' is in closure.
             # We assume strict single-threaded execution during this test.
             qwen2_mod.apply_rotary_pos_emb = patched_apply_rope_wrapper
             self._original_rope_fn = original_apply_rope # Save to restore later if needed
             # print(f"DEBUG: Monkey-patched Qwen2 apply_rotary_pos_emb")
             
        except ImportError:
             pass 

        # Embedding
        self._attach_embedding_hook()

        # Layers
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        if layers is not None:
            for layer_idx, layer_module in enumerate(layers):
                if layer_idx >= self.num_layers:
                    break
                self._attach_layer_hooks(layer_idx, layer_module)

        # Final hooks
        self._attach_final_hooks()

        return self

    def remove_all(self) -> None:
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def attach_reference_hooks(
    model: nn.Module,
    num_layers: int,
    last_pos: int,
    architecture: str = "mistral",
) -> ReferenceHookManager:
    """Attach reference hooks to a HuggingFace model.

    Args:
        model: HuggingFace model instance
        num_layers: Number of transformer layers
        last_pos: Last token position to capture
        architecture: Model architecture ("mistral", "llama")

    Returns:
        ReferenceHookManager instance (call .remove_all() when done)
    """
    manager = ReferenceHookManager(model, num_layers, last_pos, architecture)
    return manager.attach_all()


def detect_architecture(model: nn.Module) -> str:
    """Detect model architecture from module names."""
    model_type = type(model).__name__.lower()

    if "mistral" in model_type or "devstral" in model_type:
        return "mistral"
    elif "llama" in model_type:
        return "llama"
    elif "qwen" in model_type:
        return "llama"  # Qwen2 uses similar structure
    else:
        # Default to mistral-style
        return "mistral"


def get_num_layers(model: nn.Module) -> int:
    """Get number of transformer layers from model."""
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "layers"):
        return len(model.layers)
    else:
        raise ValueError("Cannot determine number of layers from model")
