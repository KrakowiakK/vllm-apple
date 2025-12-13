# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight Loader for vLLM-Apple Metal Engine v2.0.

This module provides weight loading from HuggingFace models to MTLBuffer.
Weights are loaded at initialization time (not in hot path).

Key Principle: Load weights to MTLBuffer once at startup.

Usage:
    from vllm_apple.engine.weight_loader import EngineWeightLoader

    # Create loader
    loader = EngineWeightLoader(context, model_config)

    # Load weights from HuggingFace model
    weights = loader.load_from_hf_model(hf_model)

    # Or load from state dict
    weights = loader.load_from_state_dict(state_dict)

    # Access loaded weights
    embedding_weights = weights.embedding
    layer_weights = weights.layers[0]  # First layer
    lm_head_weights = weights.lm_head
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vllm.logger import init_logger
from .tensor import EngineDType

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLResourceStorageModeShared
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


@dataclass
class TransformerLayerWeights:
    """Weights for a single transformer layer."""
    # Attention
    q_proj: Any = None  # [num_heads * head_size, hidden_size]
    k_proj: Any = None  # [num_kv_heads * head_size, hidden_size]
    v_proj: Any = None  # [num_kv_heads * head_size, hidden_size]
    o_proj: Any = None  # [hidden_size, num_heads * head_size]

    # Fused QKV (alternative to separate)
    qkv_proj: Any = None  # [(num_heads + 2*num_kv_heads) * head_size, hidden_size]

    # Attention biases (GPT-2)
    qkv_bias: Any = None
    o_proj_bias: Any = None

    # Norms
    input_layernorm: Any = None   # [hidden_size]
    post_attention_layernorm: Any = None  # [hidden_size]

    # Norm biases (GPT-2 uses LayerNorm with bias)
    input_layernorm_bias: Any = None
    post_attention_layernorm_bias: Any = None

    # MLP
    gate_proj: Any = None  # [intermediate_size, hidden_size]
    up_proj: Any = None    # [intermediate_size, hidden_size]
    down_proj: Any = None  # [hidden_size, intermediate_size]

    # MLP biases (GPT-2)
    up_proj_bias: Any = None
    down_proj_bias: Any = None

    # Fused gate-up (alternative)
    gate_up_proj: Any = None  # [2 * intermediate_size, hidden_size]


@dataclass
class ModelWeights:
    """Complete model weights in MTLBuffer format."""
    # Token embedding
    embedding: Any = None  # [vocab_size, hidden_size]

    # Position embedding (GPT-2)
    pos_embedding: Any = None  # [max_pos, hidden_size]

    # Transformer layers
    layers: List[TransformerLayerWeights] = field(default_factory=list)

    # Final norm
    final_norm: Any = None  # [hidden_size]
    final_norm_bias: Any = None  # [hidden_size] (GPT-2)

    # LM head
    lm_head: Any = None  # [vocab_size, hidden_size]

    # Metadata
    num_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    intermediate_size: int = 0
    head_size: int = 0


class EngineWeightLoader:
    """Loads model weights from PyTorch/HuggingFace to MTLBuffer.

    This loader handles:
    - Converting PyTorch tensors to MTLBuffer
    - Proper memory layout for Metal operations
    - Fusing QKV projections if separate
    - Weight transposition where needed

    Attributes:
        context: MetalEngineContext
        model_config: Model configuration
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        model_config: Optional[Any] = None,  # vLLM ModelConfig
    ):
        """Initialize weight loader.

        Args:
            context: MetalEngineContext for buffer allocation
            model_config: Optional model configuration
        """
        self._context = context
        self._model_config = model_config

        # Weight name mappings for different model architectures
        self._arch_mappings = {
            "qwen2": self._get_qwen2_mapping(),
            "llama": self._get_llama_mapping(),
            "gpt2": self._get_gpt2_mapping(),
        }

        logger.info("EngineWeightLoader initialized")

    def _get_qwen2_mapping(self) -> Dict[str, str]:
        """Get weight name mapping for Qwen2 architecture."""
        return {
            "model.embed_tokens.weight": "embedding",
            "model.norm.weight": "final_norm",
            "lm_head.weight": "lm_head",
            # Layer patterns
            "model.layers.{}.self_attn.q_proj.weight": "q_proj",
            "model.layers.{}.self_attn.k_proj.weight": "k_proj",
            "model.layers.{}.self_attn.v_proj.weight": "v_proj",
            "model.layers.{}.self_attn.o_proj.weight": "o_proj",
            "model.layers.{}.input_layernorm.weight": "input_layernorm",
            "model.layers.{}.post_attention_layernorm.weight": "post_attention_layernorm",
            "model.layers.{}.mlp.gate_proj.weight": "gate_proj",
            "model.layers.{}.mlp.up_proj.weight": "up_proj",
            "model.layers.{}.mlp.down_proj.weight": "down_proj",
        }

    def _get_llama_mapping(self) -> Dict[str, str]:
        """Get weight name mapping for LLaMA architecture."""
        return self._get_qwen2_mapping()  # Same structure

    def _get_gpt2_mapping(self) -> Dict[str, str]:
        """Get weight name mapping for GPT-2 architecture."""
        return {
            "transformer.wte.weight": "embedding",
            "transformer.wpe.weight": "pos_embedding",  # GPT-2 has position embeddings
            "transformer.ln_f.weight": "final_norm",
            "transformer.ln_f.bias": "final_norm_bias",
            "lm_head.weight": "lm_head",  # May be tied to wte
            # Layer patterns
            "transformer.h.{}.attn.c_attn.weight": "c_attn",  # Fused QKV
            "transformer.h.{}.attn.c_attn.bias": "c_attn_bias",
            "transformer.h.{}.attn.c_proj.weight": "o_proj",
            "transformer.h.{}.attn.c_proj.bias": "o_proj_bias",
            "transformer.h.{}.ln_1.weight": "input_layernorm",
            "transformer.h.{}.ln_1.bias": "input_layernorm_bias",
            "transformer.h.{}.ln_2.weight": "post_attention_layernorm",
            "transformer.h.{}.ln_2.bias": "post_attention_layernorm_bias",
            "transformer.h.{}.mlp.c_fc.weight": "up_proj",  # GPT-2 MLP
            "transformer.h.{}.mlp.c_fc.bias": "up_proj_bias",
            "transformer.h.{}.mlp.c_proj.weight": "down_proj",
            "transformer.h.{}.mlp.c_proj.bias": "down_proj_bias",
        }

    def _tensor_to_mtlbuffer(
        self,
        tensor: Any,  # torch.Tensor
        name: str = "weight",
    ) -> Any:
        """Convert PyTorch tensor to MTLBuffer.

        Args:
            tensor: PyTorch tensor (any device)
            name: Name for debugging

        Returns:
            MTLBuffer containing the tensor data
        """
        import torch

        # Ensure contiguous and on CPU
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.contiguous()

        # Convert to float16 if needed
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)

        # Get numpy array
        np_array = tensor.numpy()

        # Ensure contiguous C-order array for correct memory layout
        np_array = np.ascontiguousarray(np_array)
        size_bytes = np_array.nbytes

        # Create MTLBuffer with bytes directly - this is more reliable than
        # trying to copy into an existing buffer with PyObjC
        buffer = self._context.device.newBufferWithBytes_length_options_(
            np_array.tobytes(),
            size_bytes,
            MTLResourceStorageModeShared,
        )

        if buffer is None:
            raise RuntimeError(f"Failed to allocate MTLBuffer for {name} ({size_bytes} bytes)")

        logger.debug(f"Loaded weight {name}: shape={tensor.shape}, size={size_bytes / 1024 / 1024:.2f}MB")
        return buffer

    def load_from_hf_model(
        self,
        hf_model: Any,  # nn.Module
        arch: str = "qwen2",
    ) -> ModelWeights:
        """Load weights from a HuggingFace model.

        Args:
            hf_model: HuggingFace model (nn.Module)
            arch: Model architecture ("qwen2", "llama")

        Returns:
            ModelWeights with all weights loaded to MTLBuffer
        """
        state_dict = hf_model.state_dict()
        return self.load_from_state_dict(state_dict, arch)

    def load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        arch: str = "qwen2",
    ) -> ModelWeights:
        """Load weights from a state dict.

        Args:
            state_dict: PyTorch state dict
            arch: Model architecture

        Returns:
            ModelWeights with all weights loaded to MTLBuffer
        """
        if arch == "gpt2":
            return self._load_gpt2_weights(state_dict)
        else:
            return self._load_llama_weights(state_dict)

    def _load_llama_weights(self, state_dict: Dict[str, Any]) -> ModelWeights:
        """Load weights for Llama/Qwen2 architecture."""
        weights = ModelWeights()

        # Detect number of layers
        num_layers = 0
        for key in state_dict.keys():
            if "layers." in key:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            num_layers = max(num_layers, layer_idx + 1)
                        except ValueError:
                            pass

        weights.num_layers = num_layers
        weights.layers = [TransformerLayerWeights() for _ in range(num_layers)]

        # Load non-layer weights
        for key, tensor in state_dict.items():
            if "layers" not in key:
                if "embed_tokens" in key:
                    weights.embedding = self._tensor_to_mtlbuffer(tensor, "embedding")
                    weights.vocab_size = tensor.shape[0]
                    weights.hidden_size = tensor.shape[1]
                elif "norm.weight" in key and "layernorm" not in key:
                    weights.final_norm = self._tensor_to_mtlbuffer(tensor, "final_norm")
                elif "lm_head" in key:
                    weights.lm_head = self._tensor_to_mtlbuffer(tensor, "lm_head")

        # Load layer weights
        for layer_idx in range(num_layers):
            layer = weights.layers[layer_idx]
            prefix = f"model.layers.{layer_idx}."

            for key, tensor in state_dict.items():
                if not key.startswith(prefix):
                    continue

                suffix = key[len(prefix):]

                if "q_proj.weight" in suffix:
                    layer.q_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.q_proj")
                    weights.num_attention_heads = tensor.shape[0] // (weights.hidden_size // tensor.shape[0] if tensor.shape[0] > 0 else 1)
                elif "k_proj.weight" in suffix:
                    layer.k_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.k_proj")
                elif "v_proj.weight" in suffix:
                    layer.v_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.v_proj")
                elif "o_proj.weight" in suffix:
                    layer.o_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.o_proj")
                elif "input_layernorm.weight" in suffix:
                    layer.input_layernorm = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.input_ln")
                elif "post_attention_layernorm.weight" in suffix:
                    layer.post_attention_layernorm = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.post_attn_ln")
                elif "gate_proj.weight" in suffix:
                    layer.gate_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.gate_proj")
                    weights.intermediate_size = tensor.shape[0]
                elif "up_proj.weight" in suffix:
                    layer.up_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.up_proj")
                elif "down_proj.weight" in suffix:
                    layer.down_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.down_proj")

        # Infer remaining config
        if weights.hidden_size > 0 and weights.num_attention_heads > 0:
            weights.head_size = weights.hidden_size // weights.num_attention_heads

        # Check for tied embeddings
        if weights.lm_head is None and weights.embedding is not None:
            logger.info("LM head tied to embedding weights")
            weights.lm_head = weights.embedding

        logger.info(
            f"Loaded model weights: {weights.num_layers} layers, "
            f"hidden_size={weights.hidden_size}, vocab_size={weights.vocab_size}, "
            f"intermediate_size={weights.intermediate_size}"
        )

        return weights

    def _load_gpt2_weights(self, state_dict: Dict[str, Any]) -> ModelWeights:
        """Load weights for GPT-2 architecture."""
        weights = ModelWeights()

        # Detect number of layers from transformer.h.X pattern
        num_layers = 0
        for key in state_dict.keys():
            if "transformer.h." in key:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "h" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            num_layers = max(num_layers, layer_idx + 1)
                        except ValueError:
                            pass

        weights.num_layers = num_layers
        weights.layers = [TransformerLayerWeights() for _ in range(num_layers)]

        # Load non-layer weights
        for key, tensor in state_dict.items():
            if "transformer.h." not in key:
                if key == "transformer.wte.weight":
                    weights.embedding = self._tensor_to_mtlbuffer(tensor, "embedding")
                    weights.vocab_size = tensor.shape[0]
                    weights.hidden_size = tensor.shape[1]
                elif key == "transformer.wpe.weight":
                    # Position embeddings - store separately
                    weights.pos_embedding = self._tensor_to_mtlbuffer(tensor, "pos_embedding")
                elif key == "transformer.ln_f.weight":
                    weights.final_norm = self._tensor_to_mtlbuffer(tensor, "final_norm")
                elif key == "transformer.ln_f.bias":
                    weights.final_norm_bias = self._tensor_to_mtlbuffer(tensor, "final_norm_bias")
                elif key == "lm_head.weight":
                    weights.lm_head = self._tensor_to_mtlbuffer(tensor, "lm_head")

        # Load layer weights
        for layer_idx in range(num_layers):
            layer = weights.layers[layer_idx]
            prefix = f"transformer.h.{layer_idx}."

            for key, tensor in state_dict.items():
                if not key.startswith(prefix):
                    continue

                suffix = key[len(prefix):]

                if suffix == "attn.c_attn.weight":
                    # GPT-2 has fused QKV as c_attn [3*hidden, hidden]
                    # Store as qkv_proj (fused)
                    layer.qkv_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.qkv_proj")
                    # Also extract Q, K, V for compatibility
                    hidden = weights.hidden_size
                    layer.q_proj = self._tensor_to_mtlbuffer(tensor[:hidden, :], f"layer{layer_idx}.q_proj")
                    layer.k_proj = self._tensor_to_mtlbuffer(tensor[hidden:2*hidden, :], f"layer{layer_idx}.k_proj")
                    layer.v_proj = self._tensor_to_mtlbuffer(tensor[2*hidden:, :], f"layer{layer_idx}.v_proj")
                elif suffix == "attn.c_attn.bias":
                    layer.qkv_bias = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.qkv_bias")
                elif suffix == "attn.c_proj.weight":
                    layer.o_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.o_proj")
                elif suffix == "attn.c_proj.bias":
                    layer.o_proj_bias = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.o_proj_bias")
                elif suffix == "ln_1.weight":
                    layer.input_layernorm = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.input_ln")
                elif suffix == "ln_1.bias":
                    layer.input_layernorm_bias = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.input_ln_bias")
                elif suffix == "ln_2.weight":
                    layer.post_attention_layernorm = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.post_attn_ln")
                elif suffix == "ln_2.bias":
                    layer.post_attention_layernorm_bias = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.post_attn_ln_bias")
                elif suffix == "mlp.c_fc.weight":
                    # GPT-2 MLP: up projection
                    layer.up_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.up_proj")
                    weights.intermediate_size = tensor.shape[0]
                elif suffix == "mlp.c_fc.bias":
                    layer.up_proj_bias = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.up_proj_bias")
                elif suffix == "mlp.c_proj.weight":
                    # GPT-2 MLP: down projection
                    layer.down_proj = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.down_proj")
                elif suffix == "mlp.c_proj.bias":
                    layer.down_proj_bias = self._tensor_to_mtlbuffer(tensor, f"layer{layer_idx}.down_proj_bias")

        # GPT-2 uses same num heads as hidden/64
        if weights.hidden_size > 0:
            weights.num_attention_heads = weights.hidden_size // 64  # GPT-2 head size is 64
            weights.head_size = 64

        # Check for tied embeddings (common in GPT-2)
        if weights.lm_head is None and weights.embedding is not None:
            logger.info("LM head tied to embedding weights")
            weights.lm_head = weights.embedding

        logger.info(
            f"Loaded model weights: {num_layers} layers, "
            f"hidden_size={weights.hidden_size}, vocab_size={weights.vocab_size}, "
            f"intermediate_size={weights.intermediate_size}"
        )

        return weights

    def fuse_qkv_weights(
        self,
        q_proj: Any,  # MTLBuffer
        k_proj: Any,
        v_proj: Any,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Any:
        """Fuse separate Q, K, V projections into single QKV.

        This can improve efficiency by reducing buffer bindings.

        Args:
            q_proj: Q projection [num_heads * head_size, hidden_size]
            k_proj: K projection [num_kv_heads * head_size, hidden_size]
            v_proj: V projection [num_kv_heads * head_size, hidden_size]
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads
            head_size: Head dimension

        Returns:
            Fused QKV buffer [(num_heads + 2*num_kv_heads) * head_size, hidden_size]
        """
        # For now, return None - keep separate projections
        # TODO: Implement QKV fusion for efficiency
        return None

    def get_total_size_bytes(self, weights: ModelWeights) -> int:
        """Calculate total weight buffer size.

        Args:
            weights: Loaded model weights

        Returns:
            Total size in bytes
        """
        total = 0

        # Count buffer sizes using a helper
        def add_buffer_size(buf):
            nonlocal total
            if buf is not None:
                total += buf.length()

        add_buffer_size(weights.embedding)
        add_buffer_size(weights.final_norm)
        add_buffer_size(weights.lm_head)

        for layer in weights.layers:
            add_buffer_size(layer.q_proj)
            add_buffer_size(layer.k_proj)
            add_buffer_size(layer.v_proj)
            add_buffer_size(layer.o_proj)
            add_buffer_size(layer.qkv_proj)
            add_buffer_size(layer.input_layernorm)
            add_buffer_size(layer.post_attention_layernorm)
            add_buffer_size(layer.gate_proj)
            add_buffer_size(layer.up_proj)
            add_buffer_size(layer.down_proj)
            add_buffer_size(layer.gate_up_proj)

        return total

    def get_stats(self, weights: ModelWeights) -> Dict[str, Any]:
        """Get weight loading statistics.

        Args:
            weights: Loaded model weights

        Returns:
            Dictionary of statistics
        """
        total_bytes = self.get_total_size_bytes(weights)
        return {
            "num_layers": weights.num_layers,
            "hidden_size": weights.hidden_size,
            "vocab_size": weights.vocab_size,
            "intermediate_size": weights.intermediate_size,
            "head_size": weights.head_size,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "total_gb": total_bytes / (1024 * 1024 * 1024),
        }
