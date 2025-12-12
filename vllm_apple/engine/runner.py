# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Runner for vLLM-Apple Metal Engine v2.0.

This module provides the main execution engine for transformer models.
It encodes a complete forward pass to a single command buffer and executes
with step-boundary-only synchronization.

Key Principle: One command buffer per step, one wait at the end.

The EngineRunner implements the three-phase execution model:
1. ENCODE: Build command buffer with all operations
2. SUBMIT: Commit command buffer to GPU
3. READBACK: Wait and read outputs

Usage:
    from vllm_apple.engine.runner import EngineRunner

    # Create runner
    runner = EngineRunner(context, model_desc, weights)

    # Execute step
    outputs = runner.execute_step(step_desc, inputs)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import struct
import ctypes

import torch
import numpy as np

from vllm.logger import init_logger
from .context import MetalEngineContext
from .step import EngineStepContext
from .tensor import EngineTensor, EngineDType
from .descriptors import (
    StepDescriptor,
    EngineInputs,
    EngineOutputs,
    ModelDescriptor,
    KVCacheDescriptor,
)
from .weight_loader import ModelWeights, TransformerLayerWeights
from .kv_cache import EngineKVCache
from .ops import (
    EngineEmbedding,
    EngineRMSNorm,
    EngineQKVProjection,
    EngineOProjection,
    EngineRoPE,
    EngineElementwiseOps,
    EngineMLP,
    EngineLMHead,
    PagedAttentionOp,
    KVWriteOp,
)

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLResourceStorageModeShared
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


@dataclass
class TransformerLayerOps:
    """Operations for a single transformer layer."""
    input_norm: EngineRMSNorm
    qkv_proj: EngineQKVProjection
    rope: EngineRoPE
    attention: PagedAttentionOp
    kv_write: KVWriteOp
    o_proj: EngineOProjection
    post_attn_norm: EngineRMSNorm
    mlp: EngineMLP


class EngineRunner:
    """Main execution engine for transformer models.

    This class coordinates all engine operations to execute a complete
    forward pass with step-boundary-only synchronization.

    The runner:
    1. Creates all operation instances at initialization
    2. Binds weights to operations
    3. Executes forward passes with single command buffer

    Attributes:
        context: MetalEngineContext
        model_desc: ModelDescriptor with model configuration
        weights: ModelWeights loaded to MTLBuffer
        kv_cache: EngineKVCache for KV storage
    """

    def __init__(
        self,
        context: MetalEngineContext,
        model_desc: ModelDescriptor,
        weights: ModelWeights,
        kv_cache: EngineKVCache,
    ):
        """Initialize engine runner.

        Args:
            context: MetalEngineContext
            model_desc: Model configuration
            weights: Loaded model weights
            kv_cache: KV cache instance
        """
        self._context = context
        self.model_desc = model_desc
        self._weights = weights
        self._kv_cache = kv_cache

        # Step counter
        self._step_counter = 0

        # Create operations
        self._create_operations()

        # Bind weights to operations
        self._bind_weights()

        # Pre-allocate scratch buffers for intermediate results
        self._allocate_scratch()

        logger.info(
            f"EngineRunner initialized: {model_desc.num_layers} layers, "
            f"hidden_size={model_desc.hidden_size}, vocab_size={model_desc.vocab_size}"
        )

    def _create_operations(self) -> None:
        """Create all operation instances."""
        md = self.model_desc

        # Embedding
        self._embedding = EngineEmbedding(
            context=self._context,
            vocab_size=md.vocab_size,
            hidden_size=md.hidden_size,
        )

        # Create layer operations
        self._layers: List[TransformerLayerOps] = []
        for layer_idx in range(md.num_layers):
            layer_ops = TransformerLayerOps(
                input_norm=EngineRMSNorm(
                    context=self._context,
                    hidden_size=md.hidden_size,
                ),
                qkv_proj=EngineQKVProjection(
                    context=self._context,
                    hidden_size=md.hidden_size,
                    num_heads=md.num_attention_heads,
                    num_kv_heads=md.num_kv_heads,
                    head_size=md.head_size,
                ),
                rope=EngineRoPE(
                    context=self._context,
                    head_size=md.head_size,
                    max_seq_len=md.max_position_embeddings,
                    rope_theta=md.rope_theta,
                ),
                attention=PagedAttentionOp(
                    context=self._context,
                    num_heads=md.num_attention_heads,
                    num_kv_heads=md.num_kv_heads,
                    head_size=md.head_size,
                    scale=1.0 / (md.head_size ** 0.5),
                ),
                kv_write=KVWriteOp(
                    context=self._context,
                    num_kv_heads=md.num_kv_heads,
                    head_size=md.head_size,
                    block_size=self._kv_cache.block_size,
                ),
                o_proj=EngineOProjection(
                    context=self._context,
                    hidden_size=md.hidden_size,
                    num_heads=md.num_attention_heads,
                    head_size=md.head_size,
                ),
                post_attn_norm=EngineRMSNorm(
                    context=self._context,
                    hidden_size=md.hidden_size,
                ),
                mlp=EngineMLP(
                    context=self._context,
                    hidden_size=md.hidden_size,
                    intermediate_size=md.intermediate_size,
                    gated=True,  # LLaMA/Qwen style
                    activation="silu",
                ),
            )
            self._layers.append(layer_ops)

        # Final norm
        self._final_norm = EngineRMSNorm(
            context=self._context,
            hidden_size=md.hidden_size,
        )

        # LM head
        self._lm_head = EngineLMHead(
            context=self._context,
            hidden_size=md.hidden_size,
            vocab_size=md.vocab_size,
        )

        # Elementwise ops
        self._elementwise = EngineElementwiseOps(self._context)

    def _bind_weights(self) -> None:
        """Bind loaded weights to operations."""
        w = self._weights

        # Embedding
        self._embedding.set_weights(w.embedding)

        # Layers
        for layer_idx, layer_ops in enumerate(self._layers):
            lw = w.layers[layer_idx]

            layer_ops.input_norm.set_weights(lw.input_layernorm)
            layer_ops.qkv_proj.set_weights(
                q_weight=lw.q_proj,
                k_weight=lw.k_proj,
                v_weight=lw.v_proj,
            )
            layer_ops.o_proj.set_weights(lw.o_proj)
            layer_ops.post_attn_norm.set_weights(lw.post_attention_layernorm)
            layer_ops.mlp.set_weights(
                gate_proj=lw.gate_proj,
                up_proj=lw.up_proj,
                down_proj=lw.down_proj,
            )

        # Final norm
        self._final_norm.set_weights(w.final_norm)

        # LM head
        self._lm_head.set_weights(w.lm_head)

    def _allocate_scratch(self) -> None:
        """Pre-allocate scratch buffers for intermediate results.

        These are fixed-size buffers that get reused across steps.
        Per-step dynamic allocations happen via EngineStepContext.
        """
        # Max tokens we might process in a single step
        max_tokens = 4096  # TODO: Get from config

        md = self.model_desc
        element_size = 2  # float16

        # Hidden states buffer (double-buffered for residual)
        hidden_size_bytes = max_tokens * md.hidden_size * element_size
        self._hidden_buffer_a = self._context.device.newBufferWithLength_options_(
            hidden_size_bytes, MTLResourceStorageModeShared
        )
        self._hidden_buffer_b = self._context.device.newBufferWithLength_options_(
            hidden_size_bytes, MTLResourceStorageModeShared
        )

        # Logits buffer
        logits_size_bytes = max_tokens * md.vocab_size * element_size
        self._logits_buffer = self._context.device.newBufferWithLength_options_(
            logits_size_bytes, MTLResourceStorageModeShared
        )

        logger.debug(
            f"Allocated scratch buffers: hidden={2 * hidden_size_bytes / 1024 / 1024:.1f}MB, "
            f"logits={logits_size_bytes / 1024 / 1024:.1f}MB"
        )

    def execute_step(
        self,
        step_desc: StepDescriptor,
        inputs: EngineInputs,
    ) -> EngineOutputs:
        """Execute a single forward pass.

        This is the main entry point for inference. It:
        1. Creates a step context with command buffer
        2. Encodes embedding lookup
        3. Encodes all transformer layers
        4. Encodes LM head projection
        5. Submits and waits
        6. Returns logits on CPU

        Args:
            step_desc: Step descriptor with metadata
            inputs: Input tensors (all on CPU)

        Returns:
            EngineOutputs with logits on CPU
        """
        self._step_counter += 1
        num_tokens = step_desc.num_scheduled_tokens

        if num_tokens == 0:
            # Empty step - return empty logits
            return EngineOutputs(
                logits=torch.empty(0, self.model_desc.vocab_size, dtype=torch.float16)
            )

        # Copy inputs to GPU buffers
        token_ids_buffer = self._copy_to_buffer(inputs.token_ids, "token_ids")
        positions_buffer = self._copy_to_buffer(inputs.positions, "positions")
        block_table_buffer = self._copy_to_buffer(inputs.block_table, "block_table")
        slot_mapping_buffer = self._copy_to_buffer(inputs.slot_mapping, "slot_mapping")
        seq_lens_buffer = self._copy_to_buffer(inputs.seq_lens, "seq_lens")

        # Create step context
        with EngineStepContext(
            engine_context=self._context,
            step_id=self._step_counter,
            step_kind=step_desc.step_kind,
            num_tokens=num_tokens,
            num_seqs=step_desc.num_seqs_active,
        ) as step_ctx:
            # === ENCODE PHASE ===

            # Embedding lookup
            self._embedding.encode(
                step_ctx=step_ctx,
                token_ids=token_ids_buffer,
                output_buffer=self._hidden_buffer_a,
                num_tokens=num_tokens,
            )

            # Track which buffer has current hidden states
            current_hidden = self._hidden_buffer_a
            other_hidden = self._hidden_buffer_b

            # Memory barrier after embedding
            step_ctx.memory_barrier()

            # Process each transformer layer
            for layer_idx, layer_ops in enumerate(self._layers):
                # Process transformer layer
                current_hidden, other_hidden = self._encode_transformer_layer(
                    step_ctx=step_ctx,
                    layer_idx=layer_idx,
                    layer_ops=layer_ops,
                    hidden_states=current_hidden,
                    residual_buffer=other_hidden,
                    positions_buffer=positions_buffer,
                    block_table_buffer=block_table_buffer,
                    slot_mapping_buffer=slot_mapping_buffer,
                    seq_lens_buffer=seq_lens_buffer,
                    num_tokens=num_tokens,
                    num_seqs=step_desc.num_seqs_active,
                    step_desc=step_desc,
                )

            # Final norm
            self._final_norm.encode(
                step_ctx=step_ctx,
                input_buffer=current_hidden,
                output_buffer=other_hidden,
                num_tokens=num_tokens,
            )
            current_hidden, other_hidden = other_hidden, current_hidden

            step_ctx.memory_barrier()

            # LM head
            self._lm_head.encode(
                step_ctx=step_ctx,
                hidden_states=current_hidden,
                output_buffer=self._logits_buffer,
                num_tokens=num_tokens,
            )

            # === SUBMIT AND WAIT ===
            step_ctx.end_encoding()
            step_ctx.submit()
            step_ctx.wait_until_completed()

            # === READBACK PHASE ===
            logits = self._readback_logits(num_tokens)
            step_ctx.readback_complete()

        return EngineOutputs(logits=logits)

    def _encode_transformer_layer(
        self,
        step_ctx: EngineStepContext,
        layer_idx: int,
        layer_ops: TransformerLayerOps,
        hidden_states: Any,  # MTLBuffer
        residual_buffer: Any,  # MTLBuffer
        positions_buffer: Any,
        block_table_buffer: Any,
        slot_mapping_buffer: Any,
        seq_lens_buffer: Any,
        num_tokens: int,
        num_seqs: int,
        step_desc: StepDescriptor,
    ) -> Tuple[Any, Any]:
        """Encode a single transformer layer.

        Args:
            step_ctx: Step context
            layer_idx: Layer index
            layer_ops: Layer operations
            hidden_states: Input hidden states buffer
            residual_buffer: Buffer for residual connection
            positions_buffer: Position IDs buffer
            block_table_buffer: Block table buffer
            slot_mapping_buffer: Slot mapping buffer
            seq_lens_buffer: Sequence lengths buffer
            num_tokens: Number of tokens
            num_seqs: Number of sequences
            step_desc: Step descriptor

        Returns:
            Tuple of (new_hidden_states, new_residual_buffer)
        """
        md = self.model_desc

        # Allocate scratch buffers
        qkv_size = num_tokens * (md.num_attention_heads + 2 * md.num_kv_heads) * md.head_size * 2
        attn_output_size = num_tokens * md.num_attention_heads * md.head_size * 2

        qkv_buffer = step_ctx.allocate_scratch(qkv_size, f"layer{layer_idx}_qkv")
        attn_output_buffer = step_ctx.allocate_scratch(attn_output_size, f"layer{layer_idx}_attn_out")

        # 1. Input LayerNorm
        # Save hidden states as residual, output to residual_buffer
        self._elementwise.encode_copy(
            step_ctx=step_ctx,
            input_buffer=hidden_states,
            output=residual_buffer,
            num_elements=num_tokens * md.hidden_size,
        )
        layer_ops.input_norm.encode(
            step_ctx=step_ctx,
            input_buffer=hidden_states,
            output_buffer=hidden_states,  # In-place after copy
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 2. QKV Projection
        layer_ops.qkv_proj.encode(
            step_ctx=step_ctx,
            hidden_states=hidden_states,
            qkv_output=qkv_buffer,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 3. RoPE on Q and K
        q_size = num_tokens * md.num_attention_heads * md.head_size
        k_size = num_tokens * md.num_kv_heads * md.head_size

        # Create views into QKV buffer
        q_buffer = qkv_buffer  # Q is at start
        k_offset = q_size * 2  # After Q (in bytes, float16)
        # V is at k_offset + k_size * 2

        layer_ops.rope.encode(
            step_ctx=step_ctx,
            query=qkv_buffer,
            key=qkv_buffer,
            key_offset=k_offset,
            positions=positions_buffer,
            num_tokens=num_tokens,
            num_heads=md.num_attention_heads,
            num_kv_heads=md.num_kv_heads,
        )

        step_ctx.memory_barrier()

        # 4. Write K, V to cache
        layer_ops.kv_write.encode(
            step_ctx=step_ctx,
            k_buffer=qkv_buffer,
            v_buffer=qkv_buffer,
            k_offset=k_offset,
            v_offset=k_offset + k_size * 2,
            slot_mapping=slot_mapping_buffer,
            kv_cache=self._kv_cache.get_layer_cache(layer_idx),
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 5. Attention
        layer_ops.attention.encode(
            step_ctx=step_ctx,
            query=qkv_buffer,
            kv_cache=self._kv_cache.get_layer_cache(layer_idx),
            block_table=block_table_buffer,
            seq_lens=seq_lens_buffer,
            output=attn_output_buffer,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
            is_prefill=step_desc.is_prefill,
        )

        step_ctx.memory_barrier()

        # 6. O Projection + residual add
        layer_ops.o_proj.encode(
            step_ctx=step_ctx,
            attn_output=attn_output_buffer,
            output=hidden_states,
            num_tokens=num_tokens,
        )

        # Add residual
        self._elementwise.encode_residual_add(
            step_ctx=step_ctx,
            x=hidden_states,
            residual=residual_buffer,
            output=hidden_states,
            num_elements=num_tokens * md.hidden_size,
        )

        step_ctx.memory_barrier()

        # 7. Post-attention LayerNorm
        # Save for MLP residual
        self._elementwise.encode_copy(
            step_ctx=step_ctx,
            input_buffer=hidden_states,
            output=residual_buffer,
            num_elements=num_tokens * md.hidden_size,
        )
        layer_ops.post_attn_norm.encode(
            step_ctx=step_ctx,
            input_buffer=hidden_states,
            output_buffer=hidden_states,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 8. MLP
        # Allocate MLP scratch
        mlp_output_buffer = step_ctx.allocate_scratch(
            num_tokens * md.hidden_size * 2,
            f"layer{layer_idx}_mlp_out"
        )
        layer_ops.mlp.encode(
            step_ctx=step_ctx,
            hidden_states=hidden_states,
            output_buffer=mlp_output_buffer,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 9. Final residual add
        self._elementwise.encode_residual_add(
            step_ctx=step_ctx,
            x=mlp_output_buffer,
            residual=residual_buffer,
            output=hidden_states,
            num_elements=num_tokens * md.hidden_size,
        )

        step_ctx.memory_barrier()

        return hidden_states, residual_buffer

    def _copy_to_buffer(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> Any:
        """Copy CPU tensor to MTLBuffer.

        Args:
            tensor: CPU tensor
            name: Buffer name for debugging

        Returns:
            MTLBuffer containing tensor data
        """
        # Ensure contiguous
        tensor = tensor.contiguous()

        # Allocate buffer
        size_bytes = tensor.numel() * tensor.element_size()
        buffer = self._context.device.newBufferWithLength_options_(
            size_bytes, MTLResourceStorageModeShared
        )

        if buffer is None:
            raise RuntimeError(f"Failed to allocate buffer for {name}")

        # Copy data
        np_array = tensor.numpy()
        ctypes.memmove(
            buffer.contents(),
            np_array.ctypes.data,
            size_bytes,
        )

        return buffer

    def _readback_logits(self, num_tokens: int) -> torch.Tensor:
        """Read logits from GPU buffer back to CPU.

        Args:
            num_tokens: Number of tokens

        Returns:
            Logits tensor on CPU [num_tokens, vocab_size]
        """
        # Calculate size
        vocab_size = self.model_desc.vocab_size
        size_bytes = num_tokens * vocab_size * 2  # float16

        # Read from buffer
        np_array = np.zeros((num_tokens, vocab_size), dtype=np.float16)
        ctypes.memmove(
            np_array.ctypes.data,
            self._logits_buffer.contents(),
            size_bytes,
        )

        # Convert to torch
        logits = torch.from_numpy(np_array)
        return logits

    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        return {
            "step_count": self._step_counter,
            "num_layers": self.model_desc.num_layers,
            "hidden_size": self.model_desc.hidden_size,
            "vocab_size": self.model_desc.vocab_size,
        }
