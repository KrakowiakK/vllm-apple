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
                    num_heads=md.num_attention_heads,
                    num_kv_heads=md.num_kv_heads,
                    rotary_dim=None,  # Default to head_size
                    max_position=md.max_position_embeddings,
                    base=md.rope_theta,
                ),
                attention=PagedAttentionOp(
                    context=self._context,
                    num_kv_heads=md.num_kv_heads,
                    num_query_heads=md.num_attention_heads,
                    head_size=md.head_size,
                    block_size=self._kv_cache.block_size,
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

            # Handle both fused QKV and separate Q/K/V weights
            # Models may use either pattern depending on architecture/source
            if lw.qkv_proj is not None:
                # Fused QKV weights (single matrix)
                layer_ops.qkv_proj.set_weights(qkv_weight=lw.qkv_proj)
            elif lw.q_proj is not None and lw.k_proj is not None and lw.v_proj is not None:
                # Separate Q, K, V weights
                layer_ops.qkv_proj.set_weights(
                    q_weight=lw.q_proj,
                    k_weight=lw.k_proj,
                    v_weight=lw.v_proj,
                )
            else:
                # Neither found - this is a weight loading failure
                raise RuntimeError(
                    f"Layer {layer_idx}: No QKV weights found. "
                    f"Expected either qkv_proj (fused) or q_proj/k_proj/v_proj (separate). "
                    f"Got: qkv_proj={lw.qkv_proj is not None}, "
                    f"q_proj={lw.q_proj is not None}, k_proj={lw.k_proj is not None}, v_proj={lw.v_proj is not None}"
                )

            layer_ops.o_proj.set_weights(lw.o_proj)
            layer_ops.post_attn_norm.set_weights(lw.post_attention_layernorm)

            # Handle both fused gate-up and separate gate/up weights
            if lw.gate_up_proj is not None:
                layer_ops.mlp.set_weights(gate_up_proj=lw.gate_up_proj, down_proj=lw.down_proj)
            elif lw.gate_proj is not None and lw.up_proj is not None:
                layer_ops.mlp.set_weights(
                    gate_proj=lw.gate_proj,
                    up_proj=lw.up_proj,
                    down_proj=lw.down_proj,
                )
            else:
                raise RuntimeError(
                    f"Layer {layer_idx}: No MLP weights found. "
                    f"Expected either gate_up_proj (fused) or gate_proj/up_proj (separate)."
                )

        # Final norm
        self._final_norm.set_weights(w.final_norm)

        # LM head
        self._lm_head.set_weights(w.lm_head)

    def _allocate_scratch(self, max_tokens: Optional[int] = None) -> None:
        """Pre-allocate scratch buffers for intermediate results.

        These are fixed-size buffers that get reused across steps.
        Per-step dynamic allocations happen via EngineStepContext.

        Args:
            max_tokens: Maximum tokens per step. If None, uses model_desc.max_position_embeddings
                        capped at 4096. For decode-only engine, batch_size is typically the limit.

        The allocated size determines the hard limit for tokens per step.
        execute_step() will validate num_tokens <= _max_scratch_tokens.
        """
        from .config import get_engine_config

        # Determine max tokens for scratch buffers
        if max_tokens is None:
            # Use config max_batch_size (for decode, 1 token per seq = batch_size tokens)
            # Fall back to model's max position, capped at a reasonable default
            config = get_engine_config()
            max_tokens = min(
                config.max_batch_size,  # Engine config limit
                self.model_desc.max_position_embeddings,  # Model limit
                4096,  # Hard cap to prevent huge allocations
            )

        # Store for bounds checking in execute_step
        self._max_scratch_tokens = max_tokens

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

        logger.info(
            f"Allocated scratch buffers for max_tokens={max_tokens}: "
            f"hidden={2 * hidden_size_bytes / 1024 / 1024:.1f}MB, "
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

        Supports both:
        - Decode steps (typically 1 token per sequence) using fused KV-write+attention
        - Prefill/mixed steps (num_tokens may exceed num_seqs) using token-parallel attention

        Args:
            step_desc: Step descriptor with metadata
            inputs: Input tensors (all on CPU)

        Returns:
            EngineOutputs with logits on CPU

        """
        self._step_counter += 1
        num_tokens = step_desc.num_scheduled_tokens
        num_seqs = step_desc.num_seqs_active

        # Sanity check for decode: should have exactly 1 token per sequence
        if step_desc.is_decode and num_tokens != num_seqs and num_tokens > 0:
            logger.warning(
                f"Decode step has num_tokens={num_tokens} != num_seqs={num_seqs}. "
                f"This may indicate mixed prefill/decode which is not fully supported."
            )

        # Bounds check: prevent buffer overflow in scratch buffers
        if num_tokens > self._max_scratch_tokens:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds scratch buffer capacity "
                f"({self._max_scratch_tokens}). This would cause buffer overflow. "
                f"Either reduce batch size or increase max_batch_size in engine config."
            )

        if num_tokens == 0:
            # Empty step - return empty logits
            return EngineOutputs(
                logits=torch.empty(0, self.model_desc.vocab_size, dtype=torch.float16)
            )

        # Copy inputs to GPU buffers
        # Ensure proper dtypes: positions, slot_mapping, seq_lens must be int32 for Metal kernels
        # token_ids must be int32 for embedding kernel (device const int*)
        token_ids_int32 = inputs.token_ids.to(torch.int32) if inputs.token_ids.dtype != torch.int32 else inputs.token_ids
        token_ids_buffer = self._copy_to_buffer(token_ids_int32, "token_ids")
        positions_int32 = inputs.positions.to(torch.int32) if inputs.positions.dtype != torch.int32 else inputs.positions
        positions_buffer = self._copy_to_buffer(positions_int32, "positions")
        block_table_int32 = inputs.block_table.to(torch.int32) if inputs.block_table.dtype != torch.int32 else inputs.block_table
        block_table_buffer = self._copy_to_buffer(block_table_int32, "block_table")
        slot_mapping_int32 = inputs.slot_mapping.to(torch.int32) if inputs.slot_mapping.dtype != torch.int32 else inputs.slot_mapping
        slot_mapping_buffer = self._copy_to_buffer(slot_mapping_int32, "slot_mapping")
        seq_lens_int32 = inputs.seq_lens.to(torch.int32) if inputs.seq_lens.dtype != torch.int32 else inputs.seq_lens
        seq_lens_buffer = self._copy_to_buffer(seq_lens_int32, "seq_lens")

        # Compute max position for RoPE bounds checking
        max_position_in_batch = int(inputs.positions.max().item()) if num_tokens > 0 else 0

        # Compute max_seq_len for attention (on CPU before entering encode phase)
        # This is more reliable than step_desc.max_num_blocks_per_seq which may be 0
        max_seq_len = int(inputs.seq_lens.max().item()) if inputs.seq_lens.numel() > 0 else 0

        # Compute max_blocks_per_seq from actual block_table shape
        # block_table is [num_seqs, max_blocks_per_seq] - use actual shape to avoid OOB
        if inputs.block_table.ndim == 2 and inputs.block_table.shape[1] > 0:
            max_blocks_per_seq = inputs.block_table.shape[1]
        else:
            # Fallback: compute from max_seq_len and block_size
            max_blocks_per_seq = (max_seq_len + self._kv_cache.block_size - 1) // self._kv_cache.block_size if max_seq_len > 0 else 1

        if (
            inputs.block_table.ndim == 2
            and inputs.block_table.shape[1] > 0
            and max_seq_len > max_blocks_per_seq * self._kv_cache.block_size
        ):
            raise ValueError(
                f"seq_lens max ({max_seq_len}) exceeds block_table capacity "
                f"({max_blocks_per_seq * self._kv_cache.block_size}). "
                f"max_blocks_per_seq={max_blocks_per_seq}, block_size={self._kv_cache.block_size} "
                f"(step {self._step_counter})."
            )

        # Validate inputs before encoding to prevent OOB writes
        # Always validate block_table to catch seq_lens/block_table inconsistencies
        if inputs.block_table.numel() > 0:
            self._kv_cache.validate_block_table(
                inputs.block_table,
                context=f"step {self._step_counter}"
            )

        # For prefill, validate slot_mapping is within KV cache capacity
        if step_desc.is_prefill and inputs.slot_mapping.numel() > 0:
            self._kv_cache.validate_slot_mapping(
                inputs.slot_mapping,
                context=f"step {self._step_counter} prefill"
            )

        # Prefill/mixed steps need a token→sequence mapping to interpret the flattened
        # query batch. Build it once on CPU and upload as an int32 MTLBuffer.
        token_to_seq_buffer = None
        if step_desc.is_prefill:
            if inputs.query_start_locs is None:
                raise ValueError(
                    "Prefill step requires EngineInputs.query_start_locs "
                    "(cumulative start offsets, length num_seqs_active + 1)."
                )

            qsl = inputs.query_start_locs.to(torch.int32)
            if qsl.numel() != num_seqs + 1:
                raise ValueError(
                    f"query_start_locs has {qsl.numel()} elements, expected {num_seqs + 1} "
                    f"(num_seqs_active={num_seqs})."
                )
            if int(qsl[0].item()) != 0:
                raise ValueError(f"query_start_locs[0] must be 0, got {int(qsl[0].item())}")
            if int(qsl[-1].item()) != num_tokens:
                raise ValueError(
                    f"query_start_locs[-1] must equal num_tokens ({num_tokens}), "
                    f"got {int(qsl[-1].item())}."
                )

            lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
            if (lens < 0).any():
                raise ValueError("query_start_locs must be non-decreasing (negative query length found).")

            token_to_seq = torch.repeat_interleave(
                torch.arange(num_seqs, dtype=torch.int32),
                lens,
            )
            if token_to_seq.numel() != num_tokens:
                raise ValueError(
                    f"token_to_seq has {token_to_seq.numel()} elements, expected num_tokens={num_tokens}. "
                    f"This indicates inconsistent query_start_locs."
                )
            token_to_seq_buffer = self._copy_to_buffer(token_to_seq, "token_to_seq")

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
                    token_to_seq_buffer=token_to_seq_buffer,
                    num_tokens=num_tokens,
                    num_seqs=step_desc.num_seqs_active,
                    step_desc=step_desc,
                    max_position_in_batch=max_position_in_batch,
                    max_seq_len=max_seq_len,
                    max_blocks_per_seq=max_blocks_per_seq,
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
        token_to_seq_buffer: Any,
        num_tokens: int,
        num_seqs: int,
        step_desc: StepDescriptor,
        max_position_in_batch: int,
        max_seq_len: int,
        max_blocks_per_seq: int,
    ) -> Tuple[Any, Any]:
        """Encode a single transformer layer.

        Args:
            step_ctx: Step context
            layer_idx: Layer index
            layer_ops: Layer operations
            hidden_states: Input hidden states buffer
            residual_buffer: Buffer for residual connection
            positions_buffer: Position IDs buffer (int32)
            block_table_buffer: Block table buffer (int32)
            slot_mapping_buffer: Slot mapping buffer (int32)
            seq_lens_buffer: Sequence lengths buffer (int32)
            num_tokens: Number of tokens
            num_seqs: Number of sequences
            step_desc: Step descriptor
            max_position_in_batch: Maximum position ID for RoPE bounds check
            max_seq_len: Maximum sequence length for attention
            max_blocks_per_seq: Maximum blocks per sequence (block_table second dim)

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

        # Create views into QKV buffer using EngineTensor for K
        q_buffer = qkv_buffer  # Q is at start
        k_offset_bytes = q_size * 2  # After Q (in bytes, float16)
        v_offset_bytes = k_offset_bytes + k_size * 2

        # Create EngineTensor view for K at offset
        k_tensor = EngineTensor(
            buffer=qkv_buffer,
            shape=(num_tokens, md.num_kv_heads, md.head_size),
            dtype=EngineDType.FLOAT16,
            offset=k_offset_bytes,
        )

        layer_ops.rope.encode(
            step_ctx=step_ctx,
            query=qkv_buffer,
            key=k_tensor,
            positions=positions_buffer,
            num_tokens=num_tokens,
            max_position_in_batch=max_position_in_batch,
        )

        step_ctx.memory_barrier()

        # 4. Attention (decode-only): fused KV-write + attention
        # Get KV cache buffers for this layer
        k_cache, v_cache = self._kv_cache.get_buffers(layer_idx)

        if step_desc.is_prefill:
            if token_to_seq_buffer is None:
                raise RuntimeError("token_to_seq_buffer is required for prefill attention")

            # Prefill/mixed: write K/V for every token, then run token-parallel attention.
            layer_ops.kv_write.encode_prefill(
                step_ctx=step_ctx,
                new_keys_buffer=qkv_buffer,
                new_values_buffer=qkv_buffer,
                key_buffer=k_cache,
                value_buffer=v_cache,
                slot_mapping_buffer=slot_mapping_buffer,
                num_tokens=num_tokens,
                new_keys_offset=k_offset_bytes,
                new_values_offset=v_offset_bytes,
            )

            step_ctx.memory_barrier()

            layer_ops.attention.encode_prefill(
                step_ctx=step_ctx,
                query_buffer=qkv_buffer,
                key_buffer=k_cache,
                value_buffer=v_cache,
                block_table_buffer=block_table_buffer,
                token_to_seq_buffer=token_to_seq_buffer,
                positions_buffer=positions_buffer,
                output_buffer=attn_output_buffer,
                num_tokens=num_tokens,
                num_seqs=num_seqs,
                max_seq_len=max_seq_len,
                max_blocks_per_seq=max_blocks_per_seq,
                query_offset=0,
                output_offset=0,
            )
        else:
            # Decode: fused KV-write + attention.
            layer_ops.attention.encode_decode_fused(
                step_ctx=step_ctx,
                query_buffer=qkv_buffer,  # Q is at start
                new_keys_buffer=qkv_buffer,
                new_values_buffer=qkv_buffer,
                key_buffer=k_cache,
                value_buffer=v_cache,
                block_table_buffer=block_table_buffer,
                seq_lens_buffer=seq_lens_buffer,
                output_buffer=attn_output_buffer,
                num_seqs=num_seqs,
                max_seq_len=max_seq_len,
                max_blocks_per_seq=max_blocks_per_seq,
                query_offset=0,
                new_keys_offset=k_offset_bytes,
                new_values_offset=v_offset_bytes,
                output_offset=0,
            )

        step_ctx.memory_barrier()

        # 5. O Projection + residual add
        layer_ops.o_proj.encode(
            step_ctx=step_ctx,
            attn_output=attn_output_buffer,
            output_buffer=hidden_states,
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

        # Copy data using PyObjC's as_buffer() (safe for PyObjC 12+)
        np_array = tensor.numpy()
        # Get memoryview of MTLBuffer for safe copying
        buffer_view = buffer.contents().as_buffer(size_bytes)
        buffer_np = np.frombuffer(buffer_view, dtype=np_array.dtype)
        np.copyto(buffer_np, np_array.ravel())

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

        # Read from buffer using PyObjC's as_buffer() (safe for PyObjC 12+)
        buffer_view = self._logits_buffer.contents().as_buffer(size_bytes)
        np_array = np.frombuffer(buffer_view, dtype=np.float16).reshape(num_tokens, vocab_size).copy()

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
