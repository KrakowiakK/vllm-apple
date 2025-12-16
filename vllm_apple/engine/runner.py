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

import os
import torch
import numpy as np

from vllm.logger import init_logger

# Debug checkpoint capture (gated by VLLM_PREFILL_EQ_DEBUG=1)
CHECKPOINT_DEBUG_ENABLED = os.environ.get("VLLM_PREFILL_EQ_DEBUG", "0") == "1"
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
from .ops.topk import EngineTopK

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
        
        # Initialize debug checkpoint queue
        self._pending_checkpoints = []


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
        
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("DEBUG: EngineRunner __init__ called")
        self.logger.info(f"DEBUG: Initializing EngineRoPE with theta={md.rope_theta}")
        for layer_idx in range(md.num_layers):
            layer_ops = TransformerLayerOps(
                input_norm=EngineRMSNorm(
                    context=self._context,
                    hidden_size=md.hidden_size,
                    eps=md.rms_norm_eps,
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
                    eps=md.rms_norm_eps,
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
            eps=md.rms_norm_eps,
        )

        # LM head
        self._lm_head = EngineLMHead(
            context=self._context,
            hidden_size=md.hidden_size,
            vocab_size=md.vocab_size,
        )

        # Top-K logits selection (optional, enabled via VLLM_METAL_TOPK_LOGITS)
        from .config import get_topk_logits
        self._topk_k = get_topk_logits()
        if self._topk_k is not None:
            self._topk_op = EngineTopK(self._context, k=self._topk_k)
            logger.info(f"Top-K logits enabled with k={self._topk_k}")
        else:
            self._topk_op = None

        # Elementwise ops
        self._elementwise = EngineElementwiseOps(self._context)

    def _bind_weights(self) -> None:
        """Bind loaded weights to operations."""
        w = self._weights

        # Embedding
        self._embedding.set_weights(w.embedding)
        
        # DEBUG: Verify first few weights
        if w.embedding is not None:
            import numpy as np
            size = 20 * 2 # 20 elements * 2 bytes
            view = w.embedding.contents().as_buffer(size)
            chk_np = np.frombuffer(view, dtype=np.float16)
            logger.info(f"DEBUG: GPU embedding weights[:20]: {chk_np.tolist()}")

        # Layers
        for layer_idx, layer_ops in enumerate(self._layers):
            lw = w.layers[layer_idx]

            layer_ops.input_norm.set_weights(lw.input_layernorm)

            # Handle both fused QKV and separate Q/K/V weights
            # Models may use either pattern depending on architecture/source
            if lw.qkv_proj is not None:
                # Fused QKV weights (single matrix)
                layer_ops.qkv_proj.set_weights(
                    qkv_weight=lw.qkv_proj,
                    bias_buffer=lw.qkv_bias,
                )
            elif lw.q_proj is not None and lw.k_proj is not None and lw.v_proj is not None:
                # Separate Q, K, V weights
                logger.info(f"DEBUG: Setting separate weights for layer {layer_idx}: q={lw.q_proj}, k={lw.k_proj}, v={lw.v_proj}")
                layer_ops.qkv_proj.set_weights(
                    q_weight=lw.q_proj,
                    k_weight=lw.k_proj,
                    v_weight=lw.v_proj,
                    bias_buffer=lw.qkv_bias,
                )
            else:
                # Neither found - this is a weight loading failure
                raise RuntimeError(
                    f"Layer {layer_idx}: No QKV weights found. "
                    f"Expected either qkv_proj (fused) or q_proj/k_proj/v_proj (separate). "
                    f"Got: qkv_proj={lw.qkv_proj is not None}, "
                    f"q_proj={lw.q_proj is not None}, k_proj={lw.k_proj is not None}, v_proj={lw.v_proj is not None}"
                )

            layer_ops.o_proj.set_weights(lw.o_proj, bias_buffer=lw.o_proj_bias)
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
                16384,  # Hard cap - allows batch 16 x 1024 tokens
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

        # Top-K buffers (only if Top-K is enabled)
        if self._topk_op is not None:
            k = self._topk_k
            # Indices buffer: [max_tokens, k] int32
            topk_indices_size = max_tokens * k * 4
            self._topk_indices_buffer = self._context.device.newBufferWithLength_options_(
                topk_indices_size, MTLResourceStorageModeShared
            )
            # Values buffer: [max_tokens, k] float16
            topk_values_size = max_tokens * k * element_size
            self._topk_values_buffer = self._context.device.newBufferWithLength_options_(
                topk_values_size, MTLResourceStorageModeShared
            )
            topk_total = (topk_indices_size + topk_values_size) / 1024
            logger.info(f"Allocated Top-K buffers: {topk_total:.1f}KB (k={k})")
        else:
            self._topk_indices_buffer = None
            self._topk_values_buffer = None

        logger.info(
            f"Allocated scratch buffers for max_tokens={max_tokens}: "
            f"hidden={2 * hidden_size_bytes / 1024 / 1024:.1f}MB, "
            f"logits={logits_size_bytes / 1024 / 1024:.1f}MB"
        )

    def execute_step(
        self,
        step_desc: StepDescriptor,
        inputs: EngineInputs,
        return_step_ctx: bool = False,
    ):
        """Execute a single forward pass (with auto-chunking)."""
        self.logger.info(f"DEBUG: execute_step called for step {step_desc.step_id}")
        # Increment step counter once per logical step
        self._step_counter += 1
        num_tokens = step_desc.num_scheduled_tokens

        # Identify indices of the last token of each sequence
        # These are the ONLY logits we need to return.
        if inputs.query_start_locs is None:
             raise ValueError("inputs.query_start_locs is required")
        
        # Debug Tokens
        logger.info(f"Step inputs: num_tokens={num_tokens}. IDs[:10]={inputs.token_ids[:10]}. IDs[-10:]={inputs.token_ids[-10:]}")
        logger.info(f"Query Start Locs: {inputs.query_start_locs}")
        
        # Ensure CPU / Long for indexing
        qsl = inputs.query_start_locs.to(device="cpu", dtype=torch.long)
        # qsl is [0, len1, len1+len2, ...]. 
        # Last tokens are at qsl[1:] - 1
        last_token_indices = qsl[1:] - 1
        
        # Check capacity
        if num_tokens <= self._max_scratch_tokens:
            out = self._execute_slice(step_desc, inputs, return_step_ctx=return_step_ctx)
            
            # Post-process: Filter logits to only last tokens
            # Handle tuple return (outputs, profile_data)
            actual_out = out[0] if isinstance(out, tuple) else out
            
            if actual_out.logits is not None:
                # logits shape: [num_tokens, vocab]
                # filter shape: [batch_size, vocab]
                actual_out.logits = actual_out.logits[last_token_indices]
                
                # Debug Dump for Verification
                import os
                if os.environ.get("VLLM_DUMP_LOGITS"):
                    try:
                        dump_path = f"/tmp/vllm_logits_{self._step_counter}_{os.getpid()}.pt"
                        torch.save(actual_out.logits, dump_path)
                        logger.info(f"Dumped logits to {dump_path}")
                    except Exception as e:
                        logger.error(f"Failed to dump logits: {e}")

            return out

        # === CHUNKED EXECUTION ===
        logger.info(
            f"Step {self._step_counter}: splitting {num_tokens} tokens into chunks "
            f"(limit {self._max_scratch_tokens})."
        )
        
        # 1. Compute global token_to_seq (needed for prefill slices)
        global_token_to_seq = None
        if step_desc.is_prefill:
            global_token_to_seq = self._derive_token_to_seq(inputs, step_desc.num_seqs_active, num_tokens)

        # 2. Loop chunks
        chunk_size = self._max_scratch_tokens
        collected_logits = []
        import copy
        
        for start_idx in range(0, num_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, num_tokens)
            current_chunk_len = end_idx - start_idx
            
            # Slice inputs
            chunk_inputs = copy.copy(inputs)
            chunk_inputs.token_ids = inputs.token_ids[start_idx:end_idx]
            chunk_inputs.positions = inputs.positions[start_idx:end_idx]
            chunk_inputs.slot_mapping = inputs.slot_mapping[start_idx:end_idx]
            
            # Slice token_to_seq if present
            chunk_token_to_seq = None
            if global_token_to_seq is not None:
                chunk_token_to_seq = global_token_to_seq[start_idx:end_idx]
                
            # Create chunk step descriptor
            chunk_step_desc = copy.copy(step_desc)
            chunk_step_desc.num_scheduled_tokens = current_chunk_len
            
            # Execute slice
            chunk_out = self._execute_slice(
                chunk_step_desc, 
                chunk_inputs, 
                return_step_ctx=False,
                token_to_seq_override=chunk_token_to_seq
            )
            
            # Extract relevant logits
            # Find which 'last tokens' fall in this chunk
            mask = (last_token_indices >= start_idx) & (last_token_indices < end_idx)
            if mask.any():
                # Global indices of last tokens in this chunk
                relevant_globals = last_token_indices[mask]
                # Map to local chunk indices
                relevant_locals = relevant_globals - start_idx
                # Extract rows
                rows = chunk_out.logits[relevant_locals]
                collected_logits.append(rows)
            
            # Discard rest of logits to save memory
            chunk_out.logits = None # hinting GC
            
        # 3. Aggregate
        if not collected_logits:
             # Should not happen unless batch size 0?
             full_logits = torch.empty((0, self._vocab_size), dtype=torch.float16)
        else:
             full_logits = torch.cat(collected_logits, dim=0)

        # Debug Dump for Verification
        import os
        if os.environ.get("VLLM_DUMP_LOGITS"):
            try:
                # full_logits is now already filtered to [batch_size, vocab]
                dump_path = f"/tmp/vllm_logits_{self._step_counter}_{os.getpid()}.pt"
                torch.save(full_logits, dump_path)
                logger.info(f"Dumped logits to {dump_path}")
            except Exception as e:
                logger.error(f"Failed to dump logits: {e}")
        
        return EngineOutputs(
            logits=full_logits,
            topk_indices=None, 
            topk_values=None
        )

    def _derive_token_to_seq(self, inputs: EngineInputs, num_seqs: int, num_tokens: int) -> torch.Tensor:
        """Helper to derive token_to_seq map for prefill."""
        if inputs.query_start_locs is None:
             raise ValueError("Prefill step requires query_start_locs")
             
        qsl = inputs.query_start_locs.to(torch.int32)
        if qsl.numel() != num_seqs + 1:
            raise ValueError(f"query_start_locs has {qsl.numel()}, expected {num_seqs + 1}")

        lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
        if (lens < 0).any():
             raise ValueError("Negative query length found")
             
        token_to_seq = torch.repeat_interleave(
             torch.arange(num_seqs, dtype=torch.int32),
             lens,
        )
        if token_to_seq.numel() != num_tokens:
             raise ValueError(f"token_to_seq size {token_to_seq.numel()} mismatch num_tokens {num_tokens}")
             
        return token_to_seq

    def _execute_slice(
        self,
        step_desc: StepDescriptor,
        inputs: EngineInputs,
        return_step_ctx: bool = False,
        token_to_seq_override: Optional[torch.Tensor] = None,
    ):
        """Execute a single forward pass (slice)."""
        import numpy as np
        # Note: step_counter incremented in wrapper
        num_tokens = step_desc.num_scheduled_tokens
        num_seqs = step_desc.num_seqs_active

        # Sanity check for decode: should have exactly 1 token per sequence
        if step_desc.is_decode and num_tokens != num_seqs and num_tokens > 0:
            logger.warning(
                f"Decode step has num_tokens={num_tokens} != num_seqs={num_seqs}. "
                f"This may indicate mixed prefill/decode which is not fully supported."
            )

        if num_tokens == 0:
            return EngineOutputs(
                logits=torch.empty(0, self.model_desc.vocab_size, dtype=torch.float16),
                topk_indices=None,
                topk_values=None,
            )

        # Copy inputs to GPU buffers
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

        # DEBUG: Verify token_ids buffer content
        if self._step_counter <= 2:
            import numpy as np
            size = token_ids_int32.numel() * 4
            view = token_ids_buffer.contents().as_buffer(size)
            chk_np = np.frombuffer(view, dtype=np.int32)
            logger.info(f"DEBUG: GPU token_ids buffer content: {chk_np.tolist()}")


        # Debug: print inputs
        import os
        if self._step_counter <= 2 and os.environ.get('VLLM_ENGINE_DEBUG'):
            logger.info(
                f"[DEBUG] Step {self._step_counter} inputs: "
                f"kind={step_desc.step_kind}, num_tokens={num_tokens}, num_seqs={num_seqs}, "
                f"token_ids={inputs.token_ids.tolist()}, positions={inputs.positions.tolist()}, "
                f"seq_lens={inputs.seq_lens.tolist()}"
            )

        # Compute max position for RoPE bounds checking
        max_position_in_batch = int(inputs.positions.max().item()) if num_tokens > 0 else 0

        # Compute max_seq_len for attention (on CPU before entering encode phase)
        max_seq_len = int(inputs.seq_lens.max().item()) if inputs.seq_lens.numel() > 0 else 0

        # Compute max_blocks_per_seq from actual block_table shape
        if inputs.block_table.ndim == 2 and inputs.block_table.shape[1] > 0:
            max_blocks_per_seq = inputs.block_table.shape[1]
        else:
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

        # Validate inputs
        if inputs.block_table.numel() > 0:
            self._kv_cache.validate_block_table(
                inputs.block_table,
                context=f"step {self._step_counter}"
            )
        if step_desc.is_prefill and inputs.slot_mapping.numel() > 0:
            self._kv_cache.validate_slot_mapping(
                inputs.slot_mapping,
                context=f"step {self._step_counter} prefill"
            )

        # Prefill/mixed steps need a token→sequence mapping
        token_to_seq_buffer = None
        if step_desc.is_prefill:
            if token_to_seq_override is not None:
                # Use provided override (for chunked prefill)
                token_to_seq_buffer = self._copy_to_buffer(token_to_seq_override.to(torch.int32), "token_to_seq")
            else:
                # Standard derivation
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
                
                # Check for negative lengths/monotonicity
                lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
                if (lens < 0).any():
                     raise ValueError("query_start_locs must be non-decreasing")
                
                token_to_seq = torch.repeat_interleave(
                    torch.arange(num_seqs, dtype=torch.int32),
                    lens,
                )
                if token_to_seq.numel() != num_tokens:
                    raise ValueError(
                        f"token_to_seq has {token_to_seq.numel()} elements, expected num_tokens={num_tokens}."
                    )
                token_to_seq_buffer = self._copy_to_buffer(token_to_seq, "token_to_seq")

        # Initialize checkpoint context for debugging
        self._init_checkpoint_context(num_tokens)

        # Create step context
        with EngineStepContext(
            engine_context=self._context,
            step_id=self._step_counter,
            step_kind=step_desc.step_kind,
            num_tokens=num_tokens,
            num_seqs=step_desc.num_seqs_active,
        ) as step_ctx:
            # === ENCODE PHASE ===

            if step_desc.step_kind == "decode" and step_desc.num_seqs_active == 1:
                step_ctx.set_decode_single_seq(True)

            self._context.reset_scratch_generation()

            # Embedding lookup
            self._embedding.encode(
                step_ctx=step_ctx,
                token_ids=token_ids_buffer,
                output_buffer=self._hidden_buffer_a,
                num_tokens=num_tokens,
            )

            # DEBUG: Capture embedding output
            if CHECKPOINT_DEBUG_ENABLED:
                self._capture_checkpoint(
                    name="embed_output",
                    buffer=self._hidden_buffer_a,
                    shape=(num_tokens, self.model_desc.hidden_size),
                    dtype=np.float16,
                    last_token_only=False, # Capture all tokens for embedding verification
                    step_ctx=step_ctx,
                )

            current_hidden = self._hidden_buffer_a
            other_hidden = self._hidden_buffer_b

            step_ctx.memory_barrier()

            # Process each transformer layer
            for layer_idx, layer_ops in enumerate(self._layers):
                with step_ctx.layer_scope(layer_idx):
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

            # Top-K selection
            if self._topk_op is not None:
                step_ctx.memory_barrier()
                self._topk_op.encode(
                    step_ctx=step_ctx,
                    logits=self._logits_buffer,
                    indices_out=self._topk_indices_buffer,
                    values_out=self._topk_values_buffer,
                    num_tokens=num_tokens,
                    vocab_size=self.model_desc.vocab_size,
                )

            # === SUBMIT AND WAIT ===
            step_ctx.end_encoding()
            step_ctx.submit()
            step_ctx.wait_until_completed()

            # PROCESS PENDING CHECKPOINTS (Deferred Readback)
            if hasattr(self, "_pending_checkpoints") and self._pending_checkpoints:
                for req in self._pending_checkpoints:
                    self._process_checkpoint_read(
                        req["name"], req["buffer"], req["shape"], 
                        req["dtype"], req["offset"], 
                        req["last_token_only"], req["num_tokens"]
                    )
                self._pending_checkpoints.clear()

            # === CHECKPOINT CAPTURE (DEBUG) ===
            if CHECKPOINT_DEBUG_ENABLED:
                self._capture_checkpoint(
                    name="lm_head_logits",
                    buffer=self._logits_buffer,
                    shape=(num_tokens, self.model_desc.vocab_size),
                    dtype=np.float16,
                    num_tokens=num_tokens,
                )

            # === READBACK PHASE ===
            # For chunked execution (override set), we skip TopK opt to ensure simple logic
            # OR we can keep it if enabled. Logic below handles it.
            if self._topk_op is not None:
                logits, topk_indices, topk_values = self._readback_topk_logits(num_tokens)
            else:
                logits = self._readback_logits(num_tokens)
                topk_indices = None
                topk_values = None
            step_ctx.readback_complete()

            profiling_stats = None
            if return_step_ctx:
                profiling_stats = step_ctx.get_profiling_stats()

            self._context.release_scratch_for_step(step_desc.step_id)

        outputs = EngineOutputs(
            logits=logits,
        )

        if return_step_ctx:
            return outputs, profiling_stats
        return outputs

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
        # === BATCH=1 DECODE: Reduced barriers ===
        # For batch=1 decode, we coalesce barriers to reduce overhead.
        # This is safe because all ops encode to the same command buffer.
        use_reduced_barriers = step_ctx.decode_single_seq

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

        # Barrier after input_norm: Skip for batch=1 decode (coalesce with QKV)

        if not use_reduced_barriers:
            step_ctx.memory_barrier()

        if CHECKPOINT_DEBUG_ENABLED:
            self._capture_checkpoint(
                 name=f"layer{layer_idx}_input_norm",
                 buffer=hidden_states,
                 shape=(num_tokens, md.hidden_size),
                 dtype=np.float16,
                 step_ctx=step_ctx,
            )

        # 2. QKV Projection
        layer_ops.qkv_proj.encode(
            step_ctx=step_ctx,
            hidden_states=hidden_states,
            qkv_output=qkv_buffer,
            num_tokens=num_tokens,
        )

        # Capture moved


        # Barrier after QKV: Skip for batch=1 decode (coalesce with RoPE)
        if not use_reduced_barriers:
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


        # Phase B: Isolation Toggles - Hybrid Execution
        if os.environ.get("VLLM_PREFILL_FORCE_PYTORCH_ROPE") == "1":
            step_ctx.flush_and_sync()
            self._cpu_rope(step_ctx, qkv_buffer, positions_buffer, num_tokens, layer_idx)
        else:
            layer_ops.rope.encode(
                step_ctx=step_ctx,
                query=qkv_buffer,
                key=k_tensor,
                positions=positions_buffer,
                num_tokens=num_tokens,
                max_position_in_batch=max_position_in_batch,
            )

        # BARRIER 1: QKV + RoPE complete (always needed before attention)
        step_ctx.memory_barrier()

        if CHECKPOINT_DEBUG_ENABLED and os.environ.get("VLLM_DUMP_ROPE") == "1":
             logger.info(f"DEBUG: RoPE Capture - layer {layer_idx}, num_tokens={num_tokens}, head_size={md.head_size}")
             # Capture Post-RoPE outputs
             # self._capture_checkpoint(
             #     name=f"layer{layer_idx}_rope_q",
             #     buffer=qkv_buffer,
             #     shape=(num_tokens, md.num_attention_heads, md.head_size),
             #     dtype=np.float16,
             #     offset=0,
             #     last_token_only=True, 
             #     step_ctx=step_ctx,
             # )
             
             # self._capture_checkpoint(
             #     name=f"layer{layer_idx}_rope_k",
             #     buffer=qkv_buffer,
             #     shape=(num_tokens, md.num_kv_heads, md.head_size),
             #     dtype=np.float16,
             #     offset=k_offset_bytes,
             #     last_token_only=True,
             #     step_ctx=step_ctx,
             # )
             # K is harder because it is interleaved or offset? 
             # qkv_buffer is flattened [tokens, heads+2kv, head_size]
             # OR is it separate Q and K buffers if not packed?
             # In `execute_slice`, we passed `qkv_buffer` for Q and `k_tensor` for K?
             # `qkv_buffer` holds ALL (Q, K, V).
             # We should probably capture the WHOLE `qkv_out` again, now that it is rotated.
             # `qkv_buffer` holds ALL (Q, K, V).
             # We should probably capture the WHOLE `qkv_out` again, now that it is rotated.
             # self._capture_checkpoint(step_ctx, layer_idx, "rope_out_fused", qkv_buffer, offset=0)


        # Phase B: Isolation Toggles - Force PyTorch SDPA
        if os.environ.get("VLLM_PREFILL_FORCE_PYTORCH_SDPA") == "1":
            step_ctx.flush_and_sync()
            self._cpu_sdpa(
                step_ctx, qkv_buffer, attn_output_buffer, 
                num_tokens, layer_idx, num_seqs, max_seq_len,
                token_to_seq_buffer
            )

        else:
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

                # Barrier between KV write and prefill attention (always needed)
                step_ctx.memory_barrier()

                if CHECKPOINT_DEBUG_ENABLED:
                     # Capture Setup Tables
                     self._capture_checkpoint(f"layer{layer_idx}_slot_mapping", slot_mapping_buffer, (num_tokens,), np.int32, step_ctx=step_ctx)
                     self._capture_checkpoint(f"layer{layer_idx}_block_table", block_table_buffer, (num_seqs, max_blocks_per_seq), np.int32, step_ctx=step_ctx)
                     self._capture_checkpoint(f"layer{layer_idx}_token_to_seq", token_to_seq_buffer, (num_tokens,), np.int32, step_ctx=step_ctx)
                     self._capture_checkpoint(f"layer{layer_idx}_positions", positions_buffer, (num_tokens,), np.int32, step_ctx=step_ctx)
                     
                     # Capture First 2 Blocks of KV Cache (to verify write)
                     # Size: 2 * block_size * num_kv_heads * head_size
                     cache_block_size = self._kv_cache.block_size * md.num_kv_heads * md.head_size
                     self._capture_checkpoint(f"layer{layer_idx}_k_cache_block0", k_cache, (2*cache_block_size,), np.float16, step_ctx=step_ctx)
                     self._capture_checkpoint(f"layer{layer_idx}_v_cache_block0", v_cache, (2*cache_block_size,), np.float16, step_ctx=step_ctx)

                     # Capture Q and K inputs from qkv_buffer
                     q_size = num_tokens * md.num_attention_heads * md.head_size
                     k_size = num_tokens * md.num_kv_heads * md.head_size
                     self._capture_checkpoint(f"layer{layer_idx}_q_input_attn", qkv_buffer, (q_size,), np.float16, offset=0, step_ctx=step_ctx)
                     self._capture_checkpoint(f"layer{layer_idx}_k_input_write", qkv_buffer, (k_size,), np.float16, offset=k_offset_bytes, step_ctx=step_ctx)

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


        # BARRIER 2: Attention complete (always needed before O-proj)
        step_ctx.memory_barrier()

        if CHECKPOINT_DEBUG_ENABLED:
            self._capture_checkpoint(
                 name=f"layer{layer_idx}_attn_context",
                 buffer=attn_output_buffer,
                 shape=(num_tokens, md.hidden_size),
                 dtype=np.float16,
                 step_ctx=step_ctx,
            )

        # 5. O Projection + residual add
        layer_ops.o_proj.encode(
            step_ctx=step_ctx,
            attn_output=attn_output_buffer,
            output_buffer=hidden_states,
            num_tokens=num_tokens,
        )

        if CHECKPOINT_DEBUG_ENABLED:
             self._capture_checkpoint(
                 name=f"layer{layer_idx}_attn_output",
                 buffer=hidden_states,
                 shape=(num_tokens, md.hidden_size),
                 dtype=np.float16,
                 step_ctx=step_ctx,
            )

        # Add residual
        self._elementwise.encode_residual_add(
            step_ctx=step_ctx,
            x=hidden_states,
            residual=residual_buffer,
            output=hidden_states,
            num_elements=num_tokens * md.hidden_size,
        )

        # BARRIER 3: O-proj + residual complete (always needed before post-attn norm)
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

        # Barrier after post-attn norm: Skip for batch=1 decode (coalesce with MLP)
        if not use_reduced_barriers:
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

        # Barrier after MLP: Skip for batch=1 decode (coalesce with final residual)
        if not use_reduced_barriers:
            step_ctx.memory_barrier()

        # 9. Final residual add
        if CHECKPOINT_DEBUG_ENABLED:
             self._capture_checkpoint(
                 name=f"layer{layer_idx}_mlp_down",
                 buffer=mlp_output_buffer,
                 shape=(num_tokens, md.hidden_size),
                 dtype=np.float16,
                 step_ctx=step_ctx,
             )

        self._elementwise.encode_residual_add(
            step_ctx=step_ctx,
            x=mlp_output_buffer,
            residual=residual_buffer,
            output=hidden_states,
            num_elements=num_tokens * md.hidden_size,
        )

        # BARRIER 4: Layer complete (always needed before next layer)
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

        # Debug: print logits stats
        if self._step_counter <= 2:
            import os
            if os.environ.get('VLLM_ENGINE_DEBUG'):
                logger.info(
                    f"[DEBUG] Logits stats (step {self._step_counter}): "
                    f"shape={np_array.shape}, min={np_array.min():.4f}, max={np_array.max():.4f}, "
                    f"mean={np_array.mean():.4f}, std={np_array.std():.4f}, "
                    f"nan_count={np.isnan(np_array).sum()}, inf_count={np.isinf(np_array).sum()}"
                )
                # Show top-5 tokens for LAST position (used in prefill)
                last_row = np_array[-1]
                top5_idx = np.argsort(last_row)[-5:][::-1]
                top5_vals = last_row[top5_idx]
                logger.info(f"[DEBUG] Top-5 tokens (last pos, idx {num_tokens-1}): {list(zip(top5_idx.tolist(), top5_vals.tolist()))}")
                # Also show first position for reference
                if num_tokens > 1:
                    first_row = np_array[0]
                    top5_idx_first = np.argsort(first_row)[-5:][::-1]
                    top5_vals_first = first_row[top5_idx_first]
                    logger.info(f"[DEBUG] Top-5 tokens (first pos, idx 0): {list(zip(top5_idx_first.tolist(), top5_vals_first.tolist()))}")

        # Convert to torch
        logits = torch.from_numpy(np_array)
        return logits

    def _readback_topk_logits(
        self, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read top-k logits from GPU buffers and reconstruct full logits.

        This method reads only the top-k indices and values (~300 bytes per token
        for k=50) instead of full logits (~256KB per token for 128K vocab),
        achieving ~800x reduction in GPU-to-CPU data transfer.

        Args:
            num_tokens: Number of tokens

        Returns:
            Tuple of:
                - logits: Full logits tensor [num_tokens, vocab_size] with -inf for non-top-k
                - topk_indices: Top-k indices [num_tokens, k] (int32)
                - topk_values: Top-k values [num_tokens, k] (float16)
        """
        k = self._topk_k
        vocab_size = self.model_desc.vocab_size

        # Read top-k indices (int32)
        indices_size_bytes = num_tokens * k * 4
        indices_view = self._topk_indices_buffer.contents().as_buffer(indices_size_bytes)
        indices_np = np.frombuffer(indices_view, dtype=np.int32).reshape(num_tokens, k).copy()

        # Read top-k values (float16)
        values_size_bytes = num_tokens * k * 2
        values_view = self._topk_values_buffer.contents().as_buffer(values_size_bytes)
        values_np = np.frombuffer(values_view, dtype=np.float16).reshape(num_tokens, k).copy()

        # Reconstruct full logits with -inf for non-top-k positions
        # This preserves API compatibility while still benefiting from reduced transfer
        logits_np = np.full((num_tokens, vocab_size), -np.inf, dtype=np.float16)
        for i in range(num_tokens):
            logits_np[i, indices_np[i]] = values_np[i]

        # Convert to torch tensors
        logits = torch.from_numpy(logits_np)
        topk_indices = torch.from_numpy(indices_np)
        topk_values = torch.from_numpy(values_np)

        return logits, topk_indices, topk_values

    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        return {
            "step_count": self._step_counter,
            "num_layers": self.model_desc.num_layers,
            "hidden_size": self.model_desc.hidden_size,
            "vocab_size": self.model_desc.vocab_size,
        }

    # ========================================================================
    # Debug checkpoint capture (VLLM_PREFILL_EQ_DEBUG=1)
    # ========================================================================

    def _capture_checkpoint(
        self,
        name: str,
        buffer: Any,  # MTLBuffer
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float16,
        offset: int = 0,
        last_token_only: bool = True,
        num_tokens: Optional[int] = None,
        step_ctx: Optional[Any] = None,
    ) -> None:
        """Capture a checkpoint from MTLBuffer for debugging.

        This method is only active when CHECKPOINT_DEBUG_ENABLED=True.
        
        Refactored to use DEFERRED CAPTURE:
        1. If step_ctx is provided (Execution Phase):
           - Allocate a shadow buffer
           - Encode a BLIT command to copy data GPU-side
           - Add to pending list
        2. If step_ctx is None:
           - Immediate read (Warning: unsafe if GPU busy)
        """
        if not CHECKPOINT_DEBUG_ENABLED:
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DEBUG: Requesting capture checkpoint {name}")
        
        try:
            # Calculate size
            size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

            if step_ctx is not None:
                # DEFERRED CAPTURE (Safe)
                # 1. Allocate shadow buffer
                shadow_buffer = self._context.create_buffer(size_bytes + offset, "shared")
                
                # 2. Get Blit Encoder (requires ending compute encoder first)
                # Note: step_ctx.end_compute_encoder_for_mps() is for MPS, but works here too
                # to get the command buffer.
                # However, step_ctx has get_compute_encoder() which re-opens.
                # We need to manually handle the blit encoder.
                
                if step_ctx._encoder is not None:
                    step_ctx._encoder.endEncoding()
                    step_ctx._encoder = None
                
                cmd_buffer = step_ctx.command_buffer
                blit = cmd_buffer.blitCommandEncoder()
                
                if blit is None:
                    raise RuntimeError("Failed to create blit encoder")

                # 3. Encode Copy
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                    buffer, offset, shadow_buffer, offset, size_bytes
                )
                blit.endEncoding()
                
                # 4. Restore Compute Encoder for subsequent ops
                step_ctx.get_compute_encoder()
                
                # 5. Add to pending list
                self._pending_checkpoints.append({
                    "name": name,
                    "buffer": shadow_buffer,
                    "shape": shape,
                    "dtype": dtype,
                    "offset": offset,
                    "last_token_only": last_token_only,
                    "num_tokens": num_tokens
                })
                
            else:
                # IMMEDIATE READ (Unsafe if GPU pending)
                # Fallback for initialization time or non-step Context
                logger.warning(f"Immediate capture for {name} (no step_ctx). May be zero.")
                self._process_checkpoint_read(name, buffer, shape, dtype, offset, last_token_only, num_tokens)

        except Exception as e:
            logger.warning(f"Failed to capture checkpoint '{name}': {e}")

    def _process_checkpoint_read(self, name, buffer, shape, dtype, offset, last_token_only, num_tokens):
        """Process the actual readback of a buffer."""
        from vllm_apple.debug import capture_checkpoint
        try:
            # Calculate size
            size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
            
            # Read from buffer
            buffer_view = buffer.contents().as_buffer(offset + size_bytes)
            np_array = np.frombuffer(buffer_view, dtype=dtype, offset=offset).reshape(shape).copy()

            # Convert to torch tensor
            tensor = torch.from_numpy(np_array)

            # If last_token_only, extract just that position
            token_idx = None
            if last_token_only and num_tokens is not None and num_tokens > 0:
                token_idx = num_tokens - 1

            # Store checkpoint
            capture_checkpoint(name, tensor, source="engine", token_idx=token_idx)
            # logger.info(f"DEBUG: Captured {name}") # Too verbose? 
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to process readback {name}: {e}")


    def _init_checkpoint_context(self, num_tokens: int) -> None:
        """Initialize checkpoint store context for this step.

        Args:
            num_tokens: Number of tokens in this step
        """
        if not CHECKPOINT_DEBUG_ENABLED:
            return

        try:
            from vllm_apple.debug import get_checkpoint_store, reset_stores

            # Reset stores for new comparison
            reset_stores()

            # Set up engine store context
            store = get_checkpoint_store("engine")
            store.set_context(
                source="engine",
                num_layers=self.model_desc.num_layers,
                last_pos=num_tokens - 1 if num_tokens > 0 else 0,
            )
            logger.info(
                f"[CHECKPOINT] Debug capture enabled for step {self._step_counter}, "
                f"num_tokens={num_tokens}, num_layers={self.model_desc.num_layers}"
            )
            # Reset stores for new comparison
            reset_stores()

        except Exception as e:
            logger.warning(f"Failed to init checkpoint context: {e}")

    def _cpu_rope(
        self,
        step_ctx: EngineStepContext,
        qkv_buffer: Any,
        positions_buffer: Any,
        num_tokens: int,
        layer_idx: int,
    ) -> None:
        """Apply RoPE on CPU (Isolation Toggle)."""
        md = self.model_desc
        head_size = md.head_size
        num_heads = md.num_attention_heads
        num_kv_heads = md.num_kv_heads
        
        # Read buffers
        # Note: reading entire buffer, not just the slice for num_tokens if flattened?
        # buffers are scratch buffers sized for max_tokens.
        # We need to read just num_tokens.
        qkv_elements = num_tokens * (num_heads + 2 * num_kv_heads) * head_size
        qkv_size_bytes = qkv_elements * 2
        
        # Read QKV
        qkv_view = qkv_buffer.contents().as_buffer(qkv_size_bytes)
        qkv_np = np.frombuffer(qkv_view, dtype=np.float16).copy()
        
        # Read Positions
        pos_size_bytes = num_tokens * 4
        pos_view = positions_buffer.contents().as_buffer(pos_size_bytes)
        positions = np.frombuffer(pos_view, dtype=np.int32).copy()
        
        # Planar Layout: [All Q] [All K] [All V]
        q_elements = num_tokens * num_heads * head_size
        k_elements = num_tokens * num_kv_heads * head_size
        v_elements = num_tokens * num_kv_heads * head_size
        
        # Slices
        q_part = qkv_np[:q_elements]
        k_part = qkv_np[q_elements : q_elements + k_elements]
        v_part = qkv_np[q_elements + k_elements :]
        
        # Reshape to [tokens, heads, dim]
        q = torch.from_numpy(q_part).view(num_tokens, num_heads, head_size).float()
        k = torch.from_numpy(k_part).view(num_tokens, num_kv_heads, head_size).float()
        # v is untouched by RoPE, but we must preserve it for writeback
        
        # Prepare RoPE - HALF-HALF Rotation (Neox/Llama/Qwen style)
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Basic RoPE calculation:
        # q_embed = (q * cos) + (rotate_half(q) * sin)
        
        theta = md.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        # Ensure positions is float for matmul
        t = torch.from_numpy(positions).float()[:, None] * inv_freq[None, :]  # [tokens, head_size/2]
        emb = torch.cat((t, t), dim=-1)  # [tokens, head_size]
        
        cos = emb.cos().to(dtype=torch.float16)
        sin = emb.sin().to(dtype=torch.float16)

        # Reshape cos/sin for broadcasting [tokens, 1, head_size]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Apply RoPE
        q_rope = (q * cos) + (rotate_half(q) * sin)
        k_rope = (k * cos) + (rotate_half(k) * sin)
        
        # IMPORTANT: Write back in PLANAR layout
        # [Q_rope] [K_rope] [V_original]
        # v_part is already numpy array of correct size/offset
        
        q_out_np = q_rope.flatten().to(torch.float16).numpy()
        k_out_np = k_rope.flatten().to(torch.float16).numpy()
        
        # Write back to qkv_view
        # Slice view
        view_ptr = memoryview(qkv_view)
        
        # Q write
        q_bytes = q_elements * 2
        view_ptr[:q_bytes] = q_out_np.tobytes()
        
        # K write
        k_bytes = k_elements * 2
        view_ptr[q_bytes : q_bytes + k_bytes] = k_out_np.tobytes()
        
        # V is unchanged, so no write needed (it was copied from buffer originally? No, qkv_view is slice of shared buffer?)
        # qkv_np was COPY. qkv_view is direct buffer access?
        # "qkv_view = qkv_buffer.contents().as_buffer()" -> Direct memory.
        # "qkv_np = np.frombuffer(...).copy()" -> Copy.
        # So V in buffer is untouched. We only need to write Q and K.
        # Done.
        
        logger.info(f"Layer {layer_idx}: CPU RoPE Applied (Half-Half) - Max diff Q: {(q_rope-q).abs().max()}")


    def _cpu_sdpa(
        self,
        step_ctx: Any,
        qkv_buffer: Any,
        output_buffer: Any,
        num_tokens: int,
        layer_idx: int,
        num_seqs: int,
        max_seq_len: int,
        token_to_seq_buffer: Any,
    ) -> None:
        """Apply SDPA on CPU (Isolation Toggle)."""
        md = self.model_desc
        head_size = md.head_size
        num_heads = md.num_attention_heads
        num_kv_heads = md.num_kv_heads
        scale = 1.0 / (head_size ** 0.5)
        
        # Read QKV
        qkv_elements = num_tokens * (num_heads + 2 * num_kv_heads) * head_size
        qkv_size_bytes = qkv_elements * 2
        qkv_view = qkv_buffer.contents().as_buffer(qkv_size_bytes)
        qkv_np = np.frombuffer(qkv_view, dtype=np.float16).copy()
        qkv = torch.from_numpy(qkv_np).view(num_tokens, num_heads + 2 * num_kv_heads, head_size)
        
        q = qkv[:, :num_heads, :].float()
        k = qkv[:, num_heads:num_heads+num_kv_heads, :].float()
        v = qkv[:, num_heads+num_kv_heads:, :].float()
        
        # Read token_to_seq
        tts_size_bytes = num_tokens * 4
        tts_view = token_to_seq_buffer.contents().as_buffer(tts_size_bytes)
        token_to_seq = np.frombuffer(tts_view, dtype=np.int32).copy()
        
        # Output container
        output = torch.zeros(num_tokens, num_heads, head_size, dtype=torch.float32)
        
        # Iterate sequences
        unique_seqs = np.unique(token_to_seq)
        for seq_id in unique_seqs:
            # Find tokens for seq i
            # Assuming contiguous because prefill
            indices = np.where(token_to_seq == seq_id)[0]
            if len(indices) == 0: continue
            
            start = indices[0]
            end = indices[-1] + 1
            
            # Slice: [L, H, D] -> [1, H, L, D]
            s_q = q[start:end].permute(1, 0, 2).unsqueeze(0) 
            s_k = k[start:end].permute(1, 0, 2).unsqueeze(0)
            s_v = v[start:end].permute(1, 0, 2).unsqueeze(0)
            
            if num_kv_heads != num_heads:
                 s_k = s_k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                 s_v = s_v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            
            # SDPA
            s_out = torch.nn.functional.scaled_dot_product_attention(
                s_q, s_k, s_v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale
            )
            
            # [1, H, L, D] -> [L, H, D]
            output[start:end] = s_out.squeeze(0).permute(1, 0, 2)
            
        # Write back
        out_elements = num_tokens * num_heads * head_size
        out_size_bytes = out_elements * 2
        out_view = output_buffer.contents().as_buffer(out_size_bytes)
        np.copyto(np.frombuffer(out_view, dtype=np.float16), output.flatten().to(torch.float16).numpy())
        
        logger.info(f"DEBUG: Applied PyTorch SDPA for layer {layer_idx}")
