# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Descriptors for vLLM-Apple Metal Engine v2.0.

This module provides dataclasses that describe engine execution:
- StepDescriptor: Describes a single scheduler step
- BatchDescriptor: Describes a batch of sequences
- EngineInputs: CPU tensors passed to the engine
- EngineOutputs: CPU tensors returned from the engine

These descriptors form the contract between vLLM (scheduler/sampler) and
the engine. All data crossing the vLLM↔engine boundary uses these types.

Usage:
    # Created by vLLM scheduler
    step_desc = StepDescriptor(
        step_id=0,
        step_kind="decode",
        num_scheduled_tokens=4,
        num_seqs_active=4,
    )

    # CPU tensors for engine input
    inputs = EngineInputs(
        token_ids=token_ids_tensor,  # CPU tensor
        positions=positions_tensor,
        block_table=block_table_tensor,
        slot_mapping=slot_mapping_tensor,
        seq_lens=seq_lens_tensor,
    )

    # Execute engine step
    outputs = engine.execute_step(step_desc, inputs)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch


@dataclass
class StepDescriptor:
    """Describes a single engine step.

    A step is the unit of work for one scheduler iteration. The engine
    processes all sequences in a step with a single command buffer.

    Attributes:
        step_id: Unique identifier for this step
        step_kind: "prefill" or "decode"
        num_scheduled_tokens: Total tokens scheduled in this step
        num_seqs_active: Number of active sequences
        max_num_blocks_per_seq: Maximum blocks for any sequence
        is_first_step: Whether this is the first step for a request
        cache_enabled: Whether KV cache is enabled
    """

    step_id: int
    step_kind: str  # "prefill" or "decode"
    num_scheduled_tokens: int
    num_seqs_active: int
    max_num_blocks_per_seq: int = 0
    is_first_step: bool = False
    cache_enabled: bool = True

    def __post_init__(self):
        if self.step_kind not in ("prefill", "decode"):
            raise ValueError(f"step_kind must be 'prefill' or 'decode', got '{self.step_kind}'")
        if self.num_scheduled_tokens < 0:
            raise ValueError(f"num_scheduled_tokens must be >= 0, got {self.num_scheduled_tokens}")
        if self.num_seqs_active < 0:
            raise ValueError(f"num_seqs_active must be >= 0, got {self.num_seqs_active}")

    @property
    def is_prefill(self) -> bool:
        """Check if this is a prefill step."""
        return self.step_kind == "prefill"

    @property
    def is_decode(self) -> bool:
        """Check if this is a decode step."""
        return self.step_kind == "decode"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_kind": self.step_kind,
            "num_scheduled_tokens": self.num_scheduled_tokens,
            "num_seqs_active": self.num_seqs_active,
            "max_num_blocks_per_seq": self.max_num_blocks_per_seq,
            "is_first_step": self.is_first_step,
            "cache_enabled": self.cache_enabled,
        }


@dataclass
class BatchDescriptor:
    """Describes a batch of sequences in a step.

    This provides additional detail about the sequences being processed
    in a step, useful for debugging and profiling.

    Attributes:
        seq_ids: List of sequence IDs in the batch
        seq_lens: List of current sequence lengths
        query_lens: List of query lengths (tokens being processed)
        num_prompt_tokens: Total prompt tokens (prefill)
        num_decode_tokens: Total decode tokens
    """

    seq_ids: List[int] = field(default_factory=list)
    seq_lens: List[int] = field(default_factory=list)
    query_lens: List[int] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def num_seqs(self) -> int:
        """Number of sequences in the batch."""
        return len(self.seq_ids)

    @property
    def total_tokens(self) -> int:
        """Total tokens to process."""
        return self.num_prompt_tokens + self.num_decode_tokens

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length in batch."""
        return max(self.seq_lens) if self.seq_lens else 0


@dataclass
class EngineInputs:
    """Input tensors for engine step execution.

    All tensors MUST be on CPU device - this is a core v2.0 invariant.
    The engine will validate this at the boundary.

    Attributes:
        token_ids: Token IDs to process [num_tokens] or [num_seqs, seq_len]
        positions: Position IDs [num_tokens] or [num_seqs, seq_len]
        block_table: Block table for paged attention [num_seqs, max_blocks]
        slot_mapping: Slot mapping for KV cache writes [num_tokens]
        seq_lens: Current sequence lengths [num_seqs]
        query_start_locs: Query start locations for variable-length [num_seqs + 1]
        max_decode_seq_len: Maximum decode sequence length
        encoder_outputs: Optional encoder outputs for encoder-decoder models
    """

    token_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_locs: Optional[torch.Tensor] = None
    max_decode_seq_len: int = 0
    encoder_outputs: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate that all tensors are on CPU."""
        self._validate_cpu_device()

    def _validate_cpu_device(self) -> None:
        """Validate all tensors are on CPU device."""
        tensors = [
            ("token_ids", self.token_ids),
            ("positions", self.positions),
            ("block_table", self.block_table),
            ("slot_mapping", self.slot_mapping),
            ("seq_lens", self.seq_lens),
        ]
        if self.query_start_locs is not None:
            tensors.append(("query_start_locs", self.query_start_locs))
        if self.encoder_outputs is not None:
            tensors.append(("encoder_outputs", self.encoder_outputs))

        for name, tensor in tensors:
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"EngineInputs.{name} must be on CPU device, got {tensor.device}. "
                    f"Engine boundary requires CPU tensors."
                )

    @property
    def num_tokens(self) -> int:
        """Number of tokens in input."""
        return self.token_ids.numel()

    @property
    def num_seqs(self) -> int:
        """Number of sequences."""
        return self.seq_lens.numel()

    @property
    def device(self) -> torch.device:
        """Device of input tensors (should always be CPU)."""
        return self.token_ids.device

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary of tensors."""
        result = {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "block_table": self.block_table,
            "slot_mapping": self.slot_mapping,
            "seq_lens": self.seq_lens,
        }
        if self.query_start_locs is not None:
            result["query_start_locs"] = self.query_start_locs
        if self.encoder_outputs is not None:
            result["encoder_outputs"] = self.encoder_outputs
        return result


@dataclass
class EngineOutputs:
    """Output tensors from engine step execution.

    All tensors are on CPU device after readback from GPU.

    Attributes:
        logits: Output logits [num_tokens, vocab_size] or [num_seqs, vocab_size]
        hidden_states: Optional hidden states (for certain use cases)
        sampled_token_ids: Optional pre-sampled token IDs (if GPU sampling)
    """

    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    sampled_token_ids: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate that output tensors are on CPU."""
        if self.logits.device.type != "cpu":
            raise ValueError(
                f"EngineOutputs.logits must be on CPU device, got {self.logits.device}. "
                f"Engine outputs must be CPU-addressable at step boundary."
            )

    @property
    def vocab_size(self) -> int:
        """Vocabulary size from logits shape."""
        return self.logits.shape[-1]

    @property
    def num_tokens(self) -> int:
        """Number of tokens in output."""
        if self.logits.dim() == 2:
            return self.logits.shape[0]
        return self.logits.numel() // self.vocab_size


@dataclass
class KVCacheDescriptor:
    """Describes the KV cache configuration.

    This is used to configure the EngineKVCache.

    Attributes:
        num_blocks: Total number of cache blocks
        block_size: Tokens per block
        num_kv_heads: Number of key-value heads
        head_size: Size of each attention head
        num_layers: Number of transformer layers
        dtype: Data type (float16, float32)
        device: Device type (should be "metal" for engine mode)
    """

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    num_layers: int
    dtype: torch.dtype = torch.float16
    device: str = "metal"

    def __post_init__(self):
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if self.head_size not in (32, 64, 96, 128):
            raise ValueError(f"head_size must be 32/64/96/128, got {self.head_size}")

    @property
    def block_size_bytes(self) -> int:
        """Size of one cache block in bytes (K + V)."""
        element_size = 2 if self.dtype == torch.float16 else 4
        # K and V each: [num_kv_heads, block_size, head_size]
        single_kv = self.num_kv_heads * self.block_size * self.head_size * element_size
        return single_kv * 2  # K + V

    @property
    def total_cache_bytes(self) -> int:
        """Total cache size in bytes across all layers."""
        return self.block_size_bytes * self.num_blocks * self.num_layers

    @property
    def total_cache_mb(self) -> float:
        """Total cache size in megabytes."""
        return self.total_cache_bytes / (1024 * 1024)


@dataclass
class ModelDescriptor:
    """Describes the model configuration for engine execution.

    Attributes:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        head_size: Size of each attention head
        intermediate_size: MLP intermediate dimension
        vocab_size: Vocabulary size
        dtype: Model data type
        rope_theta: RoPE theta parameter
        max_position_embeddings: Maximum sequence length
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_size: int
    intermediate_size: int
    vocab_size: int
    dtype: torch.dtype = torch.float16
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192

    @property
    def num_query_groups(self) -> int:
        """Number of query groups (for GQA)."""
        return self.num_attention_heads // self.num_kv_heads

    @property
    def is_gqa(self) -> bool:
        """Whether model uses grouped query attention."""
        return self.num_attention_heads != self.num_kv_heads

    def get_kv_cache_descriptor(
        self,
        num_blocks: int,
        block_size: int,
    ) -> KVCacheDescriptor:
        """Create KV cache descriptor for this model.

        Args:
            num_blocks: Number of cache blocks
            block_size: Tokens per block

        Returns:
            KVCacheDescriptor configured for this model
        """
        return KVCacheDescriptor(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            num_layers=self.num_layers,
            dtype=self.dtype,
        )

    @classmethod
    def from_hf_config(cls, config: Any) -> "ModelDescriptor":
        """Create from HuggingFace config.

        Args:
            config: HuggingFace model config

        Returns:
            ModelDescriptor
        """
        return cls(
            num_layers=getattr(config, "num_hidden_layers", config.num_layers),
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_size=config.hidden_size // config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
        )
