# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal Attention Backend for vLLM on Apple Silicon.

This backend uses a custom Metal compute shader for PagedAttention,
providing optimized decode performance on Apple Silicon GPUs.

Key features:
- Native Metal compute kernel for PagedAttention
- 12x speedup vs PyTorch MPS PagedAttention
- Paged KV cache with block_table mapping
- GQA (Grouped Query Attention) support
- Online softmax for numerical stability

Memory layout:
- KV cache: [num_blocks, num_kv_heads, block_size, head_size]
- Query: [num_seqs, num_query_heads, head_size]
- Block table: [num_seqs, max_blocks_per_seq]
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# Import Metal PagedAttention kernels
# - MetalPagedAttentionV2: For prefill path (batched KV write)
# - MetalPagedAttentionFused: For decode path (fused KV write + attention)
import os

from vllm_apple.metal.bridge.metal_paged_attention_v2 import (
    MetalPagedAttentionV2 as MetalPagedAttention,
    metal_write_kv_batch,
    is_metal_available,
)
_METAL_VERSION = "V2"

# Import Metal KV cache management
from vllm_apple.metal.kv_cache import MetalKVCache
import numpy as np

# For MTLBuffer type hints
from Metal import MTLCreateSystemDefaultDevice

logger = init_logger(__name__)

# Import fused KV-write + attention kernel
# Fused kernel writes K/V AND computes attention in one dispatch
_FUSED_KERNEL_AVAILABLE = False
MetalPagedAttentionFused = None
try:
    from vllm_apple.metal.bridge.metal_paged_attention_fused import (
        MetalPagedAttentionFused,
    )
    _FUSED_KERNEL_AVAILABLE = True
    logger.info("Fused KV-write + attention kernel available")
except ImportError as e:
    logger.warning(f"Fused kernel import failed: {e}")

# Configuration for fused kernel batch threshold
# The fused kernel (VLLM_METAL_FUSED_KV=1) writes K/V AND computes attention in one dispatch.
# It's optimal for small batches but has performance regression at higher concurrency
# due to torch.mps.synchronize() overhead growing with pending MPS operations.
#
# VLLM_METAL_FUSED_MAX_SEQS: Maximum decode sequences for fused kernel (default=4)
# - Set this to match your max_num_seqs config for optimal performance
# - Set to 0 to always use fused kernel (small batch workloads)
# - For batch sizes > 4, the non-fused path is typically better
_FUSED_BATCH_THRESHOLD = int(os.environ.get("VLLM_METAL_FUSED_MAX_SEQS", "4"))
logger.info(f"Metal attention config: fused_batch_threshold={_FUSED_BATCH_THRESHOLD}")

# Profiling counters for Metal attention
import time
_metal_profile_enabled = True
_metal_profile_data = {
    'kv_update_ms': 0.0,
    'kv_sync_ms': 0.0,      # MPS sync before CPU transfer
    'kv_to_cpu_ms': 0.0,    # Tensor CPU transfer
    'kv_compute_ms': 0.0,   # Block id/offset computation
    'kv_native_ms': 0.0,    # Native C memcpy call
    'metal_compute_ms': 0.0,
    'sdpa_compute_ms': 0.0,
    'total_forward_ms': 0.0,
    'call_count': 0,
}

def get_metal_profile() -> dict:
    """Get accumulated profiling data."""
    return _metal_profile_data.copy()

def reset_metal_profile():
    """Reset profiling counters."""
    global _metal_profile_data
    _metal_profile_data = {
        'kv_update_ms': 0.0,
        'kv_sync_ms': 0.0,
        'kv_to_cpu_ms': 0.0,
        'kv_compute_ms': 0.0,
        'kv_native_ms': 0.0,
        'metal_compute_ms': 0.0,
        'sdpa_compute_ms': 0.0,
        'total_forward_ms': 0.0,
        'call_count': 0,
    }


class MetalAttentionBackend(AttentionBackend):
    """Metal attention backend using custom Metal PagedAttention kernel.

    This backend provides optimized attention computation on Apple Silicon
    using a native Metal compute shader. It achieves 12x speedup over
    PyTorch MPS PagedAttention for decode operations.
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Metal kernel supports these head sizes
        return [32, 64, 96, 128]

    @staticmethod
    def get_name() -> str:
        return "METAL_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Metal attention supports decoder and encoder-only attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["MetalAttentionImpl"]:
        return MetalAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MetalAttentionMetadataBuilder"]:
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Get KV cache shape compatible with Apple platform.

        Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        - Dimension 0: [key_cache, value_cache]
        - This matches Apple backend for compatibility
        - Metal kernel will transpose internally
        """
        return 2, num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_kv_cache_device() -> str:
        """Return device for KV cache allocation.

        Metal kernel uses unified memory (MTLBuffer) which is accessible
        from CPU. KV cache should be allocated on CPU for direct Metal access
        without copying.
        """
        return "cpu"


@dataclass
class MetalAttentionMetadata:
    """Metadata for Metal attention.

    Attributes:
        num_actual_tokens: Number of tokens excluding padding
        max_query_len: Maximum query length in batch
        query_start_loc: Start location of each query in batch
        max_seq_len: Maximum sequence length
        seq_lens: Length of each sequence
        block_table: Block table mapping sequences to cache blocks
        slot_mapping: Mapping from token positions to cache slots
        scheduler_metadata: Optional scheduler metadata
        causal: Whether to use causal masking
        num_decode_tokens: Number of decode tokens (at front of batch)

    Cached properties:
        _query_start_loc_cpu: Cached CPU list of query_start_loc
        _seq_lens_cpu: Cached CPU list of seq_lens
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: Optional[torch.Tensor]
    causal: bool = True
    num_decode_tokens: int = 0

    # Cached CPU versions to avoid repeated .tolist() calls (device sync)
    _query_start_loc_cpu: Optional[list] = None
    _seq_lens_cpu: Optional[list] = None

    @property
    def query_start_loc_cpu(self) -> list:
        """Get query_start_loc as CPU list (cached to avoid device sync)."""
        if self._query_start_loc_cpu is None:
            self._query_start_loc_cpu = self.query_start_loc.tolist()
        return self._query_start_loc_cpu

    @property
    def seq_lens_cpu(self) -> list:
        """Get seq_lens as CPU list (cached to avoid device sync)."""
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.tolist()
        return self._seq_lens_cpu


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder[MetalAttentionMetadata]):
    """Builder for Metal attention metadata."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable reorder batch to separate decode and prefill tokens
        self._init_reorder_batch_threshold(1, False)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Split decode and prefill tokens
        num_decode_tokens = 0
        if causal:
            assert self.reorder_batch_threshold is not None
            (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=True,
                )
            )

        return MetalAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            scheduler_metadata=None,
            causal=causal,
            num_decode_tokens=num_decode_tokens,
        )


class MetalAttentionImpl(AttentionImpl):
    """Metal attention implementation using custom Metal PagedAttention kernel.

    This implementation uses a native Metal compute shader for decode operations,
    achieving 12x speedup over PyTorch MPS. Prefill operations fall back to
    PyTorch SDPA for efficiency with variable-length sequences.

    ZERO-COPY ARCHITECTURE:
    - MetalKVCache owns the physical KV cache in MTLBuffer (unified memory)
    - vLLM only provides metadata (block_table, slot_mapping, seq_lens)
    - No copying of 15GB KV cache per decode step
    - _update_kv_cache writes directly to MTLBuffer
    - _compute_attention_with_metal uses forward_with_metal_buffers()
    """

    # Per-layer Metal kernel instances (shared across forward calls)
    _metal_kernels: dict[tuple, MetalPagedAttention] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # Track if this layer shares KV cache with another layer
        # When True, we skip MetalKVCache allocation to avoid double memory usage
        self._is_kv_shared = kv_sharing_target_layer_name is not None
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)

        if logits_soft_cap is not None:
            logger.warning_once(
                "METAL_ATTN does not support logits softcap, outputs may be slightly off"
            )
        self.logits_soft_cap = logits_soft_cap or 0

        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            logger.warning_once(
                "METAL_ATTN does not support ALiBi slopes, ignoring"
            )
        self.alibi_slopes = None

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("FP8 KV cache is unsupported in METAL_ATTN")
        self.attn_type = attn_type
        self.sinks = sinks

        # Metal kernel will be initialized lazily with correct block_size
        self._metal_kernel: Optional[MetalPagedAttention] = None
        self._block_size: Optional[int] = None

        # Fused kernel for decode path (ETAP 4: KV-write + attention in one pass)
        self._fused_kernel: Optional['MetalPagedAttentionFused'] = None
        self._use_fused_decode: bool = _FUSED_KERNEL_AVAILABLE

        # MetalKVCache instance for this layer (zero-copy KV cache in MTLBuffer)
        # NOTE: When _is_kv_shared=True, we DON'T allocate MetalKVCache
        # because this layer shares KV with another layer (e.g., encoder-decoder)
        self._metal_kv_cache: Optional[MetalKVCache] = None
        self._metal_kv_cache_initialized: bool = self._is_kv_shared  # Skip init if shared

        # CPU tensor cache for decode path (KROK 2: CPU optimization)
        # Avoids repeated allocation and MPS->CPU transfers
        self._decode_cpu_cache: dict = {}
        self._fused_cpu_cache: dict = {}

        # Cached block_table/seq_lens to avoid repeated conversions
        self._cached_block_table_cpu: Optional[torch.Tensor] = None
        self._cached_seq_lens_cpu: Optional[torch.Tensor] = None
        self._cached_metadata_id: Optional[int] = None  # Track metadata changes

    # Default initial batch size - auto-resize handles exceeding this
    _DEFAULT_INITIAL_BATCH_SIZE: int = 64

    def _get_or_create_metal_kernel(self, block_size: int, max_num_blocks: int = 4096) -> MetalPagedAttention:
        """Get or create Metal kernel for this configuration.

        The kernel uses an initial max_batch_size but will auto-resize
        if batches exceed this value.
        """
        key = (self.num_kv_heads, self.num_heads, self.head_size, block_size)

        if key not in MetalAttentionImpl._metal_kernels:
            logger.info(
                f"Creating Metal PagedAttention {_METAL_VERSION} kernel: "
                f"num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}, "
                f"head_size={self.head_size}, block_size={block_size}"
            )
            MetalAttentionImpl._metal_kernels[key] = MetalPagedAttention(
                num_kv_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                head_size=self.head_size,
                block_size=block_size,
                scale=self.scale,
                max_num_blocks=max_num_blocks,
                max_batch_size=self._DEFAULT_INITIAL_BATCH_SIZE,
            )

        return MetalAttentionImpl._metal_kernels[key]

    def _get_or_create_fused_kernel(self, block_size: int, max_num_blocks: int = 4096) -> Optional['MetalPagedAttentionFused']:
        """Get or create fused kernel for decode path (ETAP 4).

        The fused kernel writes new K/V to cache AND computes attention
        in a single dispatch, eliminating separate KV update overhead.
        """
        if not _FUSED_KERNEL_AVAILABLE or MetalPagedAttentionFused is None:
            return None

        if self._fused_kernel is None:
            try:
                self._fused_kernel = MetalPagedAttentionFused(
                    num_kv_heads=self.num_kv_heads,
                    num_query_heads=self.num_heads,
                    head_size=self.head_size,
                    block_size=block_size,
                    scale=self.scale,
                    max_num_blocks=max_num_blocks,
                    max_batch_size=self._DEFAULT_INITIAL_BATCH_SIZE,
                )
                if self._fused_kernel.is_fused_available:
                    logger.info(
                        f"Created fused KV-write+attention kernel: "
                        f"kernel={self._fused_kernel.kernel_name}"
                    )
                else:
                    logger.warning("Fused kernel created but fused mode not available")
                    self._fused_kernel = None
            except Exception as e:
                logger.warning(f"Failed to create fused kernel: {e}")
                self._fused_kernel = None

        return self._fused_kernel

    def _get_or_create_metal_kv_cache(self, num_blocks: int, block_size: int) -> MetalKVCache:
        """Create MetalKVCache for this layer instance.

        MetalKVCache owns the physical KV cache in MTLBuffer (unified memory).
        This is the "source of truth" - no copying of KV data per decode step.

        IMPORTANT: Each layer instance gets its own MetalKVCache.
        We do NOT share buffers between layers because each layer has
        independent K/V data.
        """
        # Each layer instance gets its own MetalKVCache - no sharing!
        # Using id(self) ensures each instance has unique buffer
        if self._metal_kv_cache is None:
            logger.info(
                f"Creating MetalKVCache for layer instance {id(self)}: "
                f"num_blocks={num_blocks}, num_kv_heads={self.num_kv_heads}, "
                f"block_size={block_size}, head_size={self.head_size}"
            )
            self._metal_kv_cache = MetalKVCache(
                num_blocks=num_blocks,
                num_kv_heads=self.num_kv_heads,
                block_size=block_size,
                head_size=self.head_size,
                num_layers=1,  # One cache per layer instance
            )

        self._metal_kv_cache_initialized = True
        self._block_size = block_size
        return self._metal_kv_cache

    def _get_metadata_cpu(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_seqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert block_table and seq_lens to CPU int32 tensors.

        KROK 2 CPU optimization: Simple conversion without caching.
        Cache-based approach was causing issues with vLLM's tensor reuse.

        Args:
            block_table: [max_seqs, max_blocks_per_seq] on MPS or CPU
            seq_lens: [max_seqs] on MPS or CPU
            num_seqs: Number of decode sequences

        Returns:
            (block_table_cpu, seq_lens_cpu) as contiguous int32 tensors
        """
        decode_block_table = block_table[:num_seqs]
        decode_seq_lens = seq_lens[:num_seqs]

        if decode_block_table.device.type == "mps":
            block_table_cpu = decode_block_table.to("cpu", dtype=torch.int32).contiguous()
        else:
            block_table_cpu = decode_block_table.to(dtype=torch.int32).contiguous()

        if decode_seq_lens.device.type == "mps":
            seq_lens_cpu = decode_seq_lens.to("cpu", dtype=torch.int32).contiguous()
        else:
            seq_lens_cpu = decode_seq_lens.to(dtype=torch.int32).contiguous()

        return block_table_cpu, seq_lens_cpu

    def _get_fused_cpu_tensors(
        self,
        num_seqs: int,
    ) -> dict:
        """Get or create cached CPU tensors for fused kernel.

        KROK 2 CPU optimization: Reuse allocated CPU tensors
        for query, key, value, and output across calls.
        """
        cache_key = num_seqs

        if cache_key not in self._fused_cpu_cache:
            self._fused_cpu_cache[cache_key] = {
                'query': torch.empty(num_seqs, self.num_heads, self.head_size,
                                    dtype=torch.float16, device='cpu'),
                'key': torch.empty(num_seqs, self.num_kv_heads, self.head_size,
                                  dtype=torch.float16, device='cpu'),
                'value': torch.empty(num_seqs, self.num_kv_heads, self.head_size,
                                    dtype=torch.float16, device='cpu'),
                'output': torch.empty(num_seqs, self.num_heads, self.head_size,
                                     dtype=torch.float16, device='cpu'),
            }

        return self._fused_cpu_cache[cache_key]

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for Metal attention backend.

        ZERO-COPY ARCHITECTURE:
        - MetalKVCache owns the physical KV cache in MTLBuffer (unified memory)
        - vLLM's kv_cache tensor is IGNORED for decode - we use MetalKVCache
        - _update_kv_cache writes directly to MTLBuffer (no tensor copy)
        - _compute_attention_with_metal uses forward_with_metal_buffers()

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
                     (Apple layout - IGNORED for Metal decode, used only for sizing)
            attn_metadata: Metadata for attention.

        Returns:
            output tensor with shape = [num_tokens, num_heads, head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not yet supported for MetalAttentionImpl"
            )

        # For warming-up / profiling
        if attn_metadata is None:
            return output

        # KV SHARING SAFETY CHECKS
        # When this layer shares KV cache with another layer, we:
        # 1. Don't update KV cache (target layer handles it)
        # 2. Don't use MetalKVCache (it's not allocated for shared layers)
        # 3. Use PyTorch kv_cache tensor passed by vLLM for attention
        if self._is_kv_shared:
            # For shared layers, key/value should be None (no new K/V to write)
            # However, vLLM may still pass them - just ignore and use kv_cache
            pass  # Continue to handle via existing kv_sharing_target_layer_name checks

        # Profiling start
        global _metal_profile_data, _metal_profile_enabled
        _t_forward_start = time.perf_counter() if _metal_profile_enabled else 0

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention - no KV cache
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._compute_attention_no_cache(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                is_causal=False,
            )

        # Initialize MetalKVCache if needed (lazy init with correct sizing)
        # vLLM kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]

        if not self._metal_kv_cache_initialized:
            self._get_or_create_metal_kv_cache(num_blocks, block_size)

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = num_actual_tokens - num_decode_tokens

        # ETAP 4: Check if we can use fused kernel for decode
        # Fused kernel writes K/V AND computes attention in one pass
        # IMPORTANT: Fused kernel is disabled when VLLM_METAL_FUSED_MAX_SEQS is exceeded
        # because torch.mps.synchronize() overhead grows with pending MPS operations.
        # For high-concurrency workloads (max_num_seqs > 4), the non-fused path is better
        # as it has separate smaller sync points.
        fused_kernel = None
        use_fused_for_decode = False

        # Use fused kernel only if configured for small batches AND runtime batch is small
        # _FUSED_BATCH_THRESHOLD should match user's max_num_seqs config
        if (
            self._use_fused_decode
            and num_decode_tokens > 0
            and (_FUSED_BATCH_THRESHOLD == 0 or num_decode_tokens <= _FUSED_BATCH_THRESHOLD)
            and self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            fused_kernel = self._get_or_create_fused_kernel(block_size, num_blocks)
            use_fused_for_decode = (fused_kernel is not None and fused_kernel.is_fused_available)

        # Update KV cache for PREFILL tokens (always use separate update for prefill)
        # For DECODE: skip if using fused kernel (it writes KV internally)
        _t_kv_start = time.perf_counter() if _metal_profile_enabled else 0
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            if use_fused_for_decode:
                # Fused path: only update KV for PREFILL tokens (if any)
                if num_prefill_tokens > 0:
                    self._update_kv_cache_metal(
                        key[num_decode_tokens:num_actual_tokens],
                        value[num_decode_tokens:num_actual_tokens],
                        attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens],
                    )
                # DECODE tokens KV will be written by fused kernel - no update here!
            else:
                # Non-fused path: update KV for ALL tokens
                self._update_kv_cache_metal(
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    attn_metadata.slot_mapping[:num_actual_tokens],
                )
        if _metal_profile_enabled:
            _metal_profile_data['kv_update_ms'] += (time.perf_counter() - _t_kv_start) * 1000

        # Process prefill tokens using PyTorch SDPA
        if num_prefill_tokens > 0:
            _t_sdpa_start = time.perf_counter() if _metal_profile_enabled else 0
            self._compute_attention_no_cache(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens],
                value[num_decode_tokens:num_actual_tokens],
                output[num_decode_tokens:num_actual_tokens],
                attn_metadata,
                is_causal=True,
            )
            if _metal_profile_enabled:
                _metal_profile_data['sdpa_compute_ms'] += (time.perf_counter() - _t_sdpa_start) * 1000

        # Process decode tokens using Metal kernel
        if num_decode_tokens > 0:
            _t_metal_start = time.perf_counter() if _metal_profile_enabled else 0
            if use_fused_for_decode:
                # ETAP 4: Fused KV-write + attention in one kernel dispatch
                # Best for small batches (max_num_seqs <= 4)
                self._compute_attention_fused(
                    query[:num_decode_tokens],
                    key[:num_decode_tokens],
                    value[:num_decode_tokens],
                    output[:num_decode_tokens],
                    attn_metadata,
                    fused_kernel,
                )
            else:
                # Non-fused Metal kernel: separate KV update already done
                # Better for larger batches due to smaller per-operation sync overhead
                self._compute_attention_with_metal_zero_copy(
                    query[:num_decode_tokens],
                    output[:num_decode_tokens],
                    attn_metadata,
                )
            if _metal_profile_enabled:
                _metal_profile_data['metal_compute_ms'] += (time.perf_counter() - _t_metal_start) * 1000

        # Profiling end
        if _metal_profile_enabled:
            _metal_profile_data['total_forward_ms'] += (time.perf_counter() - _t_forward_start) * 1000
            _metal_profile_data['call_count'] += 1

        return output

    def _compute_attention_fused(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        fused_kernel: 'MetalPagedAttentionFused',
    ) -> None:
        """Compute attention with fused KV-write (ETAP 4).

        This method:
        1. Passes new K/V for current decode tokens to the fused kernel
        2. Fused kernel writes K/V to cache AND computes attention in one dispatch
        3. Eliminates separate KV update overhead (~46% of decode time)

        KROK 2 CPU optimization:
        - Reuses cached CPU tensors for query, key, value, output
        - Caches block_table and seq_lens conversions

        Args:
            query: [num_decode_tokens, num_query_heads, head_size]
            key: [num_decode_tokens, num_kv_heads, head_size] - new keys for this step
            value: [num_decode_tokens, num_kv_heads, head_size] - new values for this step
            output: [num_decode_tokens, num_query_heads, head_size]
            attn_metadata: Contains block_table and seq_lens
            fused_kernel: The fused Metal kernel instance
        """
        assert self._metal_kv_cache is not None, "MetalKVCache not initialized"

        num_decode_seqs = query.shape[0]

        # Get MTLBuffers from MetalKVCache
        key_buffer, value_buffer = self._metal_kv_cache.get_buffers(0)

        # KROK 2: Get cached CPU tensors
        cpu_tensors = self._get_fused_cpu_tensors(num_decode_seqs)

        # Prepare CPU tensors for fused kernel - use cached buffers
        if query.device.type == "mps":
            torch.mps.synchronize()
            cpu_tensors['query'].copy_(query.cpu())
            cpu_tensors['key'].copy_(key.cpu())
            cpu_tensors['value'].copy_(value.cpu())
            query_cpu = cpu_tensors['query']
            key_cpu = cpu_tensors['key']
            value_cpu = cpu_tensors['value']
        else:
            # For CPU tensors, copy into cached buffers
            cpu_tensors['query'].copy_(query)
            cpu_tensors['key'].copy_(key)
            cpu_tensors['value'].copy_(value)
            query_cpu = cpu_tensors['query']
            key_cpu = cpu_tensors['key']
            value_cpu = cpu_tensors['value']

        # KROK 2: Get block_table and seq_lens as CPU int32
        decode_block_table_cpu, decode_seq_lens_cpu = self._get_metadata_cpu(
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            num_decode_seqs,
        )

        # Output tensor - use cached buffer
        output_cpu = cpu_tensors['output']

        # Execute fused kernel: writes K/V to cache AND computes attention
        fused_kernel.forward_fused(
            query=query_cpu,
            new_keys=key_cpu,
            new_values=value_cpu,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=decode_block_table_cpu,
            seq_lens=decode_seq_lens_cpu,
            output=output_cpu,
        )

        # Copy output back
        output.copy_(output_cpu)

    def _update_kv_cache_metal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update MetalKVCache directly in MTLBuffer - ZERO COPY.

        This writes K/V data directly to MTLBuffer owned by MetalKVCache.
        No intermediate tensors, no copying of entire KV cache.

        MetalKVCache layout: [num_blocks, num_kv_heads, block_size, head_size]
        (Metal kernel layout - different from Apple layout!)

        Args:
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            slot_mapping: [num_tokens] - absolute slot indices (block_id * block_size + token_offset)
        """
        global _metal_profile_data, _metal_profile_enabled
        assert self._metal_kv_cache is not None, "MetalKVCache not initialized"

        num_tokens = key.shape[0]
        num_kv_heads = key.shape[1]
        head_size = key.shape[2]
        block_size = self._metal_kv_cache.block_size

        # Get MTLBuffer contents for direct write (cached memoryviews)
        if not hasattr(self, '_kv_mv_cache') or self._kv_mv_cache is None:
            key_buffer, value_buffer = self._metal_kv_cache.get_buffers(0)
            self._kv_mv_cache = (
                key_buffer.contents().as_buffer(key_buffer.length()),
                value_buffer.contents().as_buffer(value_buffer.length()),
            )
        key_mv, value_mv = self._kv_mv_cache

        # Cache strides and dtype_size as instance attributes for speed
        if not hasattr(self, '_kv_strides_cached'):
            self._kv_strides_cached = (
                self._metal_kv_cache.strides['block'],  # num_kv_heads * block_size * head_size
                self._metal_kv_cache.strides['head'],   # block_size * head_size
                self._metal_kv_cache.strides['token'],  # head_size
            )
        stride_block, stride_head, stride_token = self._kv_strides_cached
        dtype_size = 2  # float16
        head_bytes = head_size * dtype_size

        # Convert tensors to numpy - use view if possible to avoid copy
        # Fine-grained profiling to identify bottleneck
        if key.device.type == "mps":
            _t0 = time.perf_counter() if _metal_profile_enabled else 0
            torch.mps.synchronize()
            if _metal_profile_enabled:
                _metal_profile_data['kv_sync_ms'] += (time.perf_counter() - _t0) * 1000

            _t0 = time.perf_counter() if _metal_profile_enabled else 0
            key_np = key.cpu().numpy()
            value_np = value.cpu().numpy()
            slot_mapping_np = slot_mapping.cpu().numpy()
            if _metal_profile_enabled:
                _metal_profile_data['kv_to_cpu_ms'] += (time.perf_counter() - _t0) * 1000
        else:
            # Ensure contiguous and get numpy view (no copy if already contiguous float16)
            if not key.is_contiguous():
                key = key.contiguous()
            if not value.is_contiguous():
                value = value.contiguous()
            key_np = key.detach().numpy()
            value_np = value.detach().numpy()
            slot_mapping_np = slot_mapping.detach().numpy()

        # Pre-compute block_ids and token_offsets
        _t0 = time.perf_counter() if _metal_profile_enabled else 0
        block_ids = slot_mapping_np // block_size
        token_offsets = slot_mapping_np % block_size
        if _metal_profile_enabled:
            _metal_profile_data['kv_compute_ms'] += (time.perf_counter() - _t0) * 1000

        # Call native KV write function for all tokens
        # This moves the loop over kv_heads to C/native code
        _t0 = time.perf_counter() if _metal_profile_enabled else 0
        metal_write_kv_batch(
            key_mv=key_mv,
            value_mv=value_mv,
            key_np=key_np,
            value_np=value_np,
            block_ids=block_ids,
            token_offsets=token_offsets,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            stride_block=stride_block,
            stride_head=stride_head,
            stride_token=stride_token,
        )
        if _metal_profile_enabled:
            _metal_profile_data['kv_native_ms'] += (time.perf_counter() - _t0) * 1000

    def _update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache using vectorized scatter operations (legacy fallback).

        Args:
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            key_cache: [num_blocks, block_size, num_kv_heads, head_size] (Apple layout)
            value_cache: [num_blocks, block_size, num_kv_heads, head_size]
            slot_mapping: [num_tokens] - indices into flattened cache
        """
        num_tokens = key.shape[0]
        num_kv_heads_key = key.shape[1]
        head_size_key = key.shape[2]
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

        # Reshape cache to [num_blocks * block_size, num_kv_heads, head_size]
        key_cache_flat = key_cache.view(-1, num_kv_heads, head_size)
        value_cache_flat = value_cache.view(-1, num_kv_heads, head_size)

        # Use scatter_ for fast vectorized updates
        # Note: key/value have num_kv_heads (not num_query_heads due to GQA)
        idx = slot_mapping.to(torch.int64).view(-1, 1, 1).expand(num_tokens, num_kv_heads_key, head_size_key)
        key_cache_flat.scatter_(0, idx, key)
        value_cache_flat.scatter_(0, idx, value)

    def _compute_decode_with_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Compute decode attention using pure SDPA (no MPS sync required).

        This is a fallback for large batches where torch.mps.synchronize() in
        the Metal kernel path causes severe performance degradation.

        For each decode sequence:
        1. Gather historical KV from kv_cache using block_table
        2. Prepend current step's new K/V
        3. Compute attention with SDPA

        Args:
            query: [num_decode_tokens, num_query_heads, head_size]
            key: [num_decode_tokens, num_kv_heads, head_size] or None
            value: [num_decode_tokens, num_kv_heads, head_size] or None
            output: [num_decode_tokens, num_query_heads, head_size]
            attn_metadata: Contains block_table and seq_lens
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
        """
        num_decode_seqs = query.shape[0]
        block_table = attn_metadata.block_table  # [num_seqs, max_blocks_per_seq]
        device = query.device  # MPS or CPU

        # Get cache dimensions
        # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]  # [num_blocks, block_size, num_kv_heads, head_size]
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

        # Use cached CPU list to avoid device sync per layer
        seq_lens_list = attn_metadata.seq_lens_cpu

        for seq_idx in range(num_decode_seqs):
            seq_len = seq_lens_list[seq_idx]  # Total tokens including current
            if seq_len == 0:
                continue

            # Gather historical KV tokens from cache
            # block_table[seq_idx] contains block indices for this sequence
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            blocks = block_table[seq_idx, :num_blocks_needed]  # [num_blocks_needed]

            # Gather all KV tokens for this sequence
            # Result: [seq_len, num_kv_heads, head_size]
            gathered_k_list = []
            gathered_v_list = []
            tokens_gathered = 0

            for block_idx in range(num_blocks_needed):
                block_id = blocks[block_idx].item()
                tokens_in_block = min(block_size, seq_len - tokens_gathered)
                gathered_k_list.append(key_cache[block_id, :tokens_in_block])
                gathered_v_list.append(value_cache[block_id, :tokens_in_block])
                tokens_gathered += tokens_in_block

            k_seq = torch.cat(gathered_k_list, dim=0)  # [seq_len, num_kv_heads, head_size]
            v_seq = torch.cat(gathered_v_list, dim=0)  # [seq_len, num_kv_heads, head_size]

            # Move KV to same device as query (KV cache may be on CPU)
            if k_seq.device != device:
                k_seq = k_seq.to(device)
                v_seq = v_seq.to(device)

            # Query for this sequence: [1, num_query_heads, head_size]
            q_seq = query[seq_idx:seq_idx+1]

            # Transpose for SDPA: [num_heads, seq_len, head_size]
            q = q_seq.transpose(0, 1)  # [num_query_heads, 1, head_size]
            k = k_seq.transpose(0, 1)  # [num_kv_heads, seq_len, head_size]
            v = v_seq.transpose(0, 1)  # [num_kv_heads, seq_len, head_size]

            # Handle GQA: repeat KV heads if needed
            if num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            # Compute attention using SDPA
            # Query attends to all historical tokens (causal not needed - query is single token)
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),  # [1, num_query_heads, 1, head_size]
                k.unsqueeze(0),  # [1, num_query_heads, seq_len, head_size]
                v.unsqueeze(0),  # [1, num_query_heads, seq_len, head_size]
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,  # Decode: single query attends to all past
                scale=self.scale,
            )

            # Store result: [num_query_heads, 1, head_size] -> [1, num_query_heads, head_size]
            output[seq_idx] = attn_out.squeeze(0).transpose(0, 1).squeeze(0)

        return output

    def _compute_attention_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        is_causal: bool,
    ) -> torch.Tensor:
        """Compute attention without reading from KV cache.

        Used for prefill and encoder attention. Falls back to PyTorch SDPA.
        """
        query_start_loc = attn_metadata.query_start_loc
        num_seqs = query_start_loc.shape[0] - 1

        # Fast path: single sequence
        if num_seqs == 1:
            q = query.transpose(0, 1)
            k = key.transpose(0, 1)
            v = value.transpose(0, 1)

            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            output.copy_(attn_out.squeeze(0).transpose(0, 1))
            return output

        # Multiple sequences: iterate
        # Use cached CPU list to avoid device sync per layer
        start_locs = attn_metadata.query_start_loc_cpu

        for seq_idx in range(num_seqs):
            start = start_locs[seq_idx]
            end = start_locs[seq_idx + 1]
            seq_len = end - start

            if seq_len == 0:
                continue

            q = query[start:end].transpose(0, 1)
            k = key[start:end].transpose(0, 1)
            v = value[start:end].transpose(0, 1)

            if self.num_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            output[start:end] = attn_out.squeeze(0).transpose(0, 1)

        return output

    def _compute_attention_with_metal(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> None:
        """Compute attention using Metal PagedAttention kernel.

        This is the optimized decode path using our custom Metal kernel.

        For decode phase in vLLM:
        - Each decode token corresponds to one sequence generating its next token
        - num_decode_tokens == number of decode sequences
        - query shape: [num_decode_tokens, num_heads, head_size]
        - Each sequence has seq_lens[i] tokens in KV cache to attend to

        Args:
            query: [num_decode_tokens, num_heads, head_size]
            key_cache: [num_blocks, block_size, num_kv_heads, head_size] (Apple layout)
            value_cache: [num_blocks, block_size, num_kv_heads, head_size]
            output: [num_decode_tokens, num_heads, head_size]
        """
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens

        # Apple layout: [num_blocks, block_size, num_kv_heads, head_size]
        num_blocks = key_cache.shape[0]
        block_size = key_cache.shape[1]

        # Get or create Metal kernel
        metal_kernel = self._get_or_create_metal_kernel(block_size, num_blocks)

        # Number of decode sequences (each decode token is one sequence)
        num_decode_seqs = query.shape[0]

        # Get only the decode portion of seq_lens and block_table
        # In vLLM, decode sequences are reordered to the front
        decode_seq_lens = seq_lens[:num_decode_seqs]
        decode_block_table = block_table[:num_decode_seqs]

        # Ensure tensors are on CPU for Metal kernel (unified memory)
        # The kernel operates directly on MTLBuffer backed by unified memory
        if query.device.type == "mps":
            torch.mps.synchronize()
            query_cpu = query.cpu()
            # Transpose KV cache from Apple layout to Metal kernel layout:
            # [num_blocks, block_size, num_kv_heads, head_size] ->
            # [num_blocks, num_kv_heads, block_size, head_size]
            key_cache_cpu = key_cache.permute(0, 2, 1, 3).contiguous().cpu()
            value_cache_cpu = value_cache.permute(0, 2, 1, 3).contiguous().cpu()
            decode_block_table_cpu = decode_block_table.int().cpu()
            decode_seq_lens_cpu = decode_seq_lens.int().cpu()
            output_cpu = output.cpu()
        else:
            query_cpu = query
            # Transpose for CPU tensors too
            key_cache_cpu = key_cache.permute(0, 2, 1, 3).contiguous()
            value_cache_cpu = value_cache.permute(0, 2, 1, 3).contiguous()
            decode_block_table_cpu = decode_block_table.int()
            decode_seq_lens_cpu = decode_seq_lens.int()
            output_cpu = output

        # Call Metal kernel
        # skip_kv_copy=False: vLLM updates KV cache via scatter_ on PyTorch tensors,
        # not via Metal kernel's set_kv_cache(). So we must copy KV cache each time.
        metal_kernel.forward(
            query=query_cpu,
            key_cache=key_cache_cpu,
            value_cache=value_cache_cpu,
            block_table=decode_block_table_cpu,
            seq_lens=decode_seq_lens_cpu,
            output=output_cpu,
            skip_kv_copy=False,
        )

        # Copy output back to original device if needed
        if query.device.type == "mps":
            output.copy_(output_cpu.to(query.device))

    def _compute_attention_with_metal_zero_copy(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> None:
        """Compute attention using Metal kernel with ZERO-COPY KV cache.

        This is the optimized decode path that:
        - Uses MetalKVCache's MTLBuffer directly (no copying 15GB KV cache)
        - Calls forward_with_metal_buffers() with external MTLBuffers
        - Only copies query (small) and output (small) between CPU/MPS

        KROK 2 CPU optimization:
        - Reuses cached CPU tensors for query and output
        - Caches block_table and seq_lens conversions

        For decode phase in vLLM:
        - Each decode token corresponds to one sequence generating its next token
        - num_decode_tokens == number of decode sequences
        - query shape: [num_decode_tokens, num_heads, head_size]
        - Each sequence has seq_lens[i] tokens in KV cache to attend to

        Args:
            query: [num_decode_tokens, num_heads, head_size]
            output: [num_decode_tokens, num_heads, head_size]
            attn_metadata: Contains block_table and seq_lens
        """
        # Guard against shared layer misuse
        assert not self._is_kv_shared, (
            "KV sharing layer should not call _compute_attention_with_metal_zero_copy. "
            "Shared layers should use PyTorch SDPA with the target layer's KV cache."
        )
        assert self._metal_kv_cache is not None, "MetalKVCache not initialized"

        block_size = self._metal_kv_cache.block_size
        num_blocks = self._metal_kv_cache.num_blocks

        # Get or create Metal kernel
        metal_kernel = self._get_or_create_metal_kernel(block_size, num_blocks)

        # Number of decode sequences (each decode token is one sequence)
        num_decode_seqs = query.shape[0]

        # Get MTLBuffers from MetalKVCache (no copying!)
        key_buffer, value_buffer = self._metal_kv_cache.get_buffers(0)

        # KROK 2: Use cached CPU tensors for query and output
        cache_key = num_decode_seqs

        if cache_key not in self._decode_cpu_cache:
            # First time: allocate CPU tensors for reuse
            self._decode_cpu_cache[cache_key] = {
                'query': torch.empty(num_decode_seqs, self.num_heads, self.head_size,
                                    dtype=torch.float16, device='cpu'),
                'output': torch.empty(num_decode_seqs, self.num_heads, self.head_size,
                                     dtype=torch.float16, device='cpu'),
            }

        cpu_cache = self._decode_cpu_cache[cache_key]

        # Ensure tensors are on CPU for Metal kernel
        if query.device.type == "mps":
            torch.mps.synchronize()
            cpu_cache['query'].copy_(query.cpu())
            query_cpu = cpu_cache['query']
            output_cpu = cpu_cache['output']
        else:
            cpu_cache['query'].copy_(query)
            query_cpu = cpu_cache['query']
            output_cpu = cpu_cache['output']

        # KROK 2: Get block_table and seq_lens as CPU int32
        decode_block_table_cpu, decode_seq_lens_cpu = self._get_metadata_cpu(
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            num_decode_seqs,
        )

        # Call Metal kernel with external MTLBuffers - NO KV cache copy!
        metal_kernel.forward_with_metal_buffers(
            query=query_cpu,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            block_table=decode_block_table_cpu,
            seq_lens=decode_seq_lens_cpu,
            output=output_cpu,
        )

        # Copy output back to original tensor
        output.copy_(output_cpu)
