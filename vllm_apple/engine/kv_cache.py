# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine KV Cache for vLLM-Apple Metal Engine v2.0.

This module provides the engine-owned KV cache that is the single source of
truth for KV state. The engine owns MTLBuffer storage directly - vLLM's KV
tensors become lightweight stubs that reference engine-owned data.

Key Features:
- Single MTLBuffer per layer for K and V (unified memory)
- Block table validation at step boundary (not in hot path)
- Encode-only KV write operations (no internal waits)
- Integration with MetalEngineContext for device management

Layout:
    [num_blocks, num_kv_heads, block_size, head_size]
    - num_blocks: Total cache blocks
    - num_kv_heads: Number of KV heads (supports GQA)
    - block_size: Tokens per block (typically 16)
    - head_size: Dimension per head (32, 64, 96, or 128)

Usage:
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor

    desc = KVCacheDescriptor(
        num_blocks=1000,
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        num_layers=32,
    )
    kv_cache = EngineKVCache(engine_context, desc)

    # Get buffers for attention kernel
    k_buffer, v_buffer = kv_cache.get_buffers(layer_idx=0)

    # Validate block table at step boundary
    kv_cache.validate_block_table(block_table)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from vllm.logger import init_logger
from .descriptors import KVCacheDescriptor

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import (
        MTLResourceStorageModeShared,
        MTLResourceStorageModePrivate,
        NSMakeRange,
    )
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    MTLResourceStorageModeShared = None
    MTLResourceStorageModePrivate = None
    NSMakeRange = None


@dataclass
class KVCacheStats:
    """Statistics for KV cache usage."""
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    bytes_per_layer: int
    total_bytes: int
    total_mb: float
    allocated_blocks: int
    free_blocks: int
    utilization: float


class EngineKVCache:
    """Engine-owned KV cache with MTLBuffer storage.

    This class manages the KV cache as MTLBuffers owned by the engine.
    It provides:
    - Direct MTLBuffer access for Metal kernels
    - Block table validation at step boundary
    - Strides for kernel navigation
    - Memory usage statistics

    The cache is the single source of truth - vLLM's KV tensors are
    lightweight wrappers that reference this data.

    Attributes:
        context: MetalEngineContext for device management
        descriptor: KVCacheDescriptor with configuration
        key_buffers: List of MTLBuffers for key cache (one per layer)
        value_buffers: List of MTLBuffers for value cache (one per layer)
    """

    def __init__(
        self,
        engine_context: Any,  # MetalEngineContext
        descriptor: KVCacheDescriptor,
        use_private_storage: Optional[bool] = None,
    ):
        """Initialize engine KV cache.

        Args:
            engine_context: MetalEngineContext for buffer allocation
            descriptor: KVCacheDescriptor with cache configuration
            use_private_storage: If True, use MTLStorageModePrivate for KV buffers.
                If False, use MTLStorageModeShared.
                If None (default), auto-detect based on engine prefill mode:
                - Private when engine prefill is enabled (no CPU access needed)
                - Shared when engine prefill is disabled (sync_from_torch_cache needed)
        """
        from .config import is_engine_prefill_enabled

        self._context = engine_context
        self._desc = descriptor

        # Determine storage mode
        if use_private_storage is None:
            # Auto-detect: use Private when engine prefill is enabled (default),
            # because sync_from_torch_cache is not needed
            self._use_private_storage = is_engine_prefill_enabled()
        else:
            self._use_private_storage = use_private_storage

        # Validate descriptor
        descriptor.__post_init__()

        # Calculate sizes
        self._elements_per_block = (
            descriptor.num_kv_heads *
            descriptor.block_size *
            descriptor.head_size
        )
        self._dtype_size = 2 if descriptor.dtype.itemsize == 2 else 4
        self._bytes_per_block = self._elements_per_block * self._dtype_size
        self._bytes_per_layer = self._bytes_per_block * descriptor.num_blocks

        # Compute strides for Metal kernel
        self._strides = {
            'block': descriptor.num_kv_heads * descriptor.block_size * descriptor.head_size,
            'head': descriptor.block_size * descriptor.head_size,
            'token': descriptor.head_size,
            'element': 1,
        }

        # Allocate buffers
        self._key_buffers: List[Any] = []
        self._value_buffers: List[Any] = []
        self._allocate_buffers()

        # Track block allocation (simple tracking, not full allocator)
        self._allocated_blocks = 0

        storage_mode_str = "private" if self._use_private_storage else "shared"
        logger.info(
            f"EngineKVCache initialized: {descriptor.num_layers} layers, "
            f"{descriptor.num_blocks} blocks, {descriptor.total_cache_mb:.1f} MB, "
            f"storage mode: {storage_mode_str}"
        )

    def _allocate_buffers(self) -> None:
        """Allocate MTLBuffers for all layers.

        Uses MTLStorageModePrivate when engine prefill is enabled (GPU-only access),
        or MTLStorageModeShared when CPU access is needed for sync_from_torch_cache.
        """
        device = self._context.device

        # Select storage mode based on configuration
        # METAL_PLAN.md: "Prefer MTLStorageModePrivate for KV cache + intermediates;
        # use MTLStorageModeShared only for CPU inputs/outputs."
        if self._use_private_storage:
            storage_mode = MTLResourceStorageModePrivate
        else:
            storage_mode = MTLResourceStorageModeShared

        for layer_idx in range(self._desc.num_layers):
            # Allocate key buffer
            key_buffer = device.newBufferWithLength_options_(
                self._bytes_per_layer,
                storage_mode
            )
            if key_buffer is None:
                raise RuntimeError(f"Failed to allocate key buffer for layer {layer_idx}")

            # Allocate value buffer
            value_buffer = device.newBufferWithLength_options_(
                self._bytes_per_layer,
                storage_mode
            )
            if value_buffer is None:
                raise RuntimeError(f"Failed to allocate value buffer for layer {layer_idx}")

            # Zero-initialize
            self._zero_buffer(key_buffer)
            self._zero_buffer(value_buffer)

            self._key_buffers.append(key_buffer)
            self._value_buffers.append(value_buffer)

        logger.debug(f"Allocated {len(self._key_buffers)} layer KV buffers")

    def _zero_buffer(self, buffer: Any) -> None:
        """Zero-initialize an MTLBuffer using GPU-side fillBuffer.

        This uses MTLBlitCommandEncoder.fillBuffer which is efficient GPU-side
        zeroing, avoiding the need to allocate huge CPU buffers just for
        initialization. For a multi-GB KV cache, using bytes() would allocate
        the same amount of CPU memory which is wasteful.
        """
        # Create a one-shot command buffer for initialization
        cmd_queue = self._context.command_queue
        cmd_buffer = cmd_queue.commandBuffer()
        if cmd_buffer is None:
            raise RuntimeError("Failed to create command buffer for buffer zeroing")

        # Create blit encoder
        blit_encoder = cmd_buffer.blitCommandEncoder()
        if blit_encoder is None:
            raise RuntimeError("Failed to create blit encoder for buffer zeroing")

        # Fill with zeros using GPU-side operation
        fill_range = NSMakeRange(0, buffer.length())
        blit_encoder.fillBuffer_range_value_(buffer, fill_range, 0)

        # End encoding and commit
        blit_encoder.endEncoding()
        cmd_buffer.commit()

        # Wait for completion - OK here since this is initialization, not hot path
        cmd_buffer.waitUntilCompleted()

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        return self._desc.num_layers

    @property
    def num_blocks(self) -> int:
        """Number of blocks per layer."""
        return self._desc.num_blocks

    @property
    def block_size(self) -> int:
        """Tokens per block."""
        return self._desc.block_size

    @property
    def num_kv_heads(self) -> int:
        """Number of KV heads."""
        return self._desc.num_kv_heads

    @property
    def head_size(self) -> int:
        """Head dimension."""
        return self._desc.head_size

    @property
    def strides(self) -> Dict[str, int]:
        """Get strides for Metal kernel navigation."""
        return self._strides

    @property
    def uses_private_storage(self) -> bool:
        """Check if KV cache uses MTLStorageModePrivate.

        When True, buffers are GPU-only and require explicit blit copies
        for any CPU access. This is the preferred mode per METAL_PLAN.md.
        """
        return self._use_private_storage

    def get_buffers(self, layer_idx: int) -> Tuple[Any, Any]:
        """Get key and value MTLBuffers for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (key_buffer, value_buffer)
        """
        if layer_idx < 0 or layer_idx >= self._desc.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self._desc.num_layers})")
        return self._key_buffers[layer_idx], self._value_buffers[layer_idx]

    def get_key_buffer(self, layer_idx: int) -> Any:
        """Get key MTLBuffer for a layer."""
        return self._key_buffers[layer_idx]

    def get_value_buffer(self, layer_idx: int) -> Any:
        """Get value MTLBuffer for a layer."""
        return self._value_buffers[layer_idx]

    def validate_block_table(
        self,
        block_table: Union[np.ndarray, "torch.Tensor"],
        context: str = "",
    ) -> None:
        """Validate block table indices at step boundary.

        This should be called BEFORE entering the engine hot path to ensure
        all block indices are valid. Invalid indices would cause GPU errors
        or incorrect results.

        Args:
            block_table: Block table [num_seqs, max_blocks_per_seq]
            context: Context string for error messages

        Raises:
            ValueError: If any block index is out of range
            RuntimeError: If block_table is on MPS device in strict mode
        """
        import torch
        from .strict_mode import is_strict_mode

        if isinstance(block_table, torch.Tensor):
            if block_table.device.type == 'cpu':
                block_table = block_table.numpy()
            elif block_table.device.type == 'mps':
                # In strict mode, fail-fast instead of silently converting
                if is_strict_mode():
                    raise RuntimeError(
                        f"MPS tensor passed to validate_block_table() in strict mode. "
                        f"Device: {block_table.device}. "
                        f"Block table must be on CPU BEFORE reaching engine boundary. "
                        f"Convert in vLLM code, not in engine code."
                    )
                # Non-strict: convert with warning
                logger.warning(
                    f"Converting block_table from {block_table.device} to CPU. "
                    f"This should be done in vLLM code."
                )
                block_table = block_table.cpu().numpy()
            else:
                # Unknown device type
                raise ValueError(f"Unsupported device type: {block_table.device.type}")

        # Check for out-of-range indices
        max_idx = block_table.max()
        min_idx = block_table[block_table >= 0].min() if (block_table >= 0).any() else 0

        if max_idx >= self._desc.num_blocks:
            raise ValueError(
                f"Block table contains index {max_idx} >= num_blocks {self._desc.num_blocks}"
                f"{' (' + context + ')' if context else ''}"
            )

        # -1 is valid (unused slot), but other negative values are not
        invalid_mask = (block_table < -1)
        if invalid_mask.any():
            invalid_idx = block_table[invalid_mask].min()
            raise ValueError(
                f"Block table contains invalid index {invalid_idx}"
                f"{' (' + context + ')' if context else ''}"
            )

    def validate_slot_mapping(
        self,
        slot_mapping: "Union[np.ndarray, torch.Tensor]",
        context: str = "",
    ) -> None:
        """Validate slot_mapping indices are within KV cache capacity.

        This should be called before prefill KV write to ensure that
        all slot indices are valid. Invalid indices would cause GPU errors
        or incorrect results (OOB write to KV cache).

        Args:
            slot_mapping: Slot mapping [num_tokens]
            context: Context string for error messages

        Raises:
            ValueError: If any slot index is out of range
            RuntimeError: If slot_mapping is on MPS device in strict mode
        """
        import torch
        from .strict_mode import is_strict_mode

        if isinstance(slot_mapping, torch.Tensor):
            if slot_mapping.device.type == 'cpu':
                slot_mapping = slot_mapping.numpy()
            elif slot_mapping.device.type == 'mps':
                if is_strict_mode():
                    raise RuntimeError(
                        f"MPS tensor passed to validate_slot_mapping() in strict mode. "
                        f"Device: {slot_mapping.device}. "
                        f"Slot mapping must be on CPU BEFORE reaching engine boundary."
                    )
                slot_mapping = slot_mapping.cpu().numpy()
            else:
                raise ValueError(f"Unsupported device type: {slot_mapping.device.type}")

        # Total capacity is num_blocks * block_size slots
        max_valid_slot = self._desc.num_blocks * self._desc.block_size - 1

        # Check for out-of-range indices (ignore -1 which means padding/invalid)
        valid_mask = slot_mapping >= 0
        if valid_mask.any():
            max_slot = slot_mapping[valid_mask].max()
            if max_slot > max_valid_slot:
                raise ValueError(
                    f"Slot mapping contains index {max_slot} > max valid slot {max_valid_slot} "
                    f"(num_blocks={self._desc.num_blocks}, block_size={self._desc.block_size})"
                    f"{' (' + context + ')' if context else ''}"
                )

        # Check for invalid negative values (only -1 is allowed)
        invalid_mask = (slot_mapping < -1)
        if invalid_mask.any():
            invalid_idx = slot_mapping[invalid_mask].min()
            raise ValueError(
                f"Slot mapping contains invalid index {invalid_idx}"
                f"{' (' + context + ')' if context else ''}"
            )

    def get_block_offset(self, block_id: int) -> int:
        """Get byte offset for a block.

        Args:
            block_id: Logical block ID

        Returns:
            Byte offset in the buffer
        """
        return block_id * self._bytes_per_block

    def get_slot_offset(self, slot_idx: int) -> Tuple[int, int]:
        """Convert absolute slot index to (block_id, token_offset).

        Args:
            slot_idx: Absolute slot index

        Returns:
            Tuple of (block_id, token_offset)
        """
        block_id = slot_idx // self._desc.block_size
        token_offset = slot_idx % self._desc.block_size
        return block_id, token_offset

    def get_element_offset(
        self,
        block_id: int,
        head_idx: int,
        token_offset: int,
    ) -> int:
        """Get element offset for a specific (block, head, token) position.

        Args:
            block_id: Block index
            head_idx: KV head index
            token_offset: Token offset within block

        Returns:
            Element offset (multiply by dtype_size for byte offset)
        """
        return (
            block_id * self._strides['block'] +
            head_idx * self._strides['head'] +
            token_offset * self._strides['token']
        )

    def get_stats(self) -> KVCacheStats:
        """Get cache statistics.

        Returns:
            KVCacheStats with current usage information
        """
        total_bytes = self._bytes_per_layer * self._desc.num_layers * 2  # K + V
        return KVCacheStats(
            num_layers=self._desc.num_layers,
            num_blocks=self._desc.num_blocks,
            block_size=self._desc.block_size,
            num_kv_heads=self._desc.num_kv_heads,
            head_size=self._desc.head_size,
            bytes_per_layer=self._bytes_per_layer,
            total_bytes=total_bytes,
            total_mb=total_bytes / (1024 * 1024),
            allocated_blocks=self._allocated_blocks,
            free_blocks=self._desc.num_blocks - self._allocated_blocks,
            utilization=self._allocated_blocks / self._desc.num_blocks if self._desc.num_blocks > 0 else 0,
        )

    def reset(self) -> None:
        """Reset cache, zeroing all buffers.

        This should be called at scheduler reset, not during inference.
        """
        for layer_idx in range(self._desc.num_layers):
            self._zero_buffer(self._key_buffers[layer_idx])
            self._zero_buffer(self._value_buffers[layer_idx])
        self._allocated_blocks = 0
        logger.debug("EngineKVCache reset")

    def to_params_dict(self) -> Dict[str, Any]:
        """Get parameters dict for Metal kernel.

        Returns a dict suitable for creating kernel parameter buffer.

        Returns:
            Dict with cache parameters
        """
        return {
            'num_blocks': self._desc.num_blocks,
            'block_size': self._desc.block_size,
            'num_kv_heads': self._desc.num_kv_heads,
            'head_size': self._desc.head_size,
            'stride_block': self._strides['block'],
            'stride_head': self._strides['head'],
            'stride_token': self._strides['token'],
        }

    def sync_from_torch_cache(
        self,
        torch_caches: List["torch.Tensor"],
        block_table: "torch.Tensor",
        seq_lens: "torch.Tensor",
    ) -> None:
        """Sync KV data from torch cache to engine MTLBuffers.

        This is called after prefill to ensure the engine's KV cache contains
        the data written by the PyTorch attention path. This is necessary
        because prefill uses PyTorch (torch cache) but decode uses engine
        (MTLBuffer cache).

        ⚠️ TEMPORARY STOP-GAP - NOT STRICT-MODE COMPLIANT ⚠️

        This method violates two METAL_PLAN invariants:
        1. KV cache single-source-of-truth: We have BOTH torch KV + engine KV (duplication)
        2. No torch.mps.synchronize() in engine path: We call it here before reading

        This is acceptable as a TRANSITIONAL solution until:
        1. Prefill is moved to the engine (removing torch KV entirely), OR
        2. Layouts are unified so torch and MTLBuffer can share memory (zero-copy)

        DO NOT claim strict-mode compliance while using this sync bridge.
        The engine path is only strict-mode compliant for decode-only batches
        that don't require KV sync (e.g., continuing sequences).

        The sync copies ONLY active blocks (based on block_table), avoiding
        full cache materialization. Memory efficiency is critical here.

        Args:
            torch_caches: List of torch KV cache tensors, one per layer.
                         Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            block_table: Block table [num_seqs, max_blocks_per_seq]
            seq_lens: Sequence lengths [num_seqs]

        Layout conversion:
            torch:  [2, num_blocks, block_size, num_kv_heads, head_size]
            engine: [num_blocks, num_kv_heads, block_size, head_size]
        """
        import torch
        import ctypes
        from .strict_mode import is_strict_mode

        # This sync bridge is intentionally NOT strict-mode compliant.
        # Strict mode must not silently introduce PyTorch-MPS sync points.
        if is_strict_mode():
            raise RuntimeError(
                "sync_from_torch_cache() is not allowed in strict mode "
                "(VLLM_METAL_STRICT_NO_MPS=1). Enable engine prefill "
                "(VLLM_APPLE_ENGINE_PREFILL=1) or disable strict mode."
            )

        # Check if KV cache supports sync (requires Shared storage mode)
        if self._use_private_storage:
            raise RuntimeError(
                "sync_from_torch_cache() is not supported with MTLStorageModePrivate. "
                "Private storage mode is used when engine prefill is enabled. "
                "Either disable engine prefill (VLLM_APPLE_ENGINE_PREFILL=0) or "
                "ensure engine prefill is being used for the prefill step."
            )

        if len(torch_caches) != self._desc.num_layers:
            raise ValueError(
                f"torch_caches has {len(torch_caches)} layers, "
                f"expected {self._desc.num_layers}"
            )

        # SYNC BARRIER: Ensure torch cache writes are complete before reading.
        # PyTorch prefill runs on MPS, we need to ensure GPU work is done.
        # This is the "producer completed" guarantee required by the plan.
        if torch_caches[0].device.type == 'mps':
            import torch.mps
            torch.mps.synchronize()
            logger.debug("MPS sync completed before KV cache read")

        # Get active block IDs from block_table (must be on CPU for iteration)
        block_table_cpu = block_table.cpu() if block_table.device.type != 'cpu' else block_table
        seq_lens_cpu = seq_lens.cpu() if seq_lens.device.type != 'cpu' else seq_lens

        # Collect unique active block IDs
        active_blocks = set()
        for seq_idx in range(block_table_cpu.shape[0]):
            seq_len = seq_lens_cpu[seq_idx].item()
            num_blocks_needed = (seq_len + self._desc.block_size - 1) // self._desc.block_size
            for block_idx in range(min(num_blocks_needed, block_table_cpu.shape[1])):
                block_id = block_table_cpu[seq_idx, block_idx].item()
                if block_id >= 0:  # Valid block
                    active_blocks.add(block_id)

        if not active_blocks:
            logger.debug("No active blocks to sync")
            return

        logger.debug(f"Syncing {len(active_blocks)} active blocks from torch to engine KV cache")

        # Get numpy views of engine MTLBuffers for safe writes
        # Using PyObjC's as_buffer() for safe memory access
        k_engine_views = []
        v_engine_views = []
        for layer_idx in range(self._desc.num_layers):
            k_buffer = self._key_buffers[layer_idx]
            v_buffer = self._value_buffers[layer_idx]

            # Create numpy array viewing MTLBuffer memory
            # Shape: [num_blocks, num_kv_heads, block_size, head_size]
            # PyObjC 12+: buffer.contents() returns objc.varlist with as_buffer() method
            # This gives us a memoryview that numpy can directly use
            try:
                k_memview = k_buffer.contents().as_buffer(self._bytes_per_layer)
                v_memview = v_buffer.contents().as_buffer(self._bytes_per_layer)
            except (AttributeError, TypeError) as e:
                logger.error(f"Failed to get MTLBuffer contents: {e}")
                raise RuntimeError(
                    f"Cannot access MTLBuffer contents. Ensure storageModeShared is used "
                    f"and PyObjC 12+ is installed. Error: {e}"
                )

            # Create numpy arrays from memoryview (writable, zero-copy view)
            dtype = np.float16 if self._dtype_size == 2 else np.float32
            k_view = np.frombuffer(k_memview, dtype=dtype).reshape(
                self._desc.num_blocks,
                self._desc.num_kv_heads,
                self._desc.block_size,
                self._desc.head_size,
            )
            v_view = np.frombuffer(v_memview, dtype=dtype).reshape(
                self._desc.num_blocks,
                self._desc.num_kv_heads,
                self._desc.block_size,
                self._desc.head_size,
            )
            k_engine_views.append(k_view)
            v_engine_views.append(v_view)

        # Sync each layer - copy only active blocks
        # CRITICAL: Do NOT call contiguous() on entire cache - only per-block
        for layer_idx in range(self._desc.num_layers):
            torch_kv = torch_caches[layer_idx]  # [2, num_blocks, block_size, num_kv_heads, head_size]

            # torch_kv[0] is K, torch_kv[1] is V
            k_torch = torch_kv[0]  # [num_blocks, block_size, num_kv_heads, head_size]
            v_torch = torch_kv[1]

            k_engine = k_engine_views[layer_idx]
            v_engine = v_engine_views[layer_idx]

            for block_id in active_blocks:
                if block_id >= self._desc.num_blocks:
                    continue  # Safety check

                # Extract single block and transpose to engine layout
                # torch: [block_size, num_kv_heads, head_size]
                # engine: [num_kv_heads, block_size, head_size]
                # Only this single block is made contiguous, not the entire cache
                k_block_torch = k_torch[block_id].permute(1, 0, 2).contiguous()
                v_block_torch = v_torch[block_id].permute(1, 0, 2).contiguous()

                # Move to CPU and convert to numpy
                k_block_np = k_block_torch.cpu().numpy()
                v_block_np = v_block_torch.cpu().numpy()

                # Copy to engine buffer using numpy indexing (safe, no pointer arithmetic)
                k_engine[block_id] = k_block_np
                v_engine[block_id] = v_block_np

        logger.debug(f"KV cache sync complete for {len(active_blocks)} blocks")

    def supports_torch_sync(self) -> bool:
        """Check if torch cache sync is supported.

        Returns True if the MTLBuffers use storageModeShared which allows
        CPU access to the buffer memory for copying.

        When using MTLStorageModePrivate (the default with engine prefill),
        sync is not supported because buffers cannot be directly accessed
        from CPU. In this case, engine prefill should be used instead.

        Returns:
            True if sync is supported (Shared storage), False otherwise (Private storage).
        """
        # Only storageModeShared buffers support direct CPU access
        # Private storage mode requires explicit blit copies
        return not self._use_private_storage


class EngineKVCachePool:
    """Pool of KV caches for multi-model or multi-instance scenarios.

    In typical single-model usage, only one EngineKVCache is needed.
    This pool supports advanced scenarios with multiple models.
    """

    def __init__(self, engine_context: Any):
        """Initialize cache pool.

        Args:
            engine_context: MetalEngineContext for buffer allocation
        """
        self._context = engine_context
        self._caches: Dict[str, EngineKVCache] = {}

    def create_cache(
        self,
        name: str,
        descriptor: KVCacheDescriptor,
    ) -> EngineKVCache:
        """Create a named KV cache.

        Args:
            name: Cache identifier
            descriptor: Cache configuration

        Returns:
            EngineKVCache instance
        """
        if name in self._caches:
            raise ValueError(f"Cache '{name}' already exists")

        cache = EngineKVCache(self._context, descriptor)
        self._caches[name] = cache
        return cache

    def get_cache(self, name: str) -> Optional[EngineKVCache]:
        """Get a cache by name.

        Args:
            name: Cache identifier

        Returns:
            EngineKVCache or None if not found
        """
        return self._caches.get(name)

    def remove_cache(self, name: str) -> None:
        """Remove a cache.

        Args:
            name: Cache identifier
        """
        if name in self._caches:
            del self._caches[name]

    def reset_all(self) -> None:
        """Reset all caches in the pool."""
        for cache in self._caches.values():
            cache.reset()

    def get_total_memory_bytes(self) -> int:
        """Get total memory used by all caches."""
        return sum(c.get_stats().total_bytes for c in self._caches.values())
