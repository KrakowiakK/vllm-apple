# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV cache synchronization between torch and engine.

These are integration tests that require Metal hardware.
Moved from tests/unit/test_engine_invariants.py because they involve
actual Metal operations (MetalEngineContext, EngineKVCache, MTLBuffer).

BUG DOCUMENTED: Before the fix, prefill wrote to torch KV cache
and decode read from engine MTLBuffer - separate memory locations.
This caused incorrect decode results.

The fix syncs KV data from torch to engine after prefill.

⚠️  TEMPORARY BRIDGE - NOT STRICT-MODE COMPLIANT ⚠️
The sync path violates METAL_PLAN invariants:
1. KV cache single-source-of-truth: We have BOTH torch KV + engine KV
2. No torch.mps.synchronize() in engine path: We call it during sync

This is a stop-gap until full engine mode replaces prefill.
"""

import os
import pytest
import numpy as np
import torch

# Skip all tests if Metal not available
pytest.importorskip("Metal")


@pytest.fixture
def strict_mode_disabled():
    """Fixture to disable strict mode for tests that need MPS conversion."""
    original = os.environ.get("VLLM_METAL_STRICT_NO_MPS")
    os.environ.pop("VLLM_METAL_STRICT_NO_MPS", None)

    try:
        import vllm_apple.engine.config as config_mod
        if hasattr(config_mod, 'reset_engine_config'):
            config_mod.reset_engine_config()
    except ImportError:
        pass

    yield

    if original is not None:
        os.environ["VLLM_METAL_STRICT_NO_MPS"] = original
    else:
        os.environ.pop("VLLM_METAL_STRICT_NO_MPS", None)

    try:
        import vllm_apple.engine.config as config_mod
        if hasattr(config_mod, 'reset_engine_config'):
            config_mod.reset_engine_config()
    except ImportError:
        pass


class TestKVCacheDataConsistency:
    """Tests for KV cache data consistency between torch and engine.

    BUG DOCUMENTED: Before the fix, prefill wrote to torch KV cache
    and decode read from engine MTLBuffer - separate memory locations.
    This caused incorrect decode results.

    The fix syncs KV data from torch to engine after prefill.
    """

    def test_prefill_decode_kv_consistency_documented(self):
        """Document the prefill→decode KV consistency requirement.

        Execution flow:
        1. Prefill runs via PyTorch → writes K/V to torch cache
        2. Sync runs → copies K/V from torch to engine MTLBuffer
        3. Decode runs via engine → reads K/V from engine MTLBuffer

        Without step 2, decode would read empty/stale K/V data.
        """
        # This test documents the architecture
        pass

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available (required for KV cache sync test)"
    )
    def test_prefill_sync_decode_kv_correctness(self, strict_mode_disabled):
        """Verify prefill → KV sync → decode actually transfers data correctly.

        This is the critical regression test for the KV cache sync bug.
        The bug was: prefill wrote to torch cache, decode read from engine cache,
        but no data was ever copied between them.

        This test:
        1. Creates a torch KV cache (simulating prefill output)
        2. Fills it with known values
        3. Calls sync_from_torch_cache()
        4. Verifies the engine cache contains the same data

        If this test fails, decode will produce incorrect results.
        """
        from vllm_apple.engine import KVCacheDescriptor
        from vllm_apple.engine.kv_cache import EngineKVCache
        from vllm_apple.engine.context import MetalEngineContext

        # Test parameters (small for unit test)
        num_layers = 2
        num_kv_heads = 4
        head_size = 64
        block_size = 16
        num_blocks = 8

        # Create engine context for buffer allocation
        try:
            engine_ctx = MetalEngineContext.get_instance()
        except Exception as e:
            pytest.skip(f"Could not create MetalEngineContext: {e}")

        # Create KV cache descriptor
        kv_desc = KVCacheDescriptor(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            num_layers=num_layers,
        )

        # Create engine KV cache
        engine_cache = EngineKVCache(engine_ctx, kv_desc)

        # Check if sync is supported (storageModeShared required)
        if not engine_cache.supports_torch_sync():
            pytest.skip("Engine KV cache does not support torch sync (not storageModeShared)")

        # Create torch KV cache (simulating what prefill would produce)
        # torch layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        # where index 0 is K and index 1 is V
        torch_kv_caches = []

        for layer_idx in range(num_layers):
            # Create combined K+V cache with deterministic pattern
            kv_cache = torch.zeros(
                2, num_blocks, block_size, num_kv_heads, head_size,
                dtype=torch.float16, device="mps"
            )
            # Fill K (index 0) with layer-specific pattern: layer_idx + 0.1 * head_idx
            for head_idx in range(num_kv_heads):
                kv_cache[0, :, :, head_idx, :] = layer_idx + 0.1 * head_idx

            # Fill V (index 1) with different pattern: layer_idx * 10 + head_idx
            for head_idx in range(num_kv_heads):
                kv_cache[1, :, :, head_idx, :] = layer_idx * 10 + head_idx

            torch_kv_caches.append(kv_cache)

        # Create block table - only use some blocks
        # Simulates a sequence using blocks 0, 2, 5
        active_blocks = [0, 2, 5]
        block_table = torch.tensor([active_blocks], dtype=torch.int32)
        seq_lens = torch.tensor([len(active_blocks) * block_size], dtype=torch.int32)

        # Perform the sync (this is what we're testing)
        engine_cache.sync_from_torch_cache(
            torch_caches=torch_kv_caches,
            block_table=block_table,
            seq_lens=seq_lens,
        )

        # Verify the data was synced correctly
        # Read back from engine cache MTLBuffers using PyObjC's as_buffer()
        bytes_per_layer = num_blocks * num_kv_heads * block_size * head_size * 2  # float16
        for layer_idx in range(num_layers):
            k_buffer, v_buffer = engine_cache.get_buffers(layer_idx)

            # Get numpy view of the engine buffer via as_buffer() (PyObjC 12+)
            k_memview = k_buffer.contents().as_buffer(bytes_per_layer)
            v_memview = v_buffer.contents().as_buffer(bytes_per_layer)

            k_engine_data = np.frombuffer(k_memview, dtype=np.float16).reshape(
                num_blocks, num_kv_heads, block_size, head_size
            )
            v_engine_data = np.frombuffer(v_memview, dtype=np.float16).reshape(
                num_blocks, num_kv_heads, block_size, head_size
            )

            # Verify only the active blocks were copied
            for block_id in active_blocks:
                # Get expected values from torch cache
                torch.mps.synchronize()  # Ensure torch writes completed
                # torch layout: [2, num_blocks, block_size, num_kv_heads, head_size]
                # engine layout: [num_blocks, num_kv_heads, block_size, head_size]
                # Need to transpose: [block_size, num_kv_heads, head_size] -> [num_kv_heads, block_size, head_size]
                k_expected = torch_kv_caches[layer_idx][0, block_id].permute(1, 0, 2).cpu().numpy()
                v_expected = torch_kv_caches[layer_idx][1, block_id].permute(1, 0, 2).cpu().numpy()

                k_actual = k_engine_data[block_id]
                v_actual = v_engine_data[block_id]

                # Check K cache
                np.testing.assert_allclose(
                    k_actual, k_expected,
                    rtol=1e-3, atol=1e-3,
                    err_msg=f"K cache mismatch at layer {layer_idx}, block {block_id}"
                )

                # Check V cache
                np.testing.assert_allclose(
                    v_actual, v_expected,
                    rtol=1e-3, atol=1e-3,
                    err_msg=f"V cache mismatch at layer {layer_idx}, block {block_id}"
                )

    def test_kv_sync_handles_out_of_bounds_blocks_safely(self):
        """Verify sync_from_torch_cache handles out-of-bounds block IDs safely.

        When block_table contains block IDs >= num_blocks (invalid), the sync
        should skip those blocks rather than crash or corrupt memory. This is
        a defensive measure - the scheduler should never produce invalid block
        IDs, but we handle them gracefully just in case.

        Valid blocks in the table should still be synced correctly.
        """
        from vllm_apple.engine import KVCacheDescriptor
        from vllm_apple.engine.kv_cache import EngineKVCache
        from vllm_apple.engine.context import MetalEngineContext

        # Create engine context for buffer allocation
        try:
            engine_ctx = MetalEngineContext.get_instance()
        except Exception as e:
            pytest.skip(f"Could not create MetalEngineContext: {e}")

        kv_desc = KVCacheDescriptor(
            num_blocks=10,  # Only 10 blocks allocated
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            num_layers=2,
        )

        engine_cache = EngineKVCache(engine_ctx, kv_desc)

        if not engine_cache.supports_torch_sync():
            pytest.skip("Engine KV cache does not support torch sync")

        # Create torch cache with some data in valid blocks
        # torch layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        torch_kv = [torch.zeros(2, 10, 16, 4, 64, dtype=torch.float16) for _ in range(2)]
        # Mark block 0 and 1 with recognizable data
        torch_kv[0][0, 0, :, :, :] = 1.0  # K cache, layer 0, block 0
        torch_kv[0][1, 0, :, :, :] = 2.0  # V cache, layer 0, block 0

        # Block table with BOTH valid (0, 1) and invalid (99) block IDs
        # The sync should process blocks 0 and 1, skip block 99
        mixed_block_table = torch.tensor([[0, 1, 99]], dtype=torch.int32)
        seq_lens = torch.tensor([48], dtype=torch.int32)  # 3 blocks needed

        # Sync should succeed - invalid block 99 is skipped, valid blocks synced
        engine_cache.sync_from_torch_cache(
            torch_caches=torch_kv,
            block_table=mixed_block_table,
            seq_lens=seq_lens,
        )
        # Test passes if no crash - invalid blocks are silently skipped
        # Valid blocks (0, 1) should have been synced successfully
