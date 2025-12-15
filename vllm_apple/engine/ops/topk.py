# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-K Logits Selection for vLLM-Apple Metal Engine v2.0.

This module provides a custom Metal kernel for selecting top-k logits,
which significantly reduces the amount of data that needs to be read
back from GPU to CPU during sampling.

Benefit:
- Full logits readback: vocab_size * 2 bytes per token (e.g., 256KB for 128K vocab)
- Top-K readback: k * 6 bytes per token (e.g., 300 bytes for k=50)
- Reduction: ~800x less data transfer

Usage:
    from vllm_apple.engine.ops.topk import EngineTopK

    topk = EngineTopK(context, k=50)
    topk.encode(step_ctx, logits_buffer, indices_out, values_out, num_tokens, vocab_size)
"""

import os
import struct
from typing import Any, Dict, Optional, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)

# Environment variable for top-k configuration
TOPK_K_ENV = "VLLM_METAL_TOPK_K"  # Default k value

# Try to import Metal
try:
    from Metal import MTLSize
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


class EngineTopK:
    """Custom Metal Top-K selection operation.

    Selects the top-k logits for each token, outputting their indices
    and values. This is much more efficient than reading back full
    logits and doing top-k on CPU.

    Algorithm:
    - Each threadgroup handles one token (one row of logits)
    - Uses parallel reduction with local top-k per thread
    - Merges results using threadgroup memory

    Attributes:
        context: MetalEngineContext
        k: Number of top elements to select
    """

    # Simplified Top-K kernel using two-stage reduction
    # Each threadgroup handles one row (one token's logits)
    KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct TopKParams {
    uint num_tokens;
    uint vocab_size;
    uint k;
};

// Top-K selection with fixed k=50 max
// Each threadgroup processes one token's logits
// Uses 64 threads, each finding local top-k, then merge
// Threadgroup memory: 64 * 50 * 4 * 2 = 25,600 bytes (under 32KB limit)
kernel void topk_select(
    device const half* logits [[buffer(0)]],
    device int* topk_indices [[buffer(1)]],
    device half* topk_values [[buffer(2)]],
    constant TopKParams& params [[buffer(3)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Fixed maximum k per thread (for threadgroup memory sizing)
    const uint LOCAL_K = 50;
    const uint NUM_THREADS = 64;

    // Local storage for top-k
    float local_vals[LOCAL_K];
    int local_idxs[LOCAL_K];

    uint k = min(params.k, LOCAL_K);

    // Initialize with -inf
    for (uint i = 0; i < k; i++) {
        local_vals[i] = -INFINITY;
        local_idxs[i] = -1;
    }

    // Each thread processes vocab_size / 64 elements
    device const half* row = logits + token_idx * params.vocab_size;

    for (uint i = tid; i < params.vocab_size; i += NUM_THREADS) {
        float val = float(row[i]);

        // Insert if better than worst in local top-k
        if (val > local_vals[k - 1]) {
            // Find insertion position
            uint pos = k;
            for (uint j = 0; j < k; j++) {
                if (val > local_vals[j]) {
                    pos = j;
                    break;
                }
            }

            // Shift and insert
            if (pos < k) {
                for (uint j = k - 1; j > pos; j--) {
                    local_vals[j] = local_vals[j - 1];
                    local_idxs[j] = local_idxs[j - 1];
                }
                local_vals[pos] = val;
                local_idxs[pos] = int(i);
            }
        }
    }

    // Merge using threadgroup memory
    // 64 threads x 50 values = 3200 floats = 12,800 bytes
    // 64 threads x 50 indices = 3200 ints = 12,800 bytes
    // Total: 25,600 bytes (within 32KB limit)
    threadgroup float shared_vals[NUM_THREADS * LOCAL_K];
    threadgroup int shared_idxs[NUM_THREADS * LOCAL_K];

    // Copy to shared memory
    for (uint i = 0; i < k; i++) {
        shared_vals[tid * LOCAL_K + i] = local_vals[i];
        shared_idxs[tid * LOCAL_K + i] = local_idxs[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
    for (uint stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            uint other = tid + stride;

            // Merge other's top-k into this thread's top-k
            for (uint i = 0; i < k; i++) {
                float other_val = shared_vals[other * LOCAL_K + i];
                int other_idx = shared_idxs[other * LOCAL_K + i];

                if (other_val > shared_vals[tid * LOCAL_K + k - 1]) {
                    // Find position and insert
                    uint pos = k;
                    for (uint j = 0; j < k; j++) {
                        if (other_val > shared_vals[tid * LOCAL_K + j]) {
                            pos = j;
                            break;
                        }
                    }

                    if (pos < k) {
                        for (uint j = k - 1; j > pos; j--) {
                            shared_vals[tid * LOCAL_K + j] = shared_vals[tid * LOCAL_K + j - 1];
                            shared_idxs[tid * LOCAL_K + j] = shared_idxs[tid * LOCAL_K + j - 1];
                        }
                        shared_vals[tid * LOCAL_K + pos] = other_val;
                        shared_idxs[tid * LOCAL_K + pos] = other_idx;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes results
    if (tid == 0) {
        device int* out_idxs = topk_indices + token_idx * params.k;
        device half* out_vals = topk_values + token_idx * params.k;

        for (uint i = 0; i < params.k; i++) {
            out_idxs[i] = shared_idxs[i];
            out_vals[i] = half(shared_vals[i]);
        }
    }
}

// Simplified small-k kernel
kernel void topk_select_small_k(
    device const half* logits [[buffer(0)]],
    device int* topk_indices [[buffer(1)]],
    device half* topk_values [[buffer(2)]],
    constant TopKParams& params [[buffer(3)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Fixed k=8 for small-k version
    const uint K = 8;
    float vals[K];
    int idxs[K];

    for (uint i = 0; i < K; i++) {
        vals[i] = -INFINITY;
        idxs[i] = -1;
    }

    device const half* row = logits + token_idx * params.vocab_size;

    for (uint i = tid; i < params.vocab_size; i += 128) {
        float val = float(row[i]);

        if (val > vals[K - 1]) {
            // Simple insertion sort
            uint pos = K;
            for (uint j = 0; j < K; j++) {
                if (val > vals[j]) {
                    pos = j;
                    break;
                }
            }
            if (pos < K) {
                for (uint j = K - 1; j > pos; j--) {
                    vals[j] = vals[j - 1];
                    idxs[j] = idxs[j - 1];
                }
                vals[pos] = val;
                idxs[pos] = int(i);
            }
        }
    }

    // Merge using shared memory
    threadgroup float shared_vals[128 * 8];
    threadgroup int shared_idxs[128 * 8];

    for (uint i = 0; i < K; i++) {
        shared_vals[tid * K + i] = vals[i];
        shared_idxs[tid * K + i] = idxs[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            uint other = tid + stride;

            for (uint i = 0; i < K; i++) {
                float other_val = shared_vals[other * K + i];
                int other_idx = shared_idxs[other * K + i];

                if (other_val > shared_vals[tid * K + K - 1]) {
                    uint pos = K;
                    for (uint j = 0; j < K; j++) {
                        if (other_val > shared_vals[tid * K + j]) {
                            pos = j;
                            break;
                        }
                    }
                    if (pos < K) {
                        for (uint j = K - 1; j > pos; j--) {
                            shared_vals[tid * K + j] = shared_vals[tid * K + j - 1];
                            shared_idxs[tid * K + j] = shared_idxs[tid * K + j - 1];
                        }
                        shared_vals[tid * K + pos] = other_val;
                        shared_idxs[tid * K + pos] = other_idx;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes results
    if (tid == 0) {
        device int* out_idxs = topk_indices + token_idx * params.k;
        device half* out_vals = topk_values + token_idx * params.k;

        uint out_k = min(params.k, K);
        for (uint i = 0; i < out_k; i++) {
            out_idxs[i] = shared_idxs[i];
            out_vals[i] = half(shared_vals[i]);
        }
    }
}
"""

    def __init__(self, context: Any, k: int = 50):
        """Initialize Top-K operation.

        Args:
            context: MetalEngineContext
            k: Number of top elements to select (default: 50)
        """
        self._context = context
        self._device = context.device
        self._k = k

        # Check for environment override
        env_k = os.environ.get(TOPK_K_ENV)
        if env_k:
            try:
                self._k = int(env_k)
                logger.info(f"Top-K k overridden by env: {self._k}")
            except ValueError:
                pass

        # Pipelines
        self._pipeline_standard = None
        self._pipeline_small_k = None

        # Compile kernels
        self._compile_kernels()

        logger.info(f"EngineTopK initialized with k={self._k}")

    def _compile_kernels(self) -> None:
        """Compile Top-K Metal kernels."""
        self._context.compile_library("topk", source_code=self.KERNEL_SOURCE)
        self._pipeline_standard = self._context.get_pipeline("topk", "topk_select")
        self._pipeline_small_k = self._context.get_pipeline("topk", "topk_select_small_k")

        logger.info("Compiled Top-K selection kernels")

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        logits: Any,  # MTLBuffer or EngineTensor
        indices_out: Any,  # MTLBuffer for output indices
        values_out: Any,  # MTLBuffer for output values
        num_tokens: int,
        vocab_size: int,
        k: Optional[int] = None,
    ) -> None:
        """Encode Top-K selection to command buffer.

        Args:
            step_ctx: EngineStepContext with encoder
            logits: Input logits [num_tokens, vocab_size]
            indices_out: Output indices [num_tokens, k]
            values_out: Output values [num_tokens, k]
            num_tokens: Number of tokens
            vocab_size: Vocabulary size
            k: Override k value (default: use instance k)
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        k = k or self._k

        # Extract buffers
        if hasattr(logits, 'buffer'):
            logits_buf = logits.buffer
            logits_offset = logits.offset
        else:
            logits_buf = logits
            logits_offset = 0

        if hasattr(indices_out, 'buffer'):
            indices_buf = indices_out.buffer
            indices_offset = indices_out.offset
        else:
            indices_buf = indices_out
            indices_offset = 0

        if hasattr(values_out, 'buffer'):
            values_buf = values_out.buffer
            values_offset = values_out.offset
        else:
            values_buf = values_out
            values_offset = 0

        # Select kernel
        if k <= 8:
            pipeline = self._pipeline_small_k
            threads_per_group = 128  # small_k kernel uses 128 threads
        else:
            pipeline = self._pipeline_standard
            threads_per_group = 64  # standard kernel uses 64 threads (reduced for threadgroup memory)

        # Get encoder
        encoder = step_ctx.get_compute_encoder()

        # Set pipeline
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(logits_buf, logits_offset, 0)
        encoder.setBuffer_offset_atIndex_(indices_buf, indices_offset, 1)
        encoder.setBuffer_offset_atIndex_(values_buf, values_offset, 2)

        # Set parameters
        params_data = struct.pack("III", num_tokens, vocab_size, k)
        encoder.setBytes_length_atIndex_(params_data, len(params_data), 3)

        # Dispatch: one threadgroup per token
        thread_groups = MTLSize(num_tokens, 1, 1)
        threads_per_threadgroup = MTLSize(threads_per_group, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, threads_per_threadgroup
        )

    @property
    def k(self) -> int:
        """Get the k value."""
        return self._k

    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            "k": self._k,
        }
