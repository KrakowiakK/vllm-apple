# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Metal GEMM for vLLM-Apple Metal Engine v2.0.

This module provides custom Metal GEMM kernels to replace MPSMatrixMultiplication.
The goal is to achieve MPS-level or better performance with full control over
kernel behavior and tuning.

Implementation Phases:
1. Naive kernel (correctness first)
2. Tiled kernel with threadgroup memory
3. simdgroup_matrix optimized kernel
4. Specialized variants (small-M, large-N)

All kernels are encode-only (no internal waits) to maintain v2.0 invariants.

Usage:
    from vllm_apple.engine.ops.gemm_metal import EngineGEMMMetal

    # Create op during initialization
    gemm = EngineGEMMMetal(context)

    # Encode GEMM (no wait) - C = A @ B^T
    gemm.encode(
        step_ctx=step_ctx,
        A=input_tensor,   # [M, K]
        B=weight_tensor,  # [N, K] (transposed storage)
        C=output_tensor,  # [M, N]
        M=num_tokens,
        K=hidden_size,
        N=output_size,
    )
"""

import os
import struct
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLSize
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

# Environment variables for GEMM configuration
GEMM_BACKEND_ENV = "VLLM_GEMM_BACKEND"  # auto, mps, metal
GEMM_DEBUG_ENV = "VLLM_METAL_GEMM_DEBUG"  # 0 or 1
GEMM_VERIFY_ENV = "VLLM_METAL_GEMM_VERIFY"  # 0 or 1


def get_gemm_backend() -> str:
    """Get GEMM backend preference from environment."""
    return os.environ.get(GEMM_BACKEND_ENV, "auto")


def is_gemm_debug() -> bool:
    """Check if GEMM debug logging is enabled."""
    return os.environ.get(GEMM_DEBUG_ENV, "0") == "1"


def is_gemm_verify() -> bool:
    """Check if GEMM verification against MPS is enabled."""
    return os.environ.get(GEMM_VERIFY_ENV, "0") == "1"


@dataclass
class GEMMMetalConfig:
    """Configuration for Metal GEMM operation."""
    M: int  # Rows of A, rows of C
    K: int  # Cols of A, cols of B (after transpose)
    N: int  # Rows of B (stored transposed), cols of C
    # Tile sizes for optimized kernels
    BM: int = 64  # Tile size in M dimension
    BN: int = 64  # Tile size in N dimension
    BK: int = 16  # Tile size in K dimension


class EngineGEMMMetal:
    """Custom Metal GEMM operation.

    Provides custom Metal GEMM kernels as alternative to MPSMatrixMultiplication.
    Implements multiple kernel variants for different matrix shapes:
    - Naive: Simple but correct baseline
    - Tiled: Optimized with threadgroup memory
    - simdgroup: Uses hardware matrix multiply-accumulate
    - Small-M: Optimized for decode (M=1-8)
    - Large-N: Optimized for LM head (N=128K+)

    All operations follow the encode-only pattern (no internal waits).

    Attributes:
        context: MetalEngineContext
    """

    # Naive GEMM kernel - one thread per output element
    # C[M,N] = A[M,K] @ B^T[N,K]
    # Weights stored as [N, K] (transposed from PyTorch [out, in] convention)
    KERNEL_SOURCE_NAIVE = """
#include <metal_stdlib>
using namespace metal;

// GEMM parameters
struct GEMMParams {
    uint M;      // Rows of A and C
    uint N;      // Rows of B (cols of C)
    uint K;      // Cols of A, Cols of B
    uint lda;    // Leading dimension of A (K for row-major)
    uint ldb;    // Leading dimension of B (K for row-major transposed)
    uint ldc;    // Leading dimension of C (N for row-major)
};

// Naive GEMM: C[M,N] = A[M,K] @ B^T[N,K]
// Each thread computes one element of C
// A is [M, K] row-major
// B is [N, K] row-major (already transposed, i.e., weights stored as [out, in])
// C is [M, N] row-major
kernel void gemm_naive_f16(
    device const half* A [[buffer(0)]],      // [M, K]
    device const half* B [[buffer(1)]],      // [N, K] (transposed)
    device half* C [[buffer(2)]],            // [M, N]
    constant GEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension

    if (row >= params.M || col >= params.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        // A[row, k] * B[col, k] (B is transposed)
        float a_val = float(A[row * params.lda + k]);
        float b_val = float(B[col * params.ldb + k]);
        acc += a_val * b_val;
    }

    C[row * params.ldc + col] = half(acc);
}

// Vectorized naive GEMM using half4 for better memory throughput
kernel void gemm_naive_vec4_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= params.M || col >= params.N) return;

    float acc = 0.0f;
    uint k = 0;

    // Vector loop: process 4 elements at a time
    for (; k + 4 <= params.K; k += 4) {
        half4 a_vec = *((device const half4*)(A + row * params.lda + k));
        half4 b_vec = *((device const half4*)(B + col * params.ldb + k));

        acc += float(a_vec.x) * float(b_vec.x);
        acc += float(a_vec.y) * float(b_vec.y);
        acc += float(a_vec.z) * float(b_vec.z);
        acc += float(a_vec.w) * float(b_vec.w);
    }

    // Remainder loop
    for (; k < params.K; k++) {
        float a_val = float(A[row * params.lda + k]);
        float b_val = float(B[col * params.ldb + k]);
        acc += a_val * b_val;
    }

    C[row * params.ldc + col] = half(acc);
}
"""

    # Tiled GEMM kernel with threadgroup memory
    # Optimized version with vectorized loads and larger register tile
    KERNEL_SOURCE_TILED = """
#include <metal_stdlib>
using namespace metal;

struct GEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

// Tile sizes optimized for Apple Silicon
// 64x64 output tile, 16 K tile for good compute-to-bandwidth ratio
constant uint TM = 64;  // Tile rows
constant uint TN = 64;  // Tile cols
constant uint TK = 16;  // Tile K dimension

// 16x16 threads, each handles 4x4 elements
constant uint RM = 4;   // Rows per thread
constant uint RN = 4;   // Cols per thread

// Tiled GEMM with vectorized loads and optimized compute
kernel void gemm_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    // Threadgroup memory with padding to avoid bank conflicts
    // +8 padding for half4 vectorized access alignment
    threadgroup half As[TM][TK + 8];
    threadgroup half Bs[TN][TK + 8];

    // Thread's position
    uint thread_row = lid.y * RM;
    uint thread_col = lid.x * RN;

    // Global tile position
    uint tile_row = tgid.y * TM;
    uint tile_col = tgid.x * TN;

    // Accumulators in registers
    float acc[RM][RN];
    for (uint i = 0; i < RM; i++) {
        for (uint j = 0; j < RN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    uint num_k_tiles = (params.K + TK - 1) / TK;
    uint tid = lid.y * 16 + lid.x;  // 0-255

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        // Cooperative load of A tile [TM, TK]
        // 256 threads, TM*TK = 64*16 = 1024 elements = 4 per thread
        for (uint elem = 0; elem < 4; elem++) {
            uint idx = tid + elem * 256;
            if (idx < TM * TK) {
                uint i = idx / TK;
                uint j = idx % TK;
                uint a_row = tile_row + i;
                uint a_col = kt * TK + j;
                if (a_row < params.M && a_col < params.K) {
                    As[i][j] = A[a_row * params.lda + a_col];
                } else {
                    As[i][j] = half(0.0f);
                }
            }
        }

        // Cooperative load of B tile [TN, TK]
        for (uint elem = 0; elem < 4; elem++) {
            uint idx = tid + elem * 256;
            if (idx < TN * TK) {
                uint i = idx / TK;
                uint j = idx % TK;
                uint b_row = tile_col + i;
                uint b_col = kt * TK + j;
                if (b_row < params.N && b_col < params.K) {
                    Bs[i][j] = B[b_row * params.ldb + b_col];
                } else {
                    Bs[i][j] = half(0.0f);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute with unrolled inner loop for better instruction scheduling
        // Process 4 K values at a time
        for (uint k = 0; k < TK; k += 4) {
            // Load A values for 4 K iterations
            float a0[RM], a1[RM], a2[RM], a3[RM];
            for (uint i = 0; i < RM; i++) {
                a0[i] = float(As[thread_row + i][k]);
                a1[i] = float(As[thread_row + i][k + 1]);
                a2[i] = float(As[thread_row + i][k + 2]);
                a3[i] = float(As[thread_row + i][k + 3]);
            }

            // Load B values for 4 K iterations
            float b0[RN], b1[RN], b2[RN], b3[RN];
            for (uint j = 0; j < RN; j++) {
                b0[j] = float(Bs[thread_col + j][k]);
                b1[j] = float(Bs[thread_col + j][k + 1]);
                b2[j] = float(Bs[thread_col + j][k + 2]);
                b3[j] = float(Bs[thread_col + j][k + 3]);
            }

            // 4 outer products
            for (uint i = 0; i < RM; i++) {
                for (uint j = 0; j < RN; j++) {
                    acc[i][j] += a0[i] * b0[j];
                    acc[i][j] += a1[i] * b1[j];
                    acc[i][j] += a2[i] * b2[j];
                    acc[i][j] += a3[i] * b3[j];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results with bounds checking
    for (uint i = 0; i < RM; i++) {
        uint c_row = tile_row + thread_row + i;
        if (c_row >= params.M) continue;

        for (uint j = 0; j < RN; j++) {
            uint c_col = tile_col + thread_col + j;
            if (c_col < params.N) {
                C[c_row * params.ldc + c_col] = half(acc[i][j]);
            }
        }
    }
}
"""

    # simdgroup GEMM kernel using hardware matrix multiply-accumulate
    # Optimized version with smaller staging buffer and better memory access
    KERNEL_SOURCE_SIMDGROUP = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct GEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

// Optimized tile sizes
// Each threadgroup has 4 simdgroups (128 threads)
// Each simdgroup handles a 32x32 output tile
constant uint TM = 64;   // Threadgroup tile rows
constant uint TN = 64;   // Threadgroup tile cols
constant uint TK = 32;   // K tile size

// simdgroup GEMM using hardware matrix multiply
// Optimized: smaller staging buffer, store one 8x8 at a time
kernel void gemm_simdgroup_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory for input tiles with padding to avoid bank conflicts
    threadgroup half As[TM][TK + 8];
    threadgroup half Bs[TN][TK + 8];

    // Small staging buffer per simdgroup for 8x8 float->half conversion
    // 4 simdgroups * 64 floats = 256 floats = 1KB
    threadgroup float stage[4][64];

    // Global tile position
    uint tile_row = tgid.y * TM;
    uint tile_col = tgid.x * TN;

    // Each simdgroup handles a 32x32 sub-tile
    uint sg_row = (simd_group / 2) * 32;
    uint sg_col = (simd_group % 2) * 32;

    // Accumulators: 4x4 = 16 8x8 matrix tiles per simdgroup
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Number of K tiles
    uint num_k_tiles = (params.K + TK - 1) / TK;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        // Cooperative load of A tile [TM, TK] - vectorized where possible
        for (uint elem = 0; elem < 16; elem++) {
            uint idx = tid + elem * 128;
            if (idx < TM * TK) {
                uint i = idx / TK;
                uint j = idx % TK;
                uint a_row = tile_row + i;
                uint a_col = kt * TK + j;
                if (a_row < params.M && a_col < params.K) {
                    As[i][j] = A[a_row * params.lda + a_col];
                } else {
                    As[i][j] = half(0.0f);
                }
            }
        }

        // Cooperative load of B tile [TN, TK]
        for (uint elem = 0; elem < 16; elem++) {
            uint idx = tid + elem * 128;
            if (idx < TN * TK) {
                uint i = idx / TK;
                uint j = idx % TK;
                uint b_row = tile_col + i;
                uint b_col = kt * TK + j;
                if (b_row < params.N && b_col < params.K) {
                    Bs[i][j] = B[b_row * params.ldb + b_col];
                } else {
                    Bs[i][j] = half(0.0f);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute using simdgroup matrix operations
        for (uint k_frag = 0; k_frag < TK; k_frag += 8) {
            simdgroup_matrix<half, 8, 8> A_frag[4];
            simdgroup_matrix<half, 8, 8> B_frag[4];

            // Load A and B fragments
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(A_frag[i], &As[sg_row + i * 8][k_frag], TK + 8);
            }
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(B_frag[j], &Bs[sg_col + j * 8][k_frag], TK + 8, ulong2(0, 0), true);
            }

            // Matrix multiply-accumulate (unrolled inner loop)
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results using per-simdgroup staging buffer
    // Each simdgroup has its own 64-float staging area in 'stage[4][64]'
    // This avoids cross-simdgroup barriers during the store phase.
    //
    // Process: For each 8x8 accumulator tile:
    //   1. simdgroup_store writes 64 floats to stage[simd_group]
    //   2. Each lane converts and writes 2 elements to device memory
    //   3. simdgroup_barrier ensures staging is complete before next tile
    //
    // Total: 16 tiles * 64 floats = 1024 elements per simdgroup (32x32 region)
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            // Store 8x8 accumulator to small staging buffer
            simdgroup_store(acc[i][j], &stage[simd_group][0], 8);

            // NO BARRIER - each simdgroup has its own staging buffer

            // Each lane stores 2 elements
            uint base_row = tile_row + sg_row + i * 8;
            uint base_col = tile_col + sg_col + j * 8;

            uint local_idx = simd_lane * 2;
            uint local_row = local_idx / 8;
            uint local_col = local_idx % 8;
            uint c_row = base_row + local_row;
            uint c_col = base_col + local_col;

            if (c_row < params.M && c_col < params.N) {
                C[c_row * params.ldc + c_col] = half(stage[simd_group][local_idx]);
            }

            local_idx = simd_lane * 2 + 1;
            local_row = local_idx / 8;
            local_col = local_idx % 8;
            c_row = base_row + local_row;
            c_col = base_col + local_col;

            if (c_row < params.M && c_col < params.N) {
                C[c_row * params.ldc + c_col] = half(stage[simd_group][local_idx]);
            }

            // Barrier only between different tiles to ensure staging buffer is free
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"""

    # Large-M tiled kernel with 8x8 elements per thread for better register utilization
    KERNEL_SOURCE_TILED_LARGE = """
#include <metal_stdlib>
using namespace metal;

struct GEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

// Larger tiles for bigger matrices
// 128x128 output tile with 16x16 threads, each handling 8x8 elements
constant uint TM = 128;  // Tile rows
constant uint TN = 128;  // Tile cols
constant uint TK = 16;   // Tile K dimension

constant uint RM = 8;    // Rows per thread
constant uint RN = 8;    // Cols per thread

kernel void gemm_tiled_large_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup half As[TM][TK + 4];
    threadgroup half Bs[TN][TK + 4];

    uint thread_row = lid.y * RM;
    uint thread_col = lid.x * RN;

    uint tile_row = tgid.y * TM;
    uint tile_col = tgid.x * TN;

    // 64 accumulators per thread
    float acc[RM][RN];
    for (uint i = 0; i < RM; i++) {
        for (uint j = 0; j < RN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    uint num_k_tiles = (params.K + TK - 1) / TK;
    uint tid = lid.y * 16 + lid.x;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        // Load A tile [128, 16] - 2048 elements, 8 per thread
        for (uint elem = 0; elem < 8; elem++) {
            uint idx = tid + elem * 256;
            if (idx < TM * TK) {
                uint i = idx / TK;
                uint j = idx % TK;
                uint a_row = tile_row + i;
                uint a_col = kt * TK + j;
                if (a_row < params.M && a_col < params.K) {
                    As[i][j] = A[a_row * params.lda + a_col];
                } else {
                    As[i][j] = half(0.0f);
                }
            }
        }

        // Load B tile [128, 16] - 2048 elements, 8 per thread
        for (uint elem = 0; elem < 8; elem++) {
            uint idx = tid + elem * 256;
            if (idx < TN * TK) {
                uint i = idx / TK;
                uint j = idx % TK;
                uint b_row = tile_col + i;
                uint b_col = kt * TK + j;
                if (b_row < params.N && b_col < params.K) {
                    Bs[i][j] = B[b_row * params.ldb + b_col];
                } else {
                    Bs[i][j] = half(0.0f);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 8x8 partial products
        for (uint k = 0; k < TK; k++) {
            float a_vals[RM];
            float b_vals[RN];

            for (uint i = 0; i < RM; i++) {
                a_vals[i] = float(As[thread_row + i][k]);
            }
            for (uint j = 0; j < RN; j++) {
                b_vals[j] = float(Bs[thread_col + j][k]);
            }

            for (uint i = 0; i < RM; i++) {
                for (uint j = 0; j < RN; j++) {
                    acc[i][j] = fma(a_vals[i], b_vals[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write 8x8 results
    for (uint i = 0; i < RM; i++) {
        uint c_row = tile_row + thread_row + i;
        if (c_row >= params.M) continue;

        for (uint j = 0; j < RN; j++) {
            uint c_col = tile_col + thread_col + j;
            if (c_col < params.N) {
                C[c_row * params.ldc + c_col] = half(acc[i][j]);
            }
        }
    }
}
"""

    # Small-M kernel optimized for decode (M=1-8)
    KERNEL_SOURCE_SMALL_M = """
#include <metal_stdlib>
using namespace metal;

struct GEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

// Small-M GEMM: optimized for M <= 8 (decode phase)
// Each simdgroup handles one row, spreads work across N dimension
// Vectorized loads for maximum memory throughput
kernel void gemm_small_m_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // tgid indexes the N dimension
    // Each threadgroup handles multiple N columns
    // Multiple rows (M) processed by different simdgroups

    const uint COLS_PER_THREADGROUP = 256;
    const uint SIMDGROUPS_PER_TG = 4;

    uint row = simd_group % params.M;
    uint col_base = tgid * COLS_PER_THREADGROUP + (simd_group / params.M) * 64 + simd_lane * 2;

    if (row >= params.M) return;

    // Process 2 columns per thread
    float2 acc = float2(0.0f);

    uint col0 = col_base;
    uint col1 = col_base + 1;

    if (col1 < params.N) {
        // Both columns valid
        for (uint k = 0; k < params.K; k += 4) {
            half4 a_vec = *((device const half4*)(A + row * params.lda + k));

            half4 b0_vec = *((device const half4*)(B + col0 * params.ldb + k));
            half4 b1_vec = *((device const half4*)(B + col1 * params.ldb + k));

            acc.x += dot(float4(a_vec), float4(b0_vec));
            acc.y += dot(float4(a_vec), float4(b1_vec));
        }

        C[row * params.ldc + col0] = half(acc.x);
        C[row * params.ldc + col1] = half(acc.y);
    } else if (col0 < params.N) {
        // Only first column valid
        for (uint k = 0; k < params.K; k += 4) {
            half4 a_vec = *((device const half4*)(A + row * params.lda + k));
            half4 b_vec = *((device const half4*)(B + col0 * params.ldb + k));
            acc.x += dot(float4(a_vec), float4(b_vec));
        }
        C[row * params.ldc + col0] = half(acc.x);
    }
}
"""

    def __init__(self, context: Any):  # MetalEngineContext
        """Initialize Metal GEMM op.

        Args:
            context: MetalEngineContext
        """
        self._context = context
        self._device = context.device

        # Kernel pipelines
        self._pipeline_naive = None
        self._pipeline_naive_vec4 = None
        self._pipeline_tiled = None
        self._pipeline_tiled_large = None
        self._pipeline_simdgroup = None
        self._pipeline_small_m = None

        # Stats
        self._encode_count = 0
        self._total_flops = 0

        # Compile kernels
        self._compile_kernels()

        logger.info("EngineGEMMMetal initialized with custom Metal kernels")

    def _compile_kernels(self) -> None:
        """Compile all GEMM Metal kernels."""
        # Compile naive kernels
        self._context.compile_library("gemm_naive", source_code=self.KERNEL_SOURCE_NAIVE)
        self._pipeline_naive = self._context.get_pipeline("gemm_naive", "gemm_naive_f16")
        self._pipeline_naive_vec4 = self._context.get_pipeline("gemm_naive", "gemm_naive_vec4_f16")

        # Compile tiled kernel
        self._context.compile_library("gemm_tiled", source_code=self.KERNEL_SOURCE_TILED)
        self._pipeline_tiled = self._context.get_pipeline("gemm_tiled", "gemm_tiled_f16")

        # Compile large tiled kernel (for larger matrices)
        self._context.compile_library("gemm_tiled_large", source_code=self.KERNEL_SOURCE_TILED_LARGE)
        self._pipeline_tiled_large = self._context.get_pipeline("gemm_tiled_large", "gemm_tiled_large_f16")

        # Compile simdgroup kernel (uses hardware matrix multiply)
        try:
            self._context.compile_library("gemm_simdgroup", source_code=self.KERNEL_SOURCE_SIMDGROUP)
            self._pipeline_simdgroup = self._context.get_pipeline("gemm_simdgroup", "gemm_simdgroup_f16")
            logger.info("Compiled simdgroup GEMM kernel (hardware MMA)")
        except Exception as e:
            logger.warning(f"Failed to compile simdgroup kernel: {e}")
            self._pipeline_simdgroup = None

        # Compile small-M kernel
        self._context.compile_library("gemm_small_m", source_code=self.KERNEL_SOURCE_SMALL_M)
        self._pipeline_small_m = self._context.get_pipeline("gemm_small_m", "gemm_small_m_f16")

        logger.info("Compiled custom Metal GEMM kernels: naive, naive_vec4, tiled, simdgroup, small_m")

    def _validate_dtype(self, tensor: Union[EngineTensor, Any], name: str) -> None:
        """Validate tensor dtype is FP16.

        CRITICAL: All Metal GEMM kernels operate on half (float16) data.
        Passing other dtypes will result in undefined behavior.

        Args:
            tensor: EngineTensor to validate
            name: Name for error message

        Raises:
            ValueError: If dtype is not float16
        """
        if isinstance(tensor, EngineTensor):
            if tensor.dtype != EngineDType.FLOAT16:
                raise ValueError(
                    f"EngineGEMMMetal requires FP16 tensors. "
                    f"Tensor '{name}' has dtype {tensor.dtype}. "
                    f"Metal kernels use 'half' type - FP32/other dtypes are not supported."
                )
        # For raw MTLBuffers, we cannot validate dtype - caller must ensure FP16
        # This is documented in the encode() docstring

    def _validate_alignment(self, tensor: Union[EngineTensor, Any], name: str, K: int) -> bool:
        """Check if tensor meets alignment requirements for vectorized kernels.

        Vectorized kernels (naive_vec4, small_m) use half4 loads which require:
        - Buffer offset to be 8-byte aligned (half4 = 4 * 2 bytes)
        - K to be divisible by 4 for contiguous access

        Args:
            tensor: Tensor to check
            name: Name for logging
            K: Inner dimension

        Returns:
            True if aligned for vec4 access, False otherwise
        """
        # K must be divisible by 4 for half4 loads
        if K % 4 != 0:
            return False

        # Check offset alignment if EngineTensor
        if isinstance(tensor, EngineTensor):
            # half4 requires 8-byte alignment
            if tensor.offset % 8 != 0:
                return False

        return True

    def _select_kernel(self, M: int, K: int, N: int, A: Any, B: Any) -> Tuple[Any, str]:
        """Select optimal kernel based on matrix dimensions and alignment.

        Kernel selection priority:
        1. small_m for M <= 8 with aligned K (decode optimization)
        2. tiled for M >= 16 with sufficient N and K
        3. naive_vec4 for aligned K
        4. naive as fallback

        Args:
            M: Rows of output
            K: Inner dimension
            N: Columns of output
            A: Input tensor A (for alignment check)
            B: Input tensor B (for alignment check)

        Returns:
            Tuple of (pipeline, kernel_name)
        """
        # NOTE: simdgroup kernel disabled - float->half conversion overhead
        # makes it slower than tiled/naive kernels currently.
        # NOTE: tiled_large kernel disabled - register pressure causes slowdown

        # Check alignment for vectorized kernels
        is_aligned = self._validate_alignment(A, "A", K) and self._validate_alignment(B, "B", K)

        # Small-M kernel for decode (M <= 4) with aligned K
        # This kernel has 4 simdgroups per threadgroup, each handles one row.
        # For M > 4, some rows won't be computed correctly.
        # NOTE: M <= 4 restriction due to SIMDGROUPS_PER_TG = 4 in kernel
        if M <= 4 and is_aligned and self._pipeline_small_m is not None:
            if is_gemm_debug():
                logger.debug(f"Selected small_m kernel for M={M}, K={K}, N={N}")
            return self._pipeline_small_m, "small_m"

        # Tiled for medium-to-large matrices (M >= 16)
        # 64x64 tiles with 4x4 elements per thread is optimal
        if M >= 16 and N >= 64 and K >= 16:
            return self._pipeline_tiled, "tiled"

        # Vectorized naive for aligned K
        if is_aligned:
            return self._pipeline_naive_vec4, "naive_vec4"

        # Fallback to scalar naive kernel
        return self._pipeline_naive, "naive"

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        A: Union[EngineTensor, Any],  # MTLBuffer
        B: Union[EngineTensor, Any],  # MTLBuffer
        C: Union[EngineTensor, Any],  # MTLBuffer
        M: Optional[int] = None,
        K: Optional[int] = None,
        N: Optional[int] = None,
        transpose_B: bool = True,  # Default: weights are [N, K] (transposed)
    ) -> None:
        """Encode GEMM to command buffer.

        Computes: C = A @ B^T

        IMPORTANT: All Metal GEMM kernels assume B is stored transposed as [N, K].
        This matches the PyTorch weight convention [out_features, in_features].
        transpose_B=False is NOT supported - use MPS GEMM for that case.

        Args:
            step_ctx: EngineStepContext with encoder
            A: Left matrix [M, K], dtype must be float16
            B: Right matrix [N, K] (transposed storage), dtype must be float16
            C: Result matrix [M, N], dtype must be float16
            M: Rows of A (inferred from tensor if not provided)
            K: Cols of A
            N: Cols of C (rows of B since transposed)
            transpose_B: MUST be True - Metal kernels only support transposed B

        Raises:
            RuntimeError: If called outside ENCODE phase
            ValueError: If transpose_B=False (unsupported)
            ValueError: If tensor dtypes are not float16
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # CRITICAL: Enforce transpose_B contract
        # All Metal kernels assume B is [N, K] (transposed storage)
        if not transpose_B:
            raise ValueError(
                "EngineGEMMMetal only supports transpose_B=True. "
                "All kernels assume B is stored as [N, K] (transposed). "
                "Use MPS GEMM for non-transposed B matrices."
            )

        # CRITICAL: Validate FP16 dtype
        self._validate_dtype(A, "A")
        self._validate_dtype(B, "B")
        self._validate_dtype(C, "C")

        # Extract buffer info
        if isinstance(A, EngineTensor):
            A_buf = A.buffer
            A_offset = A.offset
            if M is None:
                M = A.shape[0]
            if K is None:
                K = A.shape[1]
        else:
            A_buf = A
            A_offset = 0
            if M is None or K is None:
                raise ValueError("M, K must be provided for raw MTLBuffer A")

        if isinstance(B, EngineTensor):
            B_buf = B.buffer
            B_offset = B.offset
            if transpose_B:
                if N is None:
                    N = B.shape[0]
            else:
                if N is None:
                    N = B.shape[1]
        else:
            B_buf = B
            B_offset = 0
            if N is None:
                raise ValueError("N must be provided for raw MTLBuffer B")

        if isinstance(C, EngineTensor):
            C_buf = C.buffer
            C_offset = C.offset
        else:
            C_buf = C
            C_offset = 0

        # Select kernel (pass tensors for alignment validation)
        pipeline, kernel_name = self._select_kernel(M, K, N, A, B)

        # Get compute encoder
        encoder = step_ctx.get_compute_encoder()

        # Set pipeline
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(A_buf, A_offset, 0)
        encoder.setBuffer_offset_atIndex_(B_buf, B_offset, 1)
        encoder.setBuffer_offset_atIndex_(C_buf, C_offset, 2)

        # Set parameters
        # GEMMParams: M, N, K, lda, ldb, ldc
        lda = K  # A is [M, K] row-major
        ldb = K  # B is [N, K] row-major (transposed)
        ldc = N  # C is [M, N] row-major
        params_data = struct.pack("IIIIII", M, N, K, lda, ldb, ldc)
        encoder.setBytes_length_atIndex_(params_data, len(params_data), 3)

        # Dispatch based on kernel type
        if kernel_name == "simdgroup":
            # simdgroup kernel: 64x64 tiles, 4 simdgroups (128 threads) per group
            TM, TN = 64, 64
            num_groups_m = (M + TM - 1) // TM
            num_groups_n = (N + TN - 1) // TN
            thread_groups = MTLSize(num_groups_n, num_groups_m, 1)
            threads_per_threadgroup = MTLSize(128, 1, 1)  # 4 simdgroups
        elif kernel_name == "tiled_large":
            # Large tiled kernel: 128x128 tiles, 16x16 threads
            TM, TN = 128, 128
            num_groups_m = (M + TM - 1) // TM
            num_groups_n = (N + TN - 1) // TN
            thread_groups = MTLSize(num_groups_n, num_groups_m, 1)
            threads_per_threadgroup = MTLSize(16, 16, 1)
        elif kernel_name == "tiled":
            # Tiled kernel: one threadgroup per TM x TN tile (64x64)
            # 16x16 threads per group, each handling 4x4 elements
            TM, TN = 64, 64
            num_groups_m = (M + TM - 1) // TM
            num_groups_n = (N + TN - 1) // TN
            thread_groups = MTLSize(num_groups_n, num_groups_m, 1)
            threads_per_threadgroup = MTLSize(16, 16, 1)
        elif kernel_name == "small_m":
            # Small-M: spread across N dimension
            COLS_PER_TG = 256
            num_groups = (N + COLS_PER_TG - 1) // COLS_PER_TG
            thread_groups = MTLSize(num_groups, 1, 1)
            threads_per_threadgroup = MTLSize(128, 1, 1)  # 4 simdgroups
        else:
            # Naive kernels: one thread per output element
            thread_groups = MTLSize((N + 15) // 16, (M + 15) // 16, 1)
            threads_per_threadgroup = MTLSize(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, threads_per_threadgroup
        )

        # Update stats
        self._encode_count += 1
        self._total_flops += 2 * M * K * N

        if is_gemm_debug():
            logger.debug(
                f"GEMM encode: M={M}, K={K}, N={N}, kernel={kernel_name}, "
                f"groups={thread_groups}, threads={threads_per_threadgroup}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            "encode_count": self._encode_count,
            "total_flops": self._total_flops,
            "total_tflops": self._total_flops / 1e12,
        }
