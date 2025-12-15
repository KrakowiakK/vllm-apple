# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for custom Metal GEMM kernels.

Tests correctness of EngineGEMMMetal against numpy reference.

Test shapes:
- (1, 4096, 4096) - Single token decode
- (8, 4096, 4096) - Small batch decode
- (128, 4096, 4096) - Medium prefill
- (128, 4096, 12288) - QKV projection shape
- (256, 11008, 4096) - MLP down shape
- (1, 4096, 128256) - LM head (large N)
- (17, 4095, 4097) - Non-aligned edge case
"""

import pytest
import numpy as np
import time
import struct
from typing import Any, Optional

# Skip all tests if Metal not available
pytest.importorskip("Metal")


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def metal_device():
    """Create Metal device for tests."""
    from Metal import MTLCreateSystemDefaultDevice
    device = MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("Metal device not available")
    return device


@pytest.fixture
def engine_context():
    """Create MetalEngineContext for tests."""
    from vllm_apple.engine.context import MetalEngineContext
    ctx = MetalEngineContext.get_instance()
    return ctx


@pytest.fixture
def gemm_metal(engine_context):
    """Create EngineGEMMMetal for tests."""
    from vllm_apple.engine.ops.gemm_metal import EngineGEMMMetal
    return EngineGEMMMetal(engine_context)


class MockStepContext:
    """Mock step context for testing kernel encoding.

    Provides minimal implementation of EngineStepContext interface
    needed for GEMM testing.
    """

    def __init__(self, device: Any, command_queue: Any):
        self._device = device
        self._command_queue = command_queue
        self._cmd_buffer = None
        self._encoder = None
        self._is_encoding = True

    @property
    def is_encoding(self) -> bool:
        return self._is_encoding

    def _ensure_cmd_buffer(self):
        if self._cmd_buffer is None:
            self._cmd_buffer = self._command_queue.commandBuffer()
            if self._cmd_buffer is None:
                raise RuntimeError("Failed to create command buffer")

    def get_compute_encoder(self):
        self._ensure_cmd_buffer()
        if self._encoder is None:
            self._encoder = self._cmd_buffer.computeCommandEncoder()
            if self._encoder is None:
                raise RuntimeError("Failed to create compute encoder")
        return self._encoder

    def end_compute_encoder_for_mps(self):
        """End encoder and return command buffer for MPS encoding."""
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        self._ensure_cmd_buffer()
        return self._cmd_buffer

    def commit_and_wait(self):
        """Commit command buffer and wait for completion."""
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        if self._cmd_buffer is not None:
            self._cmd_buffer.commit()
            self._cmd_buffer.waitUntilCompleted()
            self._cmd_buffer = None


@pytest.fixture
def step_context(engine_context):
    """Create mock step context for tests."""
    return MockStepContext(engine_context.device, engine_context.command_queue)


def create_metal_buffer(device: Any, data: np.ndarray) -> Any:
    """Create MTLBuffer from numpy array."""
    from Metal import MTLResourceStorageModeShared
    data_bytes = data.astype(np.float16).tobytes()
    buffer = device.newBufferWithBytes_length_options_(
        data_bytes, len(data_bytes), MTLResourceStorageModeShared
    )
    return buffer


def read_metal_buffer(buffer: Any, shape: tuple, dtype=np.float16) -> np.ndarray:
    """Read numpy array from MTLBuffer."""
    import ctypes
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    contents = buffer.contents()
    data = ctypes.string_at(contents, size)
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def gemm_reference(A: np.ndarray, B: np.ndarray, transpose_B: bool = True) -> np.ndarray:
    """Numpy reference GEMM.

    Args:
        A: Left matrix [M, K]
        B: Right matrix [N, K] if transpose_B else [K, N]
        transpose_B: Whether B is stored transposed

    Returns:
        C: Result [M, N]
    """
    A_fp32 = A.astype(np.float32)
    B_fp32 = B.astype(np.float32)
    if transpose_B:
        C = A_fp32 @ B_fp32.T
    else:
        C = A_fp32 @ B_fp32
    return C.astype(np.float16)


# ============================================================================
# GEMM Correctness Tests
# ============================================================================


class TestGEMMMetalCorrectness:
    """Tests for custom Metal GEMM correctness."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),      # Single token
        (8, 4096, 4096),      # Small batch decode
        (32, 4096, 4096),     # Medium batch
        (128, 4096, 4096),    # Larger batch
    ])
    def test_gemm_square_shapes(self, engine_context, gemm_metal, step_context, M, K, N):
        """Test GEMM with square-ish shapes."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)

        expected = gemm_reference(A, B, transpose_B=True)

        # Create buffers
        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(
            M * N * 2, 0  # MTLResourceStorageModeShared
        )

        # Encode GEMM
        gemm_metal.encode(step_context, A_buf, B_buf, C_buf, M=M, K=K, N=N)

        # Execute
        step_context.commit_and_wait()

        # Read result
        actual = read_metal_buffer(C_buf, (M, N))

        # Compare with tolerance for float16
        np.testing.assert_allclose(
            actual, expected, rtol=1e-2, atol=1e-2,
            err_msg=f"GEMM mismatch for shape ({M}, {K}, {N})"
        )

    @pytest.mark.parametrize("M,K,N", [
        (128, 4096, 12288),   # QKV projection
        (256, 4096, 11008),   # MLP up/gate
        (256, 11008, 4096),   # MLP down
    ])
    def test_gemm_model_shapes(self, engine_context, gemm_metal, step_context, M, K, N):
        """Test GEMM with typical model shapes."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)

        expected = gemm_reference(A, B, transpose_B=True)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(
            M * N * 2, 0
        )

        gemm_metal.encode(step_context, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_context.commit_and_wait()

        actual = read_metal_buffer(C_buf, (M, N))
        np.testing.assert_allclose(
            actual, expected, rtol=1e-2, atol=1e-2,
            err_msg=f"GEMM mismatch for shape ({M}, {K}, {N})"
        )

    @pytest.mark.parametrize("M,K,N", [
        (17, 4095, 4097),     # Non-aligned
        (13, 127, 255),       # Small odd sizes
        (1, 512, 1024),       # Decode small K
    ])
    def test_gemm_edge_cases(self, engine_context, gemm_metal, step_context, M, K, N):
        """Test GEMM with edge case shapes."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)

        expected = gemm_reference(A, B, transpose_B=True)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(
            M * N * 2, 0
        )

        gemm_metal.encode(step_context, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_context.commit_and_wait()

        actual = read_metal_buffer(C_buf, (M, N))
        np.testing.assert_allclose(
            actual, expected, rtol=1e-2, atol=1e-2,
            err_msg=f"GEMM mismatch for shape ({M}, {K}, {N})"
        )

    def test_gemm_large_n_lm_head(self, engine_context, gemm_metal, step_context):
        """Test GEMM with large N (LM head shape)."""
        # Smaller than real LM head but tests large-N path
        M, K, N = 1, 4096, 32000

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)

        expected = gemm_reference(A, B, transpose_B=True)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(
            M * N * 2, 0
        )

        gemm_metal.encode(step_context, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_context.commit_and_wait()

        actual = read_metal_buffer(C_buf, (M, N))
        np.testing.assert_allclose(
            actual, expected, rtol=1e-2, atol=1e-2,
            err_msg=f"GEMM mismatch for LM head shape ({M}, {K}, {N})"
        )


# ============================================================================
# GEMM vs MPS Comparison Tests
# ============================================================================


class TestGEMMMetalVsMPS:
    """Compare custom Metal GEMM against MPS baseline."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),
        (8, 4096, 4096),
        (128, 4096, 4096),
    ])
    def test_gemm_matches_mps(self, engine_context, gemm_metal, step_context, M, K, N):
        """Test that custom Metal GEMM produces same results as MPS."""
        from vllm_apple.engine.ops.gemm import EngineGEMM

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)

        # Run Metal GEMM
        C_metal = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)
        gemm_metal.encode(step_context, A_buf, B_buf, C_metal, M=M, K=K, N=N)
        step_context.commit_and_wait()
        metal_result = read_metal_buffer(C_metal, (M, N))

        # Run MPS GEMM
        step_context_mps = MockStepContext(engine_context.device, engine_context.command_queue)
        mps_gemm = EngineGEMM(engine_context)
        C_mps = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)
        mps_gemm.encode(step_context_mps, A_buf, B_buf, C_mps, M=M, K=K, N=N, transpose_B=True)
        step_context_mps.commit_and_wait()
        mps_result = read_metal_buffer(C_mps, (M, N))

        # Compare
        np.testing.assert_allclose(
            metal_result, mps_result, rtol=1e-2, atol=1e-2,
            err_msg=f"Metal GEMM doesn't match MPS for shape ({M}, {K}, {N})"
        )


# ============================================================================
# GEMM Performance Tests
# ============================================================================


class TestGEMMMetalPerformance:
    """Performance benchmarks for custom Metal GEMM."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),      # Decode
        (128, 4096, 4096),    # Prefill
    ])
    def test_gemm_performance(self, engine_context, gemm_metal, M, K, N):
        """Benchmark GEMM performance."""
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)

        # Warmup
        for _ in range(5):
            step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
            gemm_metal.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N)
            step_ctx.commit_and_wait()

        # Benchmark
        num_iters = 20
        start = time.perf_counter()
        for _ in range(num_iters):
            step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
            gemm_metal.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N)
            step_ctx.commit_and_wait()
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / num_iters) * 1000
        flops = 2 * M * K * N
        tflops = (flops / (elapsed / num_iters)) / 1e12

        print(f"\nGEMM ({M}x{K}x{N}): {avg_time_ms:.2f} ms, {tflops:.2f} TFLOPS")

        # Just verify it ran
        assert avg_time_ms > 0


# ============================================================================
# Unified GEMM Selector Tests
# ============================================================================


class TestUnifiedGEMM:
    """Tests for UnifiedGEMM selector."""

    def test_unified_gemm_auto_backend(self, engine_context, step_context):
        """Test unified GEMM with auto backend selection."""
        from vllm_apple.engine.ops.gemm_selector import UnifiedGEMM

        gemm = UnifiedGEMM(engine_context, backend="auto")

        M, K, N = 8, 4096, 4096
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)
        expected = gemm_reference(A, B, transpose_B=True)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)

        gemm.encode(step_context, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_context.commit_and_wait()

        actual = read_metal_buffer(C_buf, (M, N))
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

    def test_unified_gemm_metal_backend(self, engine_context, step_context):
        """Test unified GEMM with forced metal backend."""
        from vllm_apple.engine.ops.gemm_selector import UnifiedGEMM

        gemm = UnifiedGEMM(engine_context, backend="metal")

        M, K, N = 8, 4096, 4096
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)
        expected = gemm_reference(A, B, transpose_B=True)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)

        gemm.encode(step_context, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_context.commit_and_wait()

        actual = read_metal_buffer(C_buf, (M, N))
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

    def test_unified_gemm_mps_backend(self, engine_context):
        """Test unified GEMM with forced MPS backend."""
        from vllm_apple.engine.ops.gemm_selector import UnifiedGEMM

        gemm = UnifiedGEMM(engine_context, backend="mps")
        step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)

        M, K, N = 8, 4096, 4096
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)
        expected = gemm_reference(A, B, transpose_B=True)

        A_buf = create_metal_buffer(engine_context.device, A)
        B_buf = create_metal_buffer(engine_context.device, B)
        C_buf = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)

        gemm.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_ctx.commit_and_wait()

        actual = read_metal_buffer(C_buf, (M, N))
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
