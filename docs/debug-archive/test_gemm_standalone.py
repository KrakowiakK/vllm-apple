#!/usr/bin/env python3
"""Standalone test for custom Metal GEMM kernels.

Run with: python3 test_gemm_standalone.py

Tests correctness of EngineGEMMMetal against numpy reference.
"""

import numpy as np
import time
import sys
import traceback


def test_metal_available():
    """Check if Metal is available."""
    try:
        import Metal
        from Metal import MTLCreateSystemDefaultDevice
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            print("ERROR: Metal device not available")
            return False
        print(f"Metal device: {device.name()}")
        return True
    except ImportError as e:
        print(f"ERROR: Metal not available: {e}")
        return False


def create_metal_buffer(device, data):
    """Create MTLBuffer from numpy array."""
    from Metal import MTLResourceStorageModeShared
    data_bytes = data.astype(np.float16).tobytes()
    buffer = device.newBufferWithBytes_length_options_(
        data_bytes, len(data_bytes), MTLResourceStorageModeShared
    )
    return buffer


def read_metal_buffer(buffer, shape, dtype=np.float16):
    """Read numpy array from MTLBuffer using as_buffer() method."""
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize

    # Use PyObjC's as_buffer method (works in PyObjC 12+)
    buffer_view = buffer.contents().as_buffer(size)
    return np.frombuffer(buffer_view, dtype=dtype).reshape(shape).copy()


def gemm_reference(A, B, transpose_B=True):
    """Numpy reference GEMM."""
    A_fp32 = A.astype(np.float32)
    B_fp32 = B.astype(np.float32)
    if transpose_B:
        C = A_fp32 @ B_fp32.T
    else:
        C = A_fp32 @ B_fp32
    return C.astype(np.float16)


class MockStepContext:
    """Mock step context for testing kernel encoding."""

    def __init__(self, device, command_queue):
        self._device = device
        self._command_queue = command_queue
        self._cmd_buffer = None
        self._encoder = None
        self._is_encoding = True

    @property
    def is_encoding(self):
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
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        self._ensure_cmd_buffer()
        return self._cmd_buffer

    def commit_and_wait(self):
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        if self._cmd_buffer is not None:
            self._cmd_buffer.commit()
            self._cmd_buffer.waitUntilCompleted()
            self._cmd_buffer = None


def test_gemm_correctness(M, K, N, gemm_metal, engine_context):
    """Test GEMM correctness for given shape."""
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(N, K).astype(np.float16)

    expected = gemm_reference(A, B, transpose_B=True)

    # Create buffers
    A_buf = create_metal_buffer(engine_context.device, A)
    B_buf = create_metal_buffer(engine_context.device, B)
    C_buf = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)

    # Create step context
    step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)

    # Encode GEMM
    gemm_metal.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N)

    # Execute
    step_ctx.commit_and_wait()

    # Read result
    actual = read_metal_buffer(C_buf, (M, N))

    # Compare
    max_diff = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    rel_diff = max_diff / (np.max(np.abs(expected.astype(np.float32))) + 1e-8)

    return max_diff, rel_diff, actual, expected


def test_gemm_performance(M, K, N, gemm_metal, engine_context, num_iters=20):
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
    start = time.perf_counter()
    for _ in range(num_iters):
        step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
        gemm_metal.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N)
        step_ctx.commit_and_wait()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / num_iters) * 1000
    flops = 2 * M * K * N
    tflops = (flops / (elapsed / num_iters)) / 1e12

    return avg_time_ms, tflops


def test_mps_gemm_performance(M, K, N, mps_gemm, engine_context, num_iters=20):
    """Benchmark MPS GEMM performance for comparison."""
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(N, K).astype(np.float16)

    A_buf = create_metal_buffer(engine_context.device, A)
    B_buf = create_metal_buffer(engine_context.device, B)
    C_buf = engine_context.device.newBufferWithLength_options_(M * N * 2, 0)

    # Warmup
    for _ in range(5):
        step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
        mps_gemm.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N, transpose_B=True)
        step_ctx.commit_and_wait()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
        mps_gemm.encode(step_ctx, A_buf, B_buf, C_buf, M=M, K=K, N=N, transpose_B=True)
        step_ctx.commit_and_wait()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / num_iters) * 1000
    flops = 2 * M * K * N
    tflops = (flops / (elapsed / num_iters)) / 1e12

    return avg_time_ms, tflops


def main():
    print("=" * 60)
    print("Custom Metal GEMM Kernel Tests")
    print("=" * 60)

    # Check Metal availability
    if not test_metal_available():
        return 1

    # Import engine components
    try:
        from vllm_apple.engine.context import MetalEngineContext
        from vllm_apple.engine.ops.gemm_metal import EngineGEMMMetal
        from vllm_apple.engine.ops.gemm import EngineGEMM
        print("Engine modules imported successfully")
    except ImportError as e:
        print(f"ERROR: Failed to import engine modules: {e}")
        traceback.print_exc()
        return 1

    # Create context and GEMM ops
    try:
        engine_context = MetalEngineContext.get_instance()
        gemm_metal = EngineGEMMMetal(engine_context)
        mps_gemm = EngineGEMM(engine_context)
        print("GEMM operations created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create GEMM ops: {e}")
        traceback.print_exc()
        return 1

    # Test shapes
    test_shapes = [
        (1, 4096, 4096, "Single token decode"),
        (8, 4096, 4096, "Small batch decode"),
        (32, 4096, 4096, "Medium batch"),
        (128, 4096, 4096, "Prefill batch"),
        (128, 4096, 12288, "QKV projection"),
        (256, 4096, 11008, "MLP up/gate"),
        (256, 11008, 4096, "MLP down"),
        (1, 4096, 32000, "LM head (large N)"),
        (17, 127, 255, "Edge case (non-aligned)"),
    ]

    print("\n" + "=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    all_passed = True
    for M, K, N, desc in test_shapes:
        try:
            max_diff, rel_diff, actual, expected = test_gemm_correctness(
                M, K, N, gemm_metal, engine_context
            )
            passed = rel_diff < 0.05  # 5% relative tolerance
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(f"[{status}] {desc} ({M}x{K}x{N}): max_diff={max_diff:.4e}, rel_diff={rel_diff:.4e}")
        except Exception as e:
            all_passed = False
            print(f"[ERROR] {desc} ({M}x{K}x{N}): {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Custom Metal vs MPS")
    print("=" * 60)

    perf_shapes = [
        (1, 4096, 4096, "Decode (M=1)"),
        (8, 4096, 4096, "Decode (M=8)"),
        (128, 4096, 4096, "Prefill"),
        (128, 4096, 12288, "QKV"),
        (256, 11008, 4096, "MLP down"),
    ]

    print(f"\n{'Shape':<25} {'Metal (ms)':<12} {'MPS (ms)':<12} {'Metal TFLOPS':<12} {'MPS TFLOPS':<12} {'Speedup':<10}")
    print("-" * 85)

    for M, K, N, desc in perf_shapes:
        try:
            metal_ms, metal_tflops = test_gemm_performance(M, K, N, gemm_metal, engine_context)
            mps_ms, mps_tflops = test_mps_gemm_performance(M, K, N, mps_gemm, engine_context)
            speedup = mps_ms / metal_ms if metal_ms > 0 else 0
            print(f"({M}x{K}x{N}){'':<10} {metal_ms:<12.3f} {mps_ms:<12.3f} {metal_tflops:<12.3f} {mps_tflops:<12.3f} {speedup:<10.2f}x")
        except Exception as e:
            print(f"({M}x{K}x{N}){'':<10} ERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_passed:
        print("All correctness tests PASSED")
    else:
        print("Some correctness tests FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
