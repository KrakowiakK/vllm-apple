# Performance Optimization Plan: How to beat llama.cpp

**Current Status (v2.6-stable)**: ~10 tok/s (Devstral 22B, Batch 1).
**Hardware Limit (M3 Ultra)**: ~18 tok/s (FP16), ~60 tok/s (INT4).
**Bottleck**: Python Dispatch Overhead (CPU) + Memory Bandwidth (FP16).

To match or beat `llama.cpp` and `mlx-lm`, we must execute the following two phases.

---

## 1. Eliminate CPU Overhead (Target: 18 tok/s)
**Problem**: Currently, `runner.py` iterates over 40 layers in Python.
- CPU creates 40 separate command encoders.
- CPU submits work to GPU 40 times.
- Overhead: ~1ms per layer * 40 layers = **40ms wasted per token**.
- This caps us at ~10-12 tok/s regardless of GPU speed.

**Solution: Metal Graph Capture (Recommended Next Step)**
We must implement a mechanism to "record" the entire model execution into a single Metal Graph or Command Buffer.

**Implementation Strategy**:
1.  **Refactor `runner.py`**:
    *   Create a `capture_model_graph()` method.
    *   Run the model once with "fake" capture tensors.
    *   Record all Metal commands into an `MTLIndirectCommandBuffer` (ICB) or use `MPSGraph`.
2.  **Execution Config**:
    *   During inference, update only the pointers (KV cache positions) in the ICB.
    *   Submit ONE command buffer for the entire model.
3.  **Result**: CPU overhead drops to ~0ms. Latency drops from 96ms → ~55ms (Speed of light for FP16).
4.  **Difficulty**: High (Requires careful memory management of dynamic KV cache in Metal).

---

## 2. Reduce Memory Traffic (Target: 50+ tok/s)
**Problem**: `llama.cpp` is faster primarily because it uses **4-bit quantization** (Q4_K, Q4_0) by default.
- FP16 (Our Engine): 2 bytes per weight. Devstral = 44GB / token.
- INT4 (llama.cpp): 0.5 bytes per weight. Devstral = 11GB / token.
- **Physics**: 4x less data to move = 4x higher speed.

**Solution: Custom INT4/INT8 Kernels**
We need to write custom Metal kernels to dequantize weights on the fly.

**Implementation Strategy**:
1.  **Kernel Development**: Write `gemm_w4a16` (Weight INT4, Activation FP16) in Metal Shading Language (`.metal`).
2.  **Weight Packer**: Create a script to convert HuggingFace SafeTensors → vLLM Packed Int4 format.
3.  **Integration**: Replace `EngineGEMM` with `EngineGEMM_Int4`.
4.  **Result**: Throughput increases 3x-4x. Memory usage drops 4x.
5.  **Difficulty**: Very High (Requires low-level SIMD group optimizations in Metal).

---

## Summary comparison
| Feature | vLLM (Current) | llama.cpp | Proposed vLLM Next-Gen |
| :--- | :--- | :--- | :--- |
| **Runtime** | Python | C++ | Python + **Graph Capture** |
| **Precision** | FP16 (2 bytes) | INT4 (0.5 bytes) | **INT4** |
| **Overhead** | ~40ms/tok | ~0ms/tok | ~0ms/tok |
| **Throughput (B1)** | 10 tok/s | ~20 tok/s (CPU overhead? or FP16?) | **~18 tok/s (FP16)** / **~60 tok/s (INT4)** |

**Recommendation**:
Start with **Phase 1 (Graph Capture)**. It keeps the precision high (FP16) but maximizes the hardware efficiency. It is the "correct" engineering path for a vLLM-based engine.
Quantization (Phase 2) is a separate research project.
