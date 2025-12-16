# FINAL VALIDATION REPORT: Devstral-Small (vLLM Metal v2.6-stable)

**Date**: 2025-12-16
**Model**: `mistralai/Devstral-Small-2505` (22B Parameters)
**Hardware**: Apple Mac Studio (M3 Ultra, 128GB RAM)
**Engine Version**: v2.6-stable (Commit `edbfddb`)

## 1. Executive Summary
The vLLM Apple Metal Engine successfully runs the 22B Devstral model in full FP16 precision. The engine is numerically stable and deterministic.
Performance is currently limited by the Python-based dispatcher (CPU overhead), reaching ~57% of the hardware's theoretical limit for FP16 inference.

## 2. Measured Performance

### Memory Footprint
*   **RSS Usage**: **31.2 GB**
*   **Capacity**: Comfortably fits on 64GB and 128GB Unified Memory systems.

### Prefill Latency (Prompt Processing)
| Batch Size | Total Tokens | Latency (ms) | Throughput (tok/s) |
| :--- | :--- | :--- | :--- |
| **1** | 64 | 906.3 | **70.6** |
| **2** | 128 | 1236.0 | **103.6** |

*Note: Prefill throughput scales well with batch size, indicating efficient Metal kernels.*

### Decode Latency (Token Generation)
| Batch Size | Latency per Token (ms) | Throughput (tok/s) |
| :--- | :--- | :--- |
| **1** | 96.97 | **10.3** |

*Analysis*:
*   Theoretical Limit (Bandwidth Bound): ~18 tok/s (44GB @ 800GB/s).
*   Competitors (llama.cpp/MLX): ~15-18 tok/s.
*   **Bottleck**: CPU Overhead. The Python loop consumes ~40ms per token dispatching 40 layers.
*   **Gap to Close**: ~7-8 tok/s.

## 3. Correctness Verification
*   **Weight Loading**: Successful (Separate QKV layout handled correctly).
*   **Output Semantics**: Coherent text generation verified.
*   **Determinism**: 100% reproducible outputs with fixed seeds.

## 4. Next Steps (Performance Optimization)
To bridge the gap between 10.3 tok/s and the hardware limit of ~18 tok/s, we must eliminate the Python dispatch overhead.
See `PERFORMANCE_OPTIMIZATION_PLAN.md` for the detailed roadmap (Graph Capture).
