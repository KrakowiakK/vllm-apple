# vLLM Apple Metal Engine (v2.6-stable)

> [!WARNING]
> **VIBE CODING PROJECT**
> This is an experimental, "vibe coding" project built for educational and research purposes.
> **The author takes NO RESPONSIBILITY** for any issues, crashes, incorrect outputs, or hardware damage.
> Use at your own risk. We are moving fast and breaking tensors.

## Overview
This repository hosts the **vLLM Apple Metal Engine**, a custom inference backend designed to bring high-performance, production-grade serving capabilities to Apple Silicon.

Unlike `llama.cpp` which targets edge inference and broad compatibility, this engine targets **high-throughput batching** and **server-grade architecture** (PagedAttention, Continuous Batching) specifically for Mac Studio / Ultra class hardware.

**Testing Environment**: developed and verified on **Apple M3 Ultra (128GB)**.

## Project Status
**Version**: v2.6-stable (December 2025)
**State**: Pure Metal/MPS implementation of the core Transformer block.

### ✅ What Works (Implemented & Verified)
*   **Core Architecture**:
    *   Fully "Encode-Only" architecture (Zero CPU-GPU synchronization overhead during step execution).
    *   **PagedAttention** on Metal (Custom kernel, equivalent to CUDA implementation).
    *   **Continuous Batching** (Scheduler integration verified).
    *   **Chunked Prefill** (Splitting large prompts to maintain inter-token latency).
*   **Kernels**:
    *   **RoPE**: Rotary Embeddings (Neox-style, verified bit-exact vs PyTorch).
    *   **RMSNorm**: Fused Metal kernel.
    *   **SwiGLU**: Fused Silu + Multiply for MLP.
    *   **GEMM**: Hybrid MPS/Metal approach.
*   **Precision**:
    *   **FP16**: Fully verified. Accumulation tolerance ~1e-2 vs FP32 Reference.
    *   **Determinism**: 100% deterministic output for same seed.

### 🚧 Roadmap / Missing Features
*   **Quantization**:
    *   INT4 / INT8 kernels are **NOT** yet implemented/verified. Currently FP16 only.
    *   (Critical for running models > 24B params on standard Macs).
*   **Mixture of Experts (MoE)**:
    *   Routing logic is ready, but full kernels for expert dispatch are pending integration.
*   **Advanced Sampling**:
    *   Use basic sampling currently; advanced beam search/speculative decoding not optimized for Metal yet.
*   **Multi-GPU**:
    *   Single-device only. No distributed support for multiple Mac Studios.

## Why use this over llama.cpp?
*   **Architecture**: If you want to study how `vLLM` works under the hood or build server-side agents on Mac.
*   **Python Native**: Tightly integrated with PyTorch and Python ecosystem (unlike C++ focused llama.cpp).
*   **Throughput**: Designed for batch sizes > 1 (e.g., serving multiple agents simultaneously).

## Installation

```bash
pip install -e .
```

## Running Inference (Mac Studio M3 Ultra Recommended)

Set environment variables to engage the Metal backend:

```bash
export VLLM_PLATFORM=apple
export VLLM_METAL_ATTENTION=1
export VLLM_APPLE_USE_ENGINE=1

# Run OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-0.5B-Instruct --gpu-memory-utilization 0.9
```

## Documentation
*   [Metal Bible (internals)](docs/METAL_BIBLE.md): Read this if you want to understand the "Encode-Only" philosophy.
*   [MoE Readiness](docs/MOE_READINESS.md): Status of MoE support.

---
*Maintained by: @antigravity*
