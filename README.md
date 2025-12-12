# vLLM Apple Plugin - Metal V1.5

Native Metal backend for vLLM on Apple Silicon with custom PagedAttention kernels.

## Performance

Throughput on M3 Ultra with Qwen3-30B-A3B:

| Batch Size | tok/s | Scaling |
|------------|-------|---------|
| 1          | 6.7   | 1.00x   |
| 2          | 11.3  | 1.70x   |
| 4          | 16.5  | 2.47x   |
| 8          | 23.1  | 3.47x   |
| 16         | 32.8  | 4.92x   |

## Architecture

### Metal Kernels

- **MetalPagedAttentionV2**: Prefill path with batched KV write
- **MetalPagedAttentionFused**: Decode path with fused KV-write + attention
- **MetalKVCache**: Unified GPU/CPU memory via MTLBuffer

### Two-Phase Fused Decode

The fused kernel eliminates CPU-side KV update overhead:

1. `kv_write_decode`: Fast kernel writes new K/V to cache
2. `paged_attention_fused_h128`: Attention reads from populated cache

Both run in same GPU dispatch - no CPU sync between them.

## Installation

```bash
# Clone repository
git clone https://github.com/anthropics/vllm-apple.git
cd vllm-apple

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Usage

```python
from vllm import LLM, SamplingParams

# Plugin auto-registers via entrypoint
llm = LLM(
    model="Qwen/Qwen3-30B-A3B",
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.85,
)

outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(max_tokens=50, temperature=0.0),
)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_ATTENTION` | `1` | Enable Metal attention backend |
| `VLLM_METAL_FUSED_KV` | `1` | Enable fused KV-write + attention |

## Project Structure

```
vllm_apple/
    __init__.py              # Plugin entrypoint
    platform.py              # ApplePlatform implementation
    metal/
        __init__.py
        kv_cache.py          # MetalKVCache with MTLBuffer
        bridge/
            __init__.py
            metal_runtime.py
            metal_paged_attention_v2.py     # V2 prefill kernel
            metal_paged_attention_fused.py  # Fused decode kernel
        kernels/
            paged_attention_v2.metal        # V2 shader
            paged_attention_fused.metal     # Fused shader
    ops/
        apple_fused_moe.py   # MoE operations
        metal/
            moe_kernel_v2.metal
            moe_metal.py
    v1/
        attention/
            backends/
                metal_attn.py    # MetalAttentionBackend
        worker/
            apple_worker.py
            apple_model_runner.py
            apple_input_batch.py
```

## Tests

```bash
# Run Metal attention tests
python -m pytest tests/test_metal_attention.py -v

# Run all tests
python -m pytest tests/ -v
```

## License

Apache-2.0
