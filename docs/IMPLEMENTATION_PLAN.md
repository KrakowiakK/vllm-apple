# vLLM-Apple Implementation Plan (Updated 2025-12-11)

## Executive Summary

Complete standalone vLLM backend for Apple Silicon (MPS), following the architecture patterns
from vllm-gaudi and vllm-ascend plugins. The goal is full compatibility with vLLM 0.12.x V1 engine
with custom Metal kernels for 50+ TPS performance.

**Current State (7.8 TPS) vs Target (50+ TPS)**

| Implementation | TPS | Status |
|---------------|-----|--------|
| PyTorch MPS fallback | 7.8 | Current |
| llama.cpp Metal | 15-20 | Reference |
| **vllm-apple target** | **50+** | Goal |

**CRITICAL DISCOVERY**: Metal kernels ALREADY EXIST in vllm-repo but are NOT connected to Python!
- `vllm/attention/backends/metal/flash_attn.metal` - 534 lines of Flash Attention
- `vllm/model_executor/layers/fused_moe/metal/moe_kernels.metal` - 408 lines of MoE kernels
- Missing: C++ extension (.mm) to bridge Metal → Python

**Reference Plugins**:
- vllm-gaudi: 15,311 lines, HPUWorker + HPUModelRunner (~5,200 lines combined)
- vllm-ascend: 46,267 lines, NPUWorker + NPUModelRunner (~5,000 lines combined)

---

## 1. Target Directory Structure

```
vllm-apple/
├── setup.py                              # Entry points configuration
├── pyproject.toml                        # Build configuration
├── vllm_apple/
│   ├── __init__.py                       # Plugin registration (register, register_ops, register_models)
│   ├── platform.py                       # ApplePlatform class (~300 lines)
│   ├── envs.py                           # Environment variables
│   ├── utils.py                          # Common utilities
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── apple_worker.py               # AppleWorker(WorkerBase) (~400 lines)
│   │   └── apple_model_runner.py         # AppleModelRunner (~2,500 lines)
│   │
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── worker/
│   │   │   ├── __init__.py
│   │   │   ├── apple_worker.py           # AppleWorkerV1(WorkerBase) (~400 lines)
│   │   │   ├── apple_model_runner.py     # AppleModelRunnerV1 (~3,000 lines)
│   │   │   └── apple_input_batch.py      # AppleInputBatch (~800 lines)
│   │   └── attention/
│   │       ├── __init__.py
│   │       └── backends/
│   │           ├── __init__.py
│   │           └── apple_attn.py         # AppleAttentionBackendV1 (~600 lines)
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   └── apple_attn.py             # AppleAttentionBackend (~500 lines)
│   │   └── ops/
│   │       ├── __init__.py
│   │       └── apple_paged_attn.py       # ApplePagedAttention (~400 lines)
│   │
│   ├── ops/
│   │   ├── __init__.py                   # Custom ops registration
│   │   ├── apple_fused_moe.py            # Apple MoE ops (existing: ~200 lines)
│   │   ├── apple_layernorm.py            # LayerNorm ops (~100 lines)
│   │   └── apple_rotary_embedding.py     # RoPE ops (~150 lines)
│   │
│   ├── extension/
│   │   ├── __init__.py
│   │   ├── metal_ops.py                  # Metal kernel loader (existing)
│   │   └── kernels/                      # Metal shader files (.metal)
│   │       ├── moe_kernel.metal
│   │       └── attention_kernel.metal
│   │
│   ├── sample/
│   │   ├── __init__.py
│   │   └── apple_sampler.py              # Apple-optimized sampler (~200 lines)
│   │
│   └── distributed/
│       ├── __init__.py
│       └── apple_communicator.py         # CPU/Gloo communicator wrapper (~100 lines)

Estimated Total: ~9,000-10,000 lines of code
```

---

## 2. Implementation Phases

### Phase 1: Core Infrastructure (Priority: Critical)

#### 1.1 Platform Class Enhancement (`platform.py`)

**Current Status**: Partially implemented, needs V1 engine support

**Required Changes**:
```python
class ApplePlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    simple_compile_backend: str = "eager"
    ray_device_key: str = "MPS"
    device_control_env_var: str = "APPLE_VISIBLE_DEVICES"

    # NEW: V1 support
    @classmethod
    def supports_v1(cls, model_config) -> bool:
        return True

    # NEW: Worker class selection
    @classmethod
    def check_and_update_config(cls, vllm_config):
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = \
                "vllm_apple.v1.worker.apple_worker.AppleWorker"
        # ... rest of config validation

    # NEW: Memory management
    @classmethod
    def mem_get_info(cls, device=None) -> tuple[int, int]:
        """Return (free_memory, total_memory) for unified memory."""
        import psutil
        mem = psutil.virtual_memory()
        # Apple uses unified memory - return ~50% for GPU use
        total = mem.total // 2
        available = mem.available // 2
        return (available, total)

    @classmethod
    def synchronize(cls):
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

    @classmethod
    def empty_cache(cls):
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
```

**Key Methods to Implement**:
- `supports_v1()` - Return True
- `mem_get_info()` - Unified memory reporting
- `synchronize()` - MPS synchronization
- `empty_cache()` - Memory cleanup
- `get_attn_backend_cls()` - Return Apple attention backend
- `inference_mode()` - Return torch.no_grad()

---

#### 1.2 AppleWorker (`v1/worker/apple_worker.py`)

**Pattern**: Follow HPUWorker from vllm-gaudi

**Key Design**:
- **MUST** inherit from `WorkerBase`, NOT from `GPUWorker` (avoids Triton)
- Store all config manually in `__init__`
- Create `AppleModelRunner` in `init_device()`

```python
from vllm.v1.worker.worker_base import WorkerBase

class AppleWorker(WorkerBase):
    """Apple Silicon worker for vLLM V1 engine.

    Inherits from WorkerBase to avoid Triton dependencies in GPUWorker.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # Store configs manually (like HPUWorker)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Cache dtype handling
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        self.device = None
        self.model_runner = None

    def init_device(self):
        """Initialize MPS device and model runner."""
        self.device = torch.device("mps")

        # Initialize distributed if needed (single device for Apple)
        from vllm.distributed import init_distributed_environment
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=self.distributed_init_method,
            local_rank=0,
            backend="gloo"
        )

        # Set random seed
        from vllm.model_executor import set_random_seed
        set_random_seed(self.model_config.seed)

        # Create model runner
        from vllm_apple.v1.worker.apple_model_runner import AppleModelRunner
        self.model_runner = AppleModelRunner(
            vllm_config=self.vllm_config,
            is_driver_worker=self.is_driver_worker
        )

    def load_model(self) -> None:
        self.model_runner.load_model()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profile memory and return available for KV cache."""
        import psutil

        # Run profile pass
        self.model_runner.profile_run()
        torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None

        # Calculate available memory
        mem = psutil.virtual_memory()
        available = mem.available

        # Reserve 50% for model, use rest for KV cache
        gpu_memory_utilization = self.cache_config.gpu_memory_utilization
        cache_size = int(available * gpu_memory_utilization * 0.5)

        return cache_size

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate KV cache."""
        self.model_runner.initialize_kv_cache(kv_cache_config)
        self.compile_or_warm_up_model()

    def compile_or_warm_up_model(self) -> None:
        """Warm up model (no compilation on MPS)."""
        if not self.model_config.enforce_eager:
            self.model_runner.warmup_model()
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)
```

---

#### 1.3 AppleModelRunner (`v1/worker/apple_model_runner.py`)

**Pattern**: Follow HPUModelRunner from vllm-gaudi (4,901 lines)

**Key Design**:
- **DO NOT** inherit from `GPUModelRunner` (has Triton imports)
- Implement interface from scratch
- Use PyTorch SDPA for attention (no Flash Attention)

**Core Structure**:
```python
class AppleModelRunner:
    """Model runner for Apple Silicon.

    Implements the model runner interface without inheriting from
    GPUModelRunner to avoid Triton dependencies.
    """

    def __init__(self, vllm_config: VllmConfig, is_driver_worker: bool = False):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.is_driver_worker = is_driver_worker

        self.device = torch.device("mps")
        self.model = None
        self.kv_caches: list[torch.Tensor] = []

        # Input batch for managing requests
        self.input_batch = None

        # Sampler
        self.sampler = None

        # Attention backend
        self.attn_backend = None

    def load_model(self) -> None:
        """Load model onto MPS device."""
        from vllm.model_executor.model_loader import get_model
        self.model = get_model(vllm_config=self.vllm_config)

    def get_model(self) -> nn.Module:
        return self.model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Return KV cache specification for each layer."""
        # Iterate through model layers and collect specs
        # Similar to HPUModelRunner implementation
        pass

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate KV cache tensors."""
        # Allocate cache tensors based on config
        pass

    def profile_run(self) -> None:
        """Run dummy forward pass to profile memory."""
        pass

    def warmup_model(self) -> None:
        """Warm up model with dummy inputs."""
        pass

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        """Execute single forward pass."""
        # 1. Prepare inputs from scheduler_output
        # 2. Build attention metadata
        # 3. Run model forward
        # 4. Sample tokens
        # 5. Return ModelRunnerOutput
        pass
```

**Key Methods** (~3,000 lines total):
1. `__init__()` - Initialize all components
2. `load_model()` - Load and prepare model
3. `get_kv_cache_spec()` - Return cache specs
4. `initialize_kv_cache()` - Allocate caches
5. `profile_run()` - Memory profiling
6. `warmup_model()` - Model warmup
7. `execute_model()` - Main execution loop
8. `_prepare_inputs()` - Build model inputs from scheduler output
9. `_build_attn_metadata()` - Create attention metadata
10. `_forward()` - Run model forward pass
11. `_sample()` - Sample next tokens

---

### Phase 2: Attention Backend (Priority: High)

#### 2.1 AppleAttentionBackend (`v1/attention/backends/apple_attn.py`)

**Pattern**: Follow HPUAttentionBackendV1

```python
from vllm.attention.backends.abstract import (
    AttentionBackend, AttentionImpl, AttentionMetadata
)
from vllm.attention.backends.registry import register_backend, AttentionBackendEnum

@register_backend(AttentionBackendEnum.CUSTOM, "APPLE_ATTN")
class AppleAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "APPLE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return AppleAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AppleAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        # 2 = key + value
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(src_kv_cache, dst_kv_cache, src_to_dsts) -> None:
        """Swap KV cache blocks between CPU and MPS."""
        # Implementation using index_copy_
        pass

    @staticmethod
    def copy_blocks(kv_caches, src_to_dsts) -> None:
        """Copy blocks within KV cache."""
        # Implementation using index_copy_
        pass


@dataclass
class AppleAttentionMetadata(AttentionMetadata):
    """Metadata for Apple attention backend."""
    is_prompt: bool
    block_size: int
    slot_mapping: torch.Tensor
    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    attn_bias: Optional[torch.Tensor] = None
    block_list: Optional[torch.Tensor] = None

    @classmethod
    def make_prefill_metadata(cls, ...):
        """Create metadata for prefill phase."""
        pass

    @classmethod
    def make_decode_metadata(cls, ...):
        """Create metadata for decode phase."""
        pass


class AppleAttentionImpl(AttentionImpl, nn.Module):
    """Apple attention implementation using PyTorch SDPA."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AppleAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using PyTorch SDPA."""
        if attn_metadata.is_prompt:
            return self._forward_prefill(query, key, value, kv_cache, attn_metadata)
        else:
            return self._forward_decode(query, key, value, kv_cache, attn_metadata)

    def _forward_prefill(self, q, k, v, kv_cache, metadata):
        """Prefill using scaled_dot_product_attention."""
        # Build causal mask
        # Use F.scaled_dot_product_attention
        # Store K, V in cache
        pass

    def _forward_decode(self, q, k, v, kv_cache, metadata):
        """Decode using paged attention."""
        # Gather K, V from cache using block indices
        # Use F.scaled_dot_product_attention
        pass
```

---

### Phase 3: Input Batch Management (Priority: High)

#### 3.1 AppleInputBatch (`v1/worker/apple_input_batch.py`)

**Pattern**: Follow HPUInputBatch from vllm-gaudi

```python
@dataclass
class CachedRequestState:
    """State for a cached request."""
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: list[list[int]]
    num_computed_tokens: int
    output_token_ids: list[int]


class AppleInputBatch:
    """Manages batched inputs for Apple model runner."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.device = device

        # Request tracking
        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # Token storage
        self.token_ids_cpu = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu", dtype=torch.int32
        )

        # Sampling parameters
        self.temperature = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)

        # Block table for KV cache
        self.block_table = None  # Initialize based on block_sizes

    def add_request(self, request: CachedRequestState) -> int:
        """Add a request to the batch."""
        pass

    def remove_request(self, req_id: str) -> None:
        """Remove a request from the batch."""
        pass

    def get_model_inputs(self) -> dict[str, torch.Tensor]:
        """Get inputs for model forward pass."""
        pass

    def commit_step(self, sampled_token_ids: torch.Tensor) -> None:
        """Commit sampled tokens and update state."""
        pass
```

---

### Phase 4: Custom Operations (Priority: Medium)

#### 4.1 Enhanced MoE Operations (`ops/apple_fused_moe.py`)

**Already Implemented**: Basic Metal MoE kernel exists

**Enhancements Needed**:
- Integration with vLLM op registration system
- Support for different expert configurations
- FP16/BF16 support

```python
from vllm.model_executor.layers.fused_moe import register_fused_moe_op

@register_fused_moe_op("apple", "AppleFusedMoE")
class AppleFusedMoE:
    """Apple-optimized Mixture of Experts."""

    @staticmethod
    def apply(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Use Metal kernel if available, else PyTorch fallback
        pass
```

#### 4.2 LayerNorm Operations (`ops/apple_layernorm.py`)

```python
from vllm.model_executor.layers.layernorm import register_layernorm_op

@register_layernorm_op("apple")
def apple_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apple-optimized RMS normalization."""
    # Use Metal or PyTorch implementation
    variance = input.pow(2).mean(-1, keepdim=True)
    hidden_states = input * torch.rsqrt(variance + epsilon)
    return weight * hidden_states
```

---

### Phase 5: Entry Points & Registration (Priority: High)

#### 5.1 Plugin Entry Point (`__init__.py`)

```python
from typing import Optional

__version__ = "0.1.0"


def register() -> Optional[str]:
    """Register Apple platform with vLLM."""
    import torch
    if not torch.backends.mps.is_available():
        return None
    return "vllm_apple.platform.ApplePlatform"


def register_ops():
    """Register Apple-optimized operations."""
    from vllm_apple.ops import apple_fused_moe  # noqa: F401
    from vllm_apple.ops import apple_layernorm  # noqa: F401


def register_models():
    """Register Apple-optimized model implementations."""
    pass


def pre_register_and_update():
    """Platform patches applied before workers start."""
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

#### 5.2 Setup.py Entry Points

```python
setup(
    name="vllm-apple",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vllm>=0.8.0",
        "torch>=2.0.0",
        "psutil",
    ],
    entry_points={
        "vllm.platform_plugins": [
            "apple = vllm_apple:register"
        ],
        "vllm.general_plugins": [
            "apple_ops = vllm_apple:register_ops",
        ],
    },
)
```

---

## 3. Key Implementation Challenges & Solutions

### Challenge 1: Triton Dependency in V1 Engine

**Problem**: vLLM V1's `GPUModelRunner` imports Triton through various paths

**Solution**:
- Custom `AppleModelRunner` that doesn't inherit from `GPUModelRunner`
- Implement all required methods from scratch
- Use PyTorch SDPA instead of Flash Attention

### Challenge 2: CUDA-Specific Memory Management

**Problem**: vLLM assumes CUDA memory model (separate GPU/CPU memory)

**Solution**:
- Apple uses unified memory - implement custom `mem_get_info()`
- Use psutil for memory tracking
- Implement custom memory profiling in `determine_available_memory()`

### Challenge 3: Paged Attention Without Custom Kernels

**Problem**: vLLM's paged attention relies on Triton/CUDA kernels

**Solution**:
- Implement paged attention using PyTorch operations
- Use `index_select` and `scatter_` for cache operations
- Optional: Metal kernels for performance

### Challenge 4: Distributed Communication

**Problem**: Apple only supports single GPU

**Solution**:
- Force `world_size=1`
- Use Gloo backend for any distributed ops
- Disable tensor parallelism

---

## 4. Testing Strategy

### Phase 1 Tests: Basic Functionality
```python
# test_basic.py
def test_platform_registration():
    """Test that Apple platform registers correctly."""
    from vllm.platforms import current_platform
    assert current_platform.device_name == "mps"

def test_worker_init():
    """Test worker initialization."""
    from vllm_apple.v1.worker.apple_worker import AppleWorker
    # Create minimal config and init worker
    pass

def test_model_loading():
    """Test model loads onto MPS."""
    # Load small model (facebook/opt-125m)
    pass
```

### Phase 2 Tests: Inference
```python
def test_single_prompt():
    """Test single prompt generation."""
    from vllm import LLM
    llm = LLM(model="facebook/opt-125m")
    output = llm.generate("Hello, world!")
    assert len(output[0].outputs[0].text) > 0

def test_batch_inference():
    """Test batched generation."""
    pass

def test_moe_model():
    """Test MoE model (Qwen3-30B-A3B)."""
    pass
```

### Phase 3 Tests: Performance
```python
def test_throughput():
    """Measure tokens per second."""
    pass

def test_memory_usage():
    """Verify memory stays within bounds."""
    pass
```

---

## 5. Implementation Order

1. **Week 1**: Core Infrastructure
   - [ ] Update `platform.py` with V1 support
   - [ ] Implement `AppleWorker(WorkerBase)`
   - [ ] Create skeleton `AppleModelRunner`

2. **Week 2**: Model Runner
   - [ ] Implement `load_model()`, `get_model()`
   - [ ] Implement `get_kv_cache_spec()`, `initialize_kv_cache()`
   - [ ] Implement `execute_model()` basic flow

3. **Week 3**: Attention Backend
   - [ ] Implement `AppleAttentionBackend`
   - [ ] Implement `AppleAttentionImpl` with SDPA
   - [ ] Implement paged attention for decode

4. **Week 4**: Input Batch & Sampling
   - [ ] Implement `AppleInputBatch`
   - [ ] Implement sampling logic
   - [ ] Connect all components

5. **Week 5**: Testing & Optimization
   - [ ] Basic inference tests
   - [ ] MoE model tests
   - [ ] Performance optimization

---

## 6. Files to Create/Modify

### New Files:
1. `vllm_apple/v1/worker/apple_worker.py` (~400 lines)
2. `vllm_apple/v1/worker/apple_model_runner.py` (~3,000 lines)
3. `vllm_apple/v1/worker/apple_input_batch.py` (~800 lines)
4. `vllm_apple/v1/attention/backends/apple_attn.py` (~600 lines)
5. `vllm_apple/ops/apple_layernorm.py` (~100 lines)
6. `vllm_apple/sample/apple_sampler.py` (~200 lines)
7. `vllm_apple/envs.py` (~50 lines)

### Modify Files:
1. `vllm_apple/platform.py` - Add V1 support methods
2. `vllm_apple/__init__.py` - Add register_ops, register_models
3. `setup.py` - Add entry points

### Total Estimated: ~9,000 lines of new code

---

## 7. Reference Code Locations

### vllm-gaudi (Primary Reference):
- Worker: `vllm_gaudi/v1/worker/hpu_worker.py` (348 lines)
- ModelRunner: `vllm_gaudi/v1/worker/hpu_model_runner.py` (4,901 lines)
- InputBatch: `vllm_gaudi/v1/worker/hpu_input_batch.py` (734 lines)
- Attention: `vllm_gaudi/v1/attention/backends/hpu_attn.py` (113 lines)
- Platform: `vllm_gaudi/platform.py` (270 lines)

### vllm-ascend (Secondary Reference):
- Worker: `vllm_ascend/worker/worker_v1.py`
- ModelRunner: `vllm_ascend/worker/model_runner_v1.py` (4,692 lines)
- Attention: `vllm_ascend/attention/attention_v1.py`

### vLLM Core (Interface Reference):
- WorkerBase: `vllm/v1/worker/worker_base.py`
- AttentionBackend: `vllm/attention/backends/abstract.py`
- ModelRunnerOutput: `vllm/v1/outputs.py`
