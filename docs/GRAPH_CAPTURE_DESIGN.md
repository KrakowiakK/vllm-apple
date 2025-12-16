# Metal Graph Capture Design (Phase 1)

**Objective**: Eliminate CPU dispatch overhead to reach hardware decoding limit (~18 tok/s FP16).
**Current Overhead**: ~40ms/token (Python layer loop).
**Target Overhead**: <5ms/token.

## 1. Architectural Analysis

The current `runner.py` executes a dynamic Python loop over 40 layers. For each layer, it performs ~15-20 individual encode calls (GEMM, Bias, RoPE, Norm, Activation, Residual).
Each call involves:
1.  Python function call overhead.
2.  Argument validation.
3.  Bridge to Obj-C (`pyobjc`).
4.  Metal API encoding cost.

This "Latency Death by 1000 Cuts" prevents the GPU from being fed fast enough at Batch 1.

## 2. Solution: Hybrid Captured Execution

We cannot put the *entire* model into a single `MPSGraph` because `MPSGraph` does not support our optimized PagedAttention or custom quantization kernels natively (yet).
We cannot use `MTLIndirectCommandBuffer` (ICB) for everything because MPS (GEMM) does not support ICB encoding.

**Proposed Architecture: The "Three-Node" Layer**
We will collapse the ~20 ops per layer into **3 compiled nodes**:

1.  **Node A (MPSGraph)**: `Input -> RMSNorm -> QKV Proj -> RoPE`.
    *   *Inputs*: Hidden States, QKV Weights, RoPE Cos/Sin.
    *   *Outputs*: Q, K, V (ready for Attention).
    *   *Note*: RoPE can be implemented in MPSGraph or kept separate. Fusing QKV+RoPE is ideal.

2.  **Node B (Custom Metal)**: `PagedAttention`.
    *   *Inputs*: Q, K, V, KV Cache, Block Tables.
    *   *Outputs*: Attention Output.
    *   *Mechanism*: Direct `computeCommandEncoder` (fast).

3.  **Node C (MPSGraph)**: `Attn Output -> O Proj -> Residual -> RMSNorm -> MLP (GateUp -> Act -> Down) -> Residual`.
    *   *Inputs*: Attention Output, Residual Input, MLP Weights.
    *   *Outputs*: Next Layer Hidden State.
    *   *Optimization*: Fuses ALL dense math into one compiled graph.

## 3. Execution Flow (The "New Runner")

Instead of encoding ops dynamically, the `GraphRunner`:
1.  **Capture Phase (Warmup)**:
    *   Builds `MPSGraph` for Node A and Node C.
    *   Compiles them to `MPSGraphExecutable`.
2.  **Decode Phase**:
    *   Gets a single `MTLCommandBuffer`.
    *   **Loop (C++ or Specialized Python)**:
        *   `NodeA.encode(cmdBuf)`
        *   `NodeB.encode(cmdBuf)`
        *   `NodeC.encode(cmdBuf)`
    *   Commits Buffer.

**Why this wins**:
*   Reduces Python calls from ~600/token to ~120/token.
*   `MPSGraph` pre-compiles the command generation, making `encode()` nearly instant.
*   Enables Metal driver optimizations (instruction fusion).

## 4. Implementation Details

### A. The Graph Cache
```python
class LayerGraphCache:
    def __init__(self):
        self.qkv_graphs = {} # Key: (batch_size)
        self.mlp_graphs = {}
    
    def get_qkv_graph(self, batch_size):
        if batch_size not in self.qkv_graphs:
            self.qkv_graphs[batch_size] = build_qkv_mps_graph(batch_size)
        return self.qkv_graphs[batch_size]
```

### B. Persistent Objects (Metal)
To make this work, we must stop creating temp tensors. We need **Static Buffers**.
*   `_static_hidden_states`: [MaxBatch, Dim] (Ping-Pong buffers A/B).
*   `_static_qkv`: [MaxBatch, Heads, HeadDim].
*   `_static_intermediate`: [MaxBatch, InterDim].

### C. Pseudocode

```python
def capture_model_graph(self):
    # One-time setup of MPSGraph executables
    self.graph_qkv = compile_qkv_graph()
    self.graph_mlp = compile_mlp_graph()

def execute_decode_step(self, tokens):
    cmd_buf = self.command_queue.commandBuffer()
    
    # 1. Update Graph Arguments (Pointers to KV Cache, etc)
    # This is lightweight pointer swapping
    
    # 2. Encode Loop
    for layer in self.layers:
        # Pre-compiled encode is fast
        self.graph_qkv.encode(cmd_buf, inputs=..., outputs=...)
        self.attention.encode(cmd_buf, ...)
        self.graph_mlp.encode(cmd_buf, inputs=..., outputs=...)
        
    # 3. Submit
    cmd_buf.commit()
```

## 5. Success Criteria
*   **Zero Regressions**: Equivalence tests must pass.
*   **Throughput**: Decode Batch 1 should reach ~16-18 tok/s.
*   **Chunked Prefill**: Must still handle dynamic sequence lengths (might need dynamic graph or multiple graph buckets).

## 6. Next Step
Begin implementation of `vllm_apple/engine/graph_runner.py` implementing the logic above.
