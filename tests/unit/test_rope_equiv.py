import torch
import numpy as np
import os
import sys

# Add vllm-apple to path
sys.path.append(os.getcwd())

from vllm_apple.engine.ops.elementwise import EngineRoPE
from vllm_apple.engine.tensor import EngineDType, EngineTensor
# from vllm_apple.engine.core import EngineCore # Does not exist
# from vllm_apple.engine.step import EngineStepContext

def test_rope_equivalence():
    print("--- Testing RoPE Equivalence ---")
    
    # 1. Setup
    batch_size = 1
    seq_len = 16
    num_heads = 14
    head_size = 64
    rotary_dim = 64
    max_position = 2048
    base = 10000
    
    num_tokens = batch_size * seq_len
    hidden_size = num_heads * head_size
    
    dtype = torch.float16
    device = "cpu" # Reference on CPU
    
    # 2. Inputs
    # Random Q and K [NumTokens, NumHeads, HeadSize]
    # Flattened: [NumTokens, HiddenSize]
    q_torch = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    k_torch = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    
    # Positions
    positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1).flatten()
    
    # 3. PyTorch Reference RoPE
    def apply_rope_torch(x, cos, sin):
        # x: [bs, seq, heads, head_dim] (we have [tokens, heads, head_dim])
        # simple rotation
        # split into half
        head_dim = x.shape[-1]
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    # Precompute cos/sin
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_table = emb.cos().to(dtype=dtype)
    sin_table = emb.sin().to(dtype=dtype)
    
    # Select cos/sin for positions
    # [tokens, 1, head_dim]
    cos = cos_table[positions].unsqueeze(1)
    sin = sin_table[positions].unsqueeze(1)
    
    q_ref = apply_rope_torch(q_torch, cos, sin)
    k_ref = apply_rope_torch(k_torch, cos, sin)
    
    # 4. Engine RoPE
    # We need to setup EngineCore to get device/stream
    # But we can just use EngineRotaryEmbedding if we mock step_ctx
    # We need a dummy EngineCore or similar to init Metal
    
    # Initialize Context (minimal)
    from vllm_apple.engine.context import MetalEngineContext
    context = MetalEngineContext()
    
    rope_op = EngineRoPE(
        context=context,
        head_size=head_size,
        num_heads=num_heads,
        num_kv_heads=num_heads, # Qwen2-0.5B might be GQA but we test symmetric here
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        is_neox_style=True 
    )
    
    # Prepare Buffers
    # Input Buffer
    q_flat = q_torch.flatten(1).detach().numpy().astype(np.float16)
    k_flat = k_torch.flatten(1).detach().numpy().astype(np.float16)
    
    q_buf = context.create_buffer_from_bytes(q_flat.tobytes())
    k_buf = context.create_buffer_from_bytes(k_flat.tobytes())
    
    input_positions = context.create_buffer_from_bytes(positions.detach().numpy().astype(np.int32).tobytes())
    
    q_tensor = EngineTensor(q_buf, (num_tokens, num_heads, head_size), EngineDType.FLOAT16)
    k_tensor = EngineTensor(k_buf, (num_tokens, num_heads, head_size), EngineDType.FLOAT16) # Use symmetric heads for test
    
    # Context
    class MockStepContext:
        def __init__(self):
            import Metal
            self.command_queue = context.device.newCommandQueue()
            self._command_buffer = self.command_queue.commandBuffer()
            self.is_encoding = True
            self._encoder = None
            
        def get_compute_encoder(self):
            if self._encoder is None:
                self._encoder = self._command_buffer.computeCommandEncoder()
            return self._encoder
            
        def memory_barrier(self): pass
    
    step_ctx = MockStepContext()
    
    # Execute
    rope_op.encode(
        step_ctx,
        query=q_tensor,
        key=k_tensor,
        positions=input_positions, # EngineTensor wrapper?
        num_tokens=num_tokens
    )
    
    # End encoding manually since we are mocking the step loop
    if step_ctx._encoder:
        step_ctx._encoder.endEncoding()
    
    step_ctx._command_buffer.commit()
    step_ctx._command_buffer.waitUntilCompleted()
    
    # Read Back
    def read_buffer(buf, shape):
        size = np.prod(shape) * 2 # float16 = 2 bytes
        # Use as_buffer on the void* returned by contents()
        # This is the standard PyObjC way to get a memoryview from a void*
        try:
           raw = buf.contents().as_buffer(size)
        except AttributeError:
           # Fallback if as_buffer is missing (older PyObjC?)
           import ctypes
           # ctypes.cast(buf.contents(), ...)
           # But buf.contents() is python object.
           # Let's try getting integer address via objc.
           # Or just assume as_buffer works for now.
           raise
        arr = np.frombuffer(raw, dtype=np.float16, count=np.prod(shape)).reshape(shape)
        return torch.from_numpy(arr.copy())
        
    q_out = read_buffer(q_buf, (num_tokens, num_heads, head_size))
    k_out = read_buffer(k_buf, (num_tokens, num_heads, head_size))
    
    # Compare
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    print(f"Q Max Diff: {q_diff}")
    print(f"K Max Diff: {k_diff}")
    
    if q_diff > 1e-3 or k_diff > 1e-3:
        print("FAIL: RoPE Equivalence")
        # Print breakdown
        print("Reference sample:", q_ref[0,0,:5])
        print("Engine sample:   ", q_out[0,0,:5])
    else:
        print("PASS: RoPE Equivalence")

if __name__ == "__main__":
    test_rope_equivalence()
