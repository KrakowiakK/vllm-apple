
from transformers import AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2-0.5B-Instruct"
try:
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True)
    print("Model loaded.")
    
    print("\nChecking for QKV biases in Layer 0:")
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    
    print(f"q_proj.bias: {attn.q_proj.bias is not None}")
    if attn.q_proj.bias is not None:
        print(f"q_proj.bias shape: {attn.q_proj.bias.shape}")
        print(f"q_proj.bias max: {attn.q_proj.bias.max()}")
        
    print(f"k_proj.bias: {attn.k_proj.bias is not None}")
    if attn.k_proj.bias is not None:
        print(f"k_proj.bias shape: {attn.k_proj.bias.shape}")

    print(f"v_proj.bias: {attn.v_proj.bias is not None}")
    if attn.v_proj.bias is not None:
        print(f"v_proj.bias shape: {attn.v_proj.bias.shape}")
        
    print("\nChecking State Dict Keys:")
    sd = model.state_dict()
    keys = list(sd.keys())
    q_bias_keys = [k for k in keys if "q_proj.bias" in k]
    print(f"Found {len(q_bias_keys)} q_proj.bias keys.")
    if q_bias_keys:
        print(f"Example: {q_bias_keys[0]}")

except Exception as e:
    print(f"Error: {e}")
