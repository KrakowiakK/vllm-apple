import torch
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2-0.5B-Instruct"  # Or Qwen/Qwen2-0.5B
print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

print("\n--- Listing ALL Bias Parameters ---")
bias_keys = []
for name, param in model.named_parameters():
    if "bias" in name:
        print(f"{name}: {param.shape}")
        bias_keys.append(name)

if not bias_keys:
    print("NO BIASES FOUND!")
else:
    print(f"\nFound {len(bias_keys)} bias tensors.")
