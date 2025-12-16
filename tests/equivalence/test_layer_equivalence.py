
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Enable debug mode BEFORE imports that use it
os.environ["VLLM_PREFILL_EQ_DEBUG"] = "1"
# Ensure we use the Apple backend logic
os.environ["VLLM_PLATFORM"] = "apple" 
# Fix serialization issue
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
# Use Optimized Metal Attention
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_APPLE_USE_ENGINE"] = "1" # CRITICAL: Enable Metal Engine
# Use Disk Checkpointing logic
DISK_PATH = "/tmp/vllm_layer_test"
os.environ["VLLM_DEBUG_DISK_PATH"] = DISK_PATH

from vllm_apple.debug.pytorch_reference_hooks import attach_reference_hooks
from vllm_apple.debug.prefill_checkpoint import compare_checkpoints, print_comparison_report, reset_stores, load_from_disk
import vllm_apple.engine.runner
print(f"DEBUG: vllm_apple.engine.runner path: {vllm_apple.engine.runner.__file__}")

def run_layer_test(model_name="Qwen/Qwen2-0.5B-Instruct", prompt_len=32):
    print(f"--- Running Layer Equivalence Test (Length {prompt_len}) ---")
    reset_stores()
    
    # Clean disk path
    import shutil
    if os.path.exists(DISK_PATH):
        shutil.rmtree(DISK_PATH)
    os.makedirs(DISK_PATH, exist_ok=True)
    
    # 1. Setup Inputs
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = "A" * prompt_len # Simple prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    if len(tokens) > prompt_len: tokens = tokens[:prompt_len]
    print(f"Prompt Tokens: {len(tokens)}")
    
    # 2. Run HF (PyTorch) - Run on CPU to avoid Metal conflict/OOM
    print("Loading HF Model (CPU)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="float32", # Use float32 for reference
        trust_remote_code=True,
    )
    hf_model.to("cpu")
    hf_model.eval()
    
    print("Attaching Reference Hooks...")
    # Last pos is len-1
    attach_reference_hooks(hf_model, num_layers=hf_model.config.num_hidden_layers, last_pos=len(tokens)-1, architecture="qwen")
    
    print("Running HF Forward...")
    input_ids = torch.tensor([tokens], device="cpu")
    with torch.no_grad():
        hf_model(input_ids)
    print("HF Run Complete. Checkpoints captured.")
    
    # Cleanup HF to save RAM (though 0.5B is small)
    del hf_model
    import gc; gc.collect()
    
    # 3. Run vLLM Engine
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype="float16",
        gpu_memory_utilization=0.6,
        enforce_eager=True, # Ensure no cuda graphs etc
        trust_remote_code=True,
        max_model_len=2048
    )
    
    print("Running vLLM Generate...")
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate(prompts=[{"prompt_token_ids": tokens}], sampling_params=sampling_params)
    print("vLLM Run Complete. Checkpoints captured.")
    
    # Load checkpoints from disk (from subprocess)
    print(f"Loading checkpoints from {DISK_PATH}...")
    load_from_disk(DISK_PATH, "engine")

    # 4. Compare
    print("Comparing Checkpoints...")
    results, first_div = compare_checkpoints()
    
    print(print_comparison_report(results, first_div))
    
    if first_div:
        print("\nFAIL: Divergence detected.")
        exit(1)
    else:
        print("\nSUCCESS: No divergence!")
        exit(0)

if __name__ == "__main__":
    run_layer_test()
