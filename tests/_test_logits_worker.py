
import os
import sys
import argparse
import torch
import glob
import time

def run_hf(model_name, input_ids):
    print(f"Loading HF model {model_name}...")
    from transformers import AutoModelForCausalLM
    
    # tokenizer not needed for model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        # device_map="cpu" # Removed to avoid accelerate dependency
    )
    # Check for MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running HF on {device}")
    model.to(device)
    
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        # logits shape: [batch, seq_len, vocab]
        logits = outputs.logits
        # Last token logits
        last_token_logits = logits[0, -1, :]
        
    return last_token_logits.cpu()

def run_vllm(model_name, tokens_list, chunk_size=None):
    from vllm import LLM, SamplingParams
    
    # Configure env
    os.environ["VLLM_DUMP_LOGITS"] = "1"
    if chunk_size:
        # We need to hack config? 
        # Or just rely on max_batch_size=256 default?
        # Runner uses `self._max_scratch_tokens`.
        # We can't easily change it via LLM args unless we patch config.
        # But we can override it by patching `vllm_apple.engine.config` or similar.
        # For now, assume default 256. 
        # If Step 3 needs override, we'll handle it.
        pass

    # Clean old dumps
    for f in glob.glob("/tmp/vllm_logits_*.pt"):
        try:
            os.remove(f)
        except: pass
        
    print(f"Loading vLLM model {model_name}...")
    llm = LLM(
        model=model_name,
        dtype="float16",
        enforce_eager=False,
        gpu_memory_utilization=0.6,
        trust_remote_code=True
    )
    
    # Generate
    sampling_params = SamplingParams(max_tokens=1)
    
    # We rely on side-effect (dump)
    print("Generating...")
    # Clean again just before generate to minimize race
    # vLLM accepts list of ints as prompt if configured correctly?
    # Actually, vLLM 0.6+ supports 'prompts' as list of tokens?
    # Or TextPrompt / TokensPrompt objects?
    # We will try passing the list of ints directly.
    # Note: prompts=[tokens_list]
    try:
        llm.generate(prompts=[tokens_list], sampling_params=sampling_params)
    except Exception as e:
        print(f"Generation failed with tokens: {e}")
        print("Fallback to dict format...")
        # Try dict format: {"prompt_token_ids": ...}
        llm.generate(prompts=[{"prompt_token_ids": tokens_list}], sampling_params=sampling_params)
    
    # Find dump
    time.sleep(1.0)
    files = glob.glob("/tmp/vllm_logits_*.pt")
    if not files:
        print("Error: No dump files found")
        sys.exit(1)
        
    # Expecting one file for one step?
    # Or multiple steps (warmup)?
    # We want the one corresponding to the prompt.
    # Warmup uses small prompts.
    # Our prompt is likely the largest input.
    # So sorting by file size might help?
    # Or just latest mtime.
    files.sort(key=os.path.getmtime)
    target_file = files[-1]
    print(f"Loading dump from {target_file}")
    
    logits = torch.load(target_file)
    # Shape should be [1, vocab]
    print(f"Dump shape: {logits.shape}")
    
    return logits[0].cpu() # [vocab]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_size", type=int, default=None)
    args = parser.parse_args()
    
    # Generate deterministic prompt
    # "The quick brown fox " repeated
    base = "The quick brown fox jumps over the lazy dog. "
    prompt = base * (args.prompt_len // len(base.split()) + 1)
    # Tokenize once using HF tokenizer to ensure identity
    print(f"Tokenizing prompt using {args.model}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # input_ids is [1, seq_len]
    tokens_list = input_ids[0].tolist()
    print(f"Input tokens (len={len(tokens_list)}): {tokens_list[:10]}...{tokens_list[-5:]}")
    
    if args.backend == "hf":
        logits = run_hf(args.model, input_ids)
    else:
        logits = run_vllm(args.model, tokens_list, args.chunk_size)
    
    torch.save(logits, args.output)
    print(f"Saved logits to {args.output}")
