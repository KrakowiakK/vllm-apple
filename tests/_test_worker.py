
import os
import sys
import torch
import json
# Set env var before importing vllm just in case
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
from vllm import LLM, SamplingParams

def run_worker():
    target_len = int(os.environ.get("TARGET_LEN", "256"))
    use_engine = os.environ.get("VLLM_APPLE_USE_ENGINE", "1") == "1"
    model_name = "Qwen/Qwen2-0.5B"
    out_file = os.environ.get("OUTPUT_FILE", "logits.pt")

    # Construct prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * (target_len // 9 + 1)
    
    print(f"Initializing LLM (Engine={use_engine}, Len={target_len})...")
    
    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=16384,
        enforce_eager=False,
        gpu_memory_utilization=0.6,
        trust_remote_code=True
    )
    
    # We want generated token logprobs
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20
    )
    
    import glob
    import time
    
    # Clean up previous dumps
    if use_engine:
        for f in glob.glob(f"/tmp/vllm_logits_*{os.getpid()}.pt"):
            try:
                os.remove(f)
            except:
                pass
        os.environ["VLLM_DUMP_LOGITS"] = "1"

    print("Generating...")
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]
    
    last_lp_map = None
    
    if use_engine:
        # Load from dump
        print("Looking for dumped logits...")
        # Give a moment for flush
        time.sleep(2.0) # Wait a bit more for file system
        files = glob.glob("/tmp/vllm_logits_*.pt")
        # Filter by recent?
        # Just take the latest modified
        if not files:
            print("Error: No dump files found (glob)")
            sys.exit(1)
            
        files.sort(key=os.path.getmtime)
        latest_file = files[-1]
        print(f"Loading logits from {latest_file}")
        
        full_logits = torch.load(latest_file)
        # We want the last token's logits
        last_token_logits = full_logits[-1]
        
        logprobs = torch.nn.functional.log_softmax(last_token_logits, dim=0)
        
        # Get Top-20
        topk_vals, topk_inds = torch.topk(logprobs, 20)
        last_lp_map = {int(idx): float(val) for idx, val in zip(topk_inds, topk_vals)}
        
    else:
        # Standard Torch extraction
        gen_logprobs = output.outputs[0].logprobs
        if not gen_logprobs:
            print("Warning: No generated logprobs (Torch). Using token_id fallback.")
            if output.outputs[0].token_ids:
                 tid = output.outputs[0].token_ids[0]
                 last_lp_map = {tid: 0.0}
            else:
                 print("Error: No token_ids either!")
                 sys.exit(1)
        else:
            last_lp_map = gen_logprobs[0]

    # Dump the map
    torch.save(last_lp_map, out_file)
    print(f"Saved logprobs to {out_file}")
    sys.stdout.flush()
    # Force exit to avoid cleanup hangs
    os._exit(0)

if __name__ == "__main__":
    run_worker()
