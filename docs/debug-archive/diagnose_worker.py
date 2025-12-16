
import os
import sys
import json
import torch
import numpy as np
from vllm import LLM, SamplingParams

def get_fox_prompt(repeats):
    base_prompt = "The quick brown fox jumps over the lazy dog. "
    return base_prompt * repeats

def run_worker():
    # Parse generic args from env to avoid complex CLI for now
    try:
        target_len = int(os.environ.get("TARGET_LEN", "500"))
        # Using opt-125m as it's small and standard for testing if not specified
        model_name = os.environ.get("MODEL_NAME", "facebook/opt-125m")
        
        # Calculate repeats based on target length
        # Prompt roughly 9 tokens per repeat
        if target_len == 500:
            repeats = 55
        elif target_len == 32:
            repeats = 4
        elif target_len == 128:
            repeats = 14
        elif target_len == 512:
            repeats = 56
        elif target_len == 2048:
            repeats = 225
        elif target_len == 8192:
            repeats = 900
        else:
            repeats = max(1, int(target_len / 9))
            
        prompt = get_fox_prompt(repeats)
        
        # Output setup
        result = {
            "target_len": target_len,
            "repeats": repeats,
            "model": model_name,
            "top10": [], # list of [token_id, logprob, decoded_token]
            "top1_margin": 0.0,
            "error": None
        }

        # Initialize LLM
        # Redirect stdout/stderr to capture vLLM noise if needed, but for now we just let it show
        # and print our JSON result at the very end marked by a delimiter.
        llm = LLM(
            model=model_name,
            dtype="float16",
            max_model_len=max(16384, target_len * 2),
            enforce_eager=False,
            gpu_memory_utilization=0.6,
            trust_remote_code=True
        )
        
        sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=1, 
            logprobs=20  # Get top 20 to be safe
        )
        
        print("Starting generation...")
        sys.stdout.flush()
        try:
            outputs = llm.generate([prompt], sampling_params)
        except Exception as e:
            print(f"Generate Error: {e}")
            raise
        print("Generation complete")
        sys.stdout.flush()
        output = outputs[0]
        
        # Extract Logprobs
        try:
            if not output.outputs:
                raise ValueError("No outputs found in RequestOutput")
            if not output.outputs[0].logprobs:
                # Debug: print the whole output object
                print(f"DEBUG: Output structure: {output}")
                raise ValueError("No logprobs found in CompletionOutput")
            
            first_token_logprobs = output.outputs[0].logprobs[0] # {token_id: logprob}
        except Exception as e:
            print(f"Extraction Error: {e}")
            print(f"Full Output: {output}")
            raise
        sorted_items = sorted(first_token_logprobs.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate margin
        if len(sorted_items) >= 2:
            margin = sorted_items[0][1] - sorted_items[1][1]
        else:
            margin = 0.0
            
        result["top1_margin"] = margin
        
        # Populate top10
        tokenizer = llm.get_tokenizer()
        for i, (tid, val) in enumerate(sorted_items[:10]):
            text_val = tokenizer.decode([tid])
            result["top10"].append({
                "token_id": tid,
                "logprob": float(val),
                "text": text_val
            })
            
        # PRINT RESULT WITH DELIMITER
        print("\n<<<JSON_START>>>")
        print(json.dumps(result))
        print("<<<JSON_END>>>")
        sys.stdout.flush() 
        os._exit(0)
        
    except Exception as e:
        err_res = {"error": str(e)}
        print("\n<<<JSON_START>>>")
        print(json.dumps(err_res))
        print("<<<JSON_END>>>")
        sys.exit(1)

if __name__ == "__main__":
    run_worker()
