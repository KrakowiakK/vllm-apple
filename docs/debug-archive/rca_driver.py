
import subprocess
import json
import os
import re
import sys
import numpy as np

WORKER_SCRIPT = "diagnose_worker.py"
MODEL_NAME = "Qwen/Qwen2-0.5B"

def run_cmd(env_vars, target_len=500):
    env = os.environ.copy()
    env.update(env_vars)
    env["TARGET_LEN"] = str(target_len)
    env["MODEL_NAME"] = MODEL_NAME
    # Force localized execution
    cmd = [sys.executable, WORKER_SCRIPT]
    
    # Use explicit file object redirection
    outfile = "worker_output.txt"
    try:
        with open(outfile, "w") as f:
            ret = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            
        with open(outfile, "r") as f:
            result_stdout = f.read()
            
        if ret != 0:
            return None, f"Execution Failed (ExtCode {ret}):\n{result_stdout}"
            
    except Exception as e:
        return None, f"Execution Error: {e}"
    
    # Parse JSON
    try:
        json_match = re.search(r'<<<JSON_START>>>(.*?)<<<JSON_END>>>', result_stdout, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return data, result_stdout
        else:
            return None, result_stdout
    except Exception as e:
        return None, f"Parse Error: {e}\nOutput: {result_stdout}"

def compare_results(res_torch, res_engine):
    # Map token_ids to logprobs
    t_map = {item['token_id']: item['logprob'] for item in res_torch['top10']}
    e_map = {item['token_id']: item['logprob'] for item in res_engine['top10']}
    
    # Find common tokens to compare diffs (or just use Torch top-10)
    # The requirement is "max|logit_diff|". We can only compare for token_ids present in both or just use Torch's top list as reference.
    # We will compare for the Top 10 tokens from PyTorch.
    
    diffs = []
    max_diff = 0.0
    max_diff_token = -1
    
    for item in res_torch['top10']:
        tid = item['token_id']
        t_val = item['logprob']
        e_val = e_map.get(tid)
        
        if e_val is not None:
            diff = abs(t_val - e_val)
            diffs.append(diff)
            if diff > max_diff:
                max_diff = diff
                max_diff_token = tid
        else:
            # excessive divergence if token not in top 10?
            pass
            
    mean_diff = np.mean(diffs) if diffs else 0.0
    
    # Check Top-1 Match
    t_top1 = res_torch['top10'][0]['token_id']
    e_top1 = res_engine['top10'][0]['token_id']
    match = (t_top1 == e_top1)
    
    return {
        "max_diff": max_diff,
        "max_diff_token": max_diff_token,
        "mean_diff": mean_diff,
        "top1_match": match,
        "torch_top1": t_top1,
        "engine_top1": e_top1,
        "torch_margin": res_torch['top1_margin'],
        "engine_margin": res_engine['top1_margin']
    }

def print_step1_report(res_torch, res_engine, metrics):
    print("========================================================")
    print("STEP 1 — BASELINE LOGIT COMPARISON")
    print("========================================================")
    
    def print_path(name, res):
        print(f"PATH: {name}")
        print("Top-10:")
        for item in res['top10']:
            print(f"  {item['token_id']} = {item['logprob']:.4f} ({item['text']})")
        print(f"Top1 margin: {res['top1_margin']:.4f}")
        print("")

    print_path("PYTORCH", res_torch)
    print_path("ENGINE", res_engine)
    
    print("CROSS-COMPARISON:")
    print(f"max|diff| = {metrics['max_diff']:.4f} at token_id {metrics['max_diff_token']}")
    print(f"mean|diff| = {metrics['mean_diff']:.4f}")
    print("")

def run_step1():
    print("Running Step 1...")
    env_torch = {"VLLM_APPLE_USE_ENGINE": "0"}
    env_engine = {"VLLM_APPLE_USE_ENGINE": "1", "VLLM_APPLE_ENGINE_PREFILL": "1"}
    
    print("  Running Baseline PyTorch...")
    res_torch, _ = run_cmd(env_torch, 500)
    print("  Running Baseline Engine...")
    res_engine, _ = run_cmd(env_engine, 500)
    
    if not res_torch or not res_engine:
        print("Step 1 Failed: Could not get results.")
        return None, None
        
    metrics = compare_results(res_torch, res_engine)
    print_step1_report(res_torch, res_engine, metrics)
    return res_torch, res_engine

def run_step2():
    print("========================================================")
    print("STEP 2 — LENGTH SCALING ANALYSIS")
    print("========================================================")
    lengths = [32, 128, 512, 2048, 8192]
    
    print("| Tokens | max|diff| | mean|diff| | top1_match | top1_margin |")
    print("|---|---|---|---|---|")
    
    for length in lengths:
        print(f"  Testing length {length}...")
        env_torch = {"VLLM_APPLE_USE_ENGINE": "0"}
        env_engine = {"VLLM_APPLE_USE_ENGINE": "1", "VLLM_APPLE_ENGINE_PREFILL": "1"}
        
        # We need to run torch again for each length effectively
        res_t, _ = run_cmd(env_torch, length)
        res_e, _ = run_cmd(env_engine, length)
        
        if res_t and res_e:
            m = compare_results(res_t, res_e)
            match_str = "YES" if m['top1_match'] else "NO"
            print(f"| {length} | {m['max_diff']:.4f} | {m['mean_diff']:.4f} | {match_str} | {m['torch_margin']:.4f} |")
        else:
            print(f"| {length} | ERROR | ERROR | ERROR | ERROR |")
    print("")

def run_step3(res_torch_500):
    print("========================================================")
    print("STEP 3 — ISOLATION VIA TOGGLES")
    print("========================================================")
    
    toggles = [
        ("VLLM_PREFILL_DISABLE_CHUNKING", "1"),
        ("VLLM_PREFILL_FORCE_PYTORCH_ROPE", "1"),
        ("VLLM_PREFILL_USE_PYTORCH_ATTN", "1"),
        ("VLLM_PREFILL_FLOAT32_NORM", "1")
    ]
    
    if not res_torch_500:
        # Re-run baseline torch if needed
        res_torch_500, _ = run_cmd({"VLLM_APPLE_USE_ENGINE": "0"}, 500)

    print("| Toggle | top1_match | max|diff| | mean|diff| | Interpretation |")
    print("|---|---|---|---|---|")

    for name, val in toggles:
        print(f"  Testing toggle {name}...")
        env = {
            "VLLM_APPLE_USE_ENGINE": "1", 
            "VLLM_APPLE_ENGINE_PREFILL": "1",
            name: val
        }
        res_e, _ = run_cmd(env, 500)
        
        if res_e:
            m = compare_results(res_torch_500, res_e)
            match_str = "YES" if m['top1_match'] else "NO"
            
            # Simple interpretation logic
            interp = "No Effect"
            if m['max_diff'] < 1e-3:
                interp = "**ISOLATED** (Fixes it)"
            elif m['max_diff'] < 0.1:
                interp = "Reduces"
                
            print(f"| {name} | {match_str} | {m['max_diff']:.4f} | {m['mean_diff']:.4f} | {interp} |")
        else:
             print(f"| {name} | ERROR | ERROR | ERROR | ERROR |")
    print("")

def run_step4():
    print("========================================================")
    print("STEP 4 — FIRST DIVERGENCE CHECKPOINT")
    print("========================================================")
    
    # Needs VLLM_PREFILL_EQ_DEBUG=1
    env = {
        "VLLM_APPLE_USE_ENGINE": "1",
        "VLLM_APPLE_ENGINE_PREFILL": "1",
        "VLLM_PREFILL_EQ_DEBUG": "1"
    }
    
    # We rely on the stdout of this run to find the checkpoint lines.
    # Assuming format: "EQ_DEBUG: checkpoint=layerX_op max_diff=..."
    # The user says "Identify FIRST checkpoint where divergence appears".
    
    print("Running with DEBUG flags...")
    _, output = run_cmd(env, 500)
    
    # Parse output
    # Looking for lines that might look like:
    # [Diff] layer0_rope: max_diff=0.001
    # or anything similar. We need to scan the logs.
    
    # If I don't know the exact format, I will dump relevant lines.
    # The requirement asks me to report:
    # FIRST DIVERGENCE:
    #   checkpoint = layerX_<operation>
    #   max|diff| = X.XXXX
    
    found = False
    lines = output.split('\n')
    # Regex guess based on common debug prints or just look for "diff"
    # User prompt: "Identify FIRST checkpoint where divergence appears"
    # This implies the tool dumps checkpoint diffs.
    
    divergence_threshold = 1e-3
    
    print("Parsing debug output...")
    
    # Regex to capture typical debug pattern. 
    # Adjust valid patterns if we see real output.
    # Pattern assumption: "layer(\d+)_(\w+).*diff[:=]\s*([\d\.]+)"
    
    checkpoints = []
    
    for line in lines:
        if "diff" in line.lower() or "divergence" in line.lower():
            # Try to start capturing data
            # print(f"DEBUG_LINE: {line}") # Optional: verify
            pass

    # NOTE: Since I can't predict the exact log format without running it, 
    # I will inspect the output of a short run in the "Execution" phase if this step fails to parse.
    # For now, I will scan for general "tensor comparison" logs.
    
    # Better approach: Just print the lines containing "diff" and let the user (me) decide in the report generation,
    # OR try to parse automatically.
    # I'll try to find the first line with > 1e-3 diff.
    
    first_div = None
    
    # Mocking parser for the "layer-by-layer checkpoint capture"
    # I'll look for lines like "CHECKPOINT: name ... max_diff: val"
    
    re_chk = re.compile(r'(layer\d+_[a-zA-Z0-9_]+).*diff.*?([\d\.e-]+)', re.IGNORECASE)
    
    for line in lines:
        m = re_chk.search(line)
        if m:
            name = m.group(1)
            try:
                val = float(m.group(2))
                if val > divergence_threshold and first_div is None:
                    first_div = (name, val)
            except:
                pass
                
    if first_div:
        print(f"FIRST DIVERGENCE:")
        print(f"  checkpoint = {first_div[0]}")
        print(f"  max|diff| = {first_div[1]:.6f}")
        # mean diff might be harder to find without regex match, will add placeholders
    else:
        print("No Divergence > 1e-3 found in logs (or regex failed).")
        # Print some candidate lines for manual inspection
        print("Debug Log Sample (Lines with 'diff'):")
        count = 0
        for line in lines:
            if "diff" in line.lower():
                print(f"  {line.strip()}")
                count += 1
                if count > 10: break

def main():
    try:
        t_500, e_500 = run_step1()
        run_step2()
        run_step3(t_500)
        run_step4()
    except KeyboardInterrupt:
        print("Interrupted.")

if __name__ == "__main__":
    main()
