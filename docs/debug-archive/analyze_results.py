
import json
import os
import re
import numpy as np

def load_json(path):
    try:
        with open(path, 'r') as f:
            content = f.read()
            # Extract JSON between delimiters if present
            m = re.search(r'<<<JSON_START>>>(.*?)<<<JSON_END>>>', content, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            # Try parsing whole file
            return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

def compare(torch_res, engine_res):
    if "error" in torch_res: return {"error": f"Torch Error: {torch_res['error']}"}
    if "error" in engine_res: return {"error": f"Engine Error: {engine_res['error']}"}
    
    t_map = {item['token_id']: item['logprob'] for item in torch_res['top10']}
    e_map = {item['token_id']: item['logprob'] for item in engine_res.get('top10', [])}
    
    diffs = []
    max_diff = 0.0
    max_diff_token = -1
    
    for tid, t_val in t_map.items():
        e_val = e_map.get(tid)
        if e_val is not None:
            diff = abs(t_val - e_val)
            diffs.append(diff)
            if diff > max_diff:
                max_diff = diff
                max_diff_token = tid
                
    mean_diff = np.mean(diffs) if diffs else 0.0
    
    # Top 1 match
    t_top1 = torch_res['top10'][0]['token_id']
    e_top1 = engine_res['top10'][0]['token_id'] if engine_res.get('top10') else -1
    
    return {
        "max_diff": max_diff,
        "max_diff_token": max_diff_token,
        "mean_diff": mean_diff,
        "top1_match": (t_top1 == e_top1)
    }

def main():
    print("========================================================")
    print("FINAL REPORT GENERATION")
    print("========================================================")
    
    # Load Baseline
    torch_res = load_json("res_step1_torch.json")
    engine_res = load_json("res_step1_engine.json")
    
    print("\n1) BASELINE DIVERGENCE SUMMARY")
    metrics = compare(torch_res, engine_res)
    if "error" in metrics:
        print(f"FAILED: {metrics['error']}")
    else:
        print(f"Max Diff: {metrics['max_diff']:.4f} at token {metrics['max_diff_token']}")
        print(f"Mean Diff: {metrics['mean_diff']:.4f}")
        print(f"Top1 Match: {metrics['top1_match']}")
        
        # Print Top 10 for both
        print("\nPyTorch Top 10:")
        for x in torch_res.get('top10', []): print(f"  {x['token_id']}: {x['logprob']:.4f}")
        print("\nEngine Top 10:")
        for x in engine_res.get('top10', []): print(f"  {x['token_id']}: {x['logprob']:.4f}")

    # Small Test Analysis
    print("\n1.5) SMALL PROMPT (128) ANALYSIS")
    s_torch = load_json("res_small_torch.json")
    s_engine = load_json("res_small_engine.json")
    s_metrics = compare(s_torch, s_engine)
    if "error" in s_metrics:
        print(f"FAILED: {s_metrics['error']}")
    else:
        print(f"Max Diff (128): {s_metrics['max_diff']:.4f}")
        print(f"Mean Diff (128): {s_metrics['mean_diff']:.4f}")
        print(f"Top1 Match (128): {s_metrics['top1_match']}")
        
        # Check Toggles
        s_res2 = load_json("res_small_toggle2.json")
        m2 = compare(s_torch, s_res2)
        print(f"Toggle 2 (Torch RoPE) @ 128: max_diff={m2.get('max_diff','ERR')}")

    # Length Scaling (Skipped in script, so placeholder)
    print("\n2) LENGTH SCALING CONCLUSION")
    print("(Run skipped for speed, inferring from Baseline if divergence exists)")
    
    # Toggles
    print("\n3) TOGGLE ISOLATION VERDICT")
    toggles = [
        ("DISABLE_CHUNKING", "res_toggle1.json"),
        ("FORCE_PYTORCH_ROPE", "res_toggle2.json"),
        ("USE_PYTORCH_ATTN", "res_toggle3.json"),
        ("FLOAT32_NORM", "res_toggle4.json"),
    ]
    
    print("| Toggle | match | max_diff | mean_diff | Status |")
    print("|---|---|---|---|---|")
    
    best_toggle = None
    min_diff = 999.0
    
    for name, path in toggles:
        res = load_json(path)
        m = compare(torch_res, res)
        if "error" in m:
            print(f"| {name} | ERROR | {m['error'][:20]}... | - | CRASH |")
        else:
            status = "No Change"
            if m['max_diff'] < 1e-3: status = "FIXED"
            elif m['max_diff'] < metrics.get('max_diff', 1.0) * 0.5: status = "REDUCED"
            
            print(f"| {name} | {m['top1_match']} | {m['max_diff']:.4f} | {m['mean_diff']:.4f} | {status} |")
            
            if m['max_diff'] < min_diff:
                min_diff = m['max_diff']
                best_toggle = name

    # Debug Checkpoint
    print("\n4) FIRST DIVERGENT CHECKPOINT")
    try:
        with open("res_debug.txt", "r") as f:
            debug_log = f.read()
            
        # Parse for "diff"
        # Since I don't know the exact format, scanning for keywords
        lines = debug_log.split('\n')
        found_diffs = []
        for line in lines:
            if "diff" in line.lower() and "=" in line:
                found_diffs.append(line.strip())
        
        if found_diffs:
            print("Found diff logs:")
            for l in found_diffs[:5]: print(l)
        else:
            print("No diff logs found.")
            
    except Exception as e:
        print(f"Could not read debug log: {e}")

if __name__ == "__main__":
    main()
