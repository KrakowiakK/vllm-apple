#!/bin/bash
export MODEL_NAME=Qwen/Qwen2-0.5B
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
PYTHON=./.venv/bin/python

# Step 1
echo "Running Step 1 Torch..."
VLLM_APPLE_USE_ENGINE=0 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_step1_torch.json

echo "Running Step 1 Engine..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_step1_engine.json

# Step 3 (Toggles)
echo "Running Toggle 1 (No Chunking)..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 VLLM_PREFILL_DISABLE_CHUNKING=1 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_toggle1.json

echo "Running Toggle 2 (Torch RoPE)..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 VLLM_PREFILL_FORCE_PYTORCH_ROPE=1 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_toggle2.json

echo "Running Toggle 3 (Torch Attn)..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 VLLM_PREFILL_USE_PYTORCH_ATTN=1 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_toggle3.json

echo "Running Toggle 4 (Float32 Norm)..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 VLLM_PREFILL_FLOAT32_NORM=1 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_toggle4.json

# Step 4
echo "Running Debug..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 VLLM_PREFILL_EQ_DEBUG=1 TARGET_LEN=500 $PYTHON diagnose_worker.py > res_debug.txt 2>&1

echo "Done."
