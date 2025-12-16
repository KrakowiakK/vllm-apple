#!/bin/bash
export MODEL_NAME=Qwen/Qwen2-0.5B
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
PYTHON=./.venv/bin/python

# Step 1 Small
echo "Running Step 1 Torch (128)..."
VLLM_APPLE_USE_ENGINE=0 TARGET_LEN=128 $PYTHON diagnose_worker.py > res_small_torch.json

echo "Running Step 1 Engine (128)..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 TARGET_LEN=128 $PYTHON diagnose_worker.py > res_small_engine.json

# Toggles Small
echo "Running Toggle 2 (Torch RoPE) (128)..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 VLLM_PREFILL_FORCE_PYTORCH_ROPE=1 TARGET_LEN=128 $PYTHON diagnose_worker.py > res_small_toggle2.json
