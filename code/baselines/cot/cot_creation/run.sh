#!/bin/bash

# Set your API key and other parameters
#WING API: 
API_KEY="your_api_key"
MODEL_NAME="gpt-4"
NUM_SAMPLES=10
DATASET_NAME="scitab"  
SPLIT="train"

DATA_PATH="../../data/datasets"

# Step 1: Generate cot samples
echo "Generating cot samples for Dataset ${DATASET_NAME}..."
python "cot_creation.py" \
    --API_KEY "${API_KEY}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --num_eval_samples "${NUM_SAMPLES}" \
    --data_path "${DATA_PATH}" \
    --save_path "./results"