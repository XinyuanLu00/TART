#!/bin/bash
#model
#"gpt-3.5-turbo"
#gpt-4o
# Set your API key and other parameters
#Your API: 
API_KEY="your_api_key"
MODEL_NAME="gpt-4"
NUM_SAMPLES=188
DATASET_NAME="finqa"
SPLIT="test"

DATA_PATH="../../data/datasets"

# Step 1: Generate table formatter samples using data_creation.py
echo "Generating table formatter samples..##Model ${MODEL_NAME}### Dataset ${DATASET_NAME}"

python "data_creation.py" \
    --API_KEY "${API_KEY}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --num_eval_samples "${NUM_SAMPLES}" \
    --data_path "${DATA_PATH}" \
    --save_path "./results"

echo "Dataset ${DATASET_NAME}##Model ${MODEL_NAME} Generating formatter samples completed!"