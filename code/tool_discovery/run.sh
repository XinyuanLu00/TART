#!/bin/bash

# Set your API key and other parameters

API_KEY="your_api_key"
MODEL_NAME="gpt-4"
NUM_SAMPLES=690  
DATASET_NAME="pubhealthtab"  
SPLIT="train"

DATA_PATH="../data/datasets"
DATA_SAMPLE_CODE_PATH="../data/data_sample_code"

# Step 1: Generate tool samples using tool_creation.py
echo "Generating tool samples for Dataset ${DATASET_NAME}..."
python "tool_creation.py" \
    --API_KEY "${API_KEY}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --num_eval_samples "${NUM_SAMPLES}" \
    --data_path "${DATA_PATH}" \
    --save_path "./results"
echo "Generating tool samples completed! "

# Step 2: Parse and execute the generated samples using code_parser.py
echo "Parsing and executing generated samples for Dataset ${DATASET_NAME}..."
python "code_parser.py" \
    --json_file_path "./results/${DATASET_NAME}_${MODEL_NAME}.json" \
    --dataset_name "${DATASET_NAME}" \
    --python_files_folder "${DATA_SAMPLE_CODE_PATH}" \
    --output_folder "${DATASET_NAME}"
echo "Tool discovery process completed!"