#!/bin/bash

# Set your API key and other parameters

API_KEY="your_api_key"
MODEL_NAME="gpt-4"
NUM_SAMPLES=1000  
DATASET_NAME="finqa"  
SPLIT="train"

DATA_PATH="../../data/datasets/"
DATA_SAMPLE_CODE_PATH="../../data/data_sample_code_with_line_number/${DATASET_NAME}"
echo "Generating samples for Dataset ${DATASET_NAME}..."
python "data_generation.py" \
    --API_KEY "${API_KEY}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --num_eval_samples "${NUM_SAMPLES}" \
    --data_path "${DATA_PATH}" \
    --python_code_dir "${DATA_SAMPLE_CODE_PATH}"\
    --save_path "./results"
echo "Generating samples completed! "