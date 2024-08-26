#!/bin/bash

#--model_name 
#codellama/CodeLlama-7b-hf 
#meta-llama/Llama-2-7b-hf 
#meta-llama/Meta-Llama-3-8B
#    --tart_results_path "./output/${DATASET}_test_output_ds188_ds188.json" \

DATASET=tabmwp
COT_RESULTS_PATH="../cot/output/${DATASET}_test_output_lm3.json"


echo "## Dataset ${DATASET} ## Starting the combined inference process..."
python dataset_inference_tart.py \
    --tart_results_path "./output/${DATASET}_test_output_lm3_lm3.json" \
    --cot_results_path "${COT_RESULTS_PATH}" \
    --output_path "./output_tart_w_cot/${DATASET}_test_output_lm3_lm3_w_cot_lm3.json"

echo "## Dataset ${DATASET} ## Inference combining process completed."
