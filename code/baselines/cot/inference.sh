#--model_name 
#codellama/CodeLlama-7b-hf 
#meta-llama/Llama-2-7b-hf 
#meta-llama/Meta-Llama-3-8B
#allenai/tulu-2-7b
#deepseek-ai/deepseek-coder-7b-instruct-v1.5
#osunlp/TableLlama
#export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set your model path
OUTPUT_PATH="./model_save/codellama_7b_train_all/checkpoint-12300"
DATASET=finqa

echo "CoT## Dataset ${DATASET}...Start inference model saved in ${OUTPUT_PATH}"
python dataset_inference.py \
    --huggingface_token your_hf_token \
    --data_path ../data/datasets/${DATASET}/test.json \
    --output_path ./output/${DATASET}_test_output.json \
    --num_samples 188 \
    --base_model_name codellama/CodeLlama-7b-hf \
    --checkpoint_path "${OUTPUT_PATH}" \
    --cache_dir /home/luxinyuan/.MODEL_CACHE \
    --load_in_4bit  \
    --max_new_tokens 1400