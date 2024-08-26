export CUDA_VISIBLE_DEVICES=0

# Set your model path
OUTPUT_PATH="./model_save/llama2/checkpoint-4900"
DATASET=tabfact

echo "Explanation Generator## Dataset ${DATASET}...Start inference model saved in ${OUTPUT_PATH}"
python dataset_inference.py \
    --huggingface_token your_hf_token \
    --data_path ../tool_maker/output/${DATASET}_test_output_codellama.json \
    --output_path ./output/${DATASET}_test_output_codellama.json \
    --num_samples 100 \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --checkpoint_path "${OUTPUT_PATH}" \
    --cache_dir /home/ubuntu/.MODEL_CACHE \
    --load_in_4bit  \
    --max_new_tokens 1024