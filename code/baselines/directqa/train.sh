#--model_name
#codellama/CodeLlama-7b-hf 
#meta-llama/Llama-2-7b-hf 
#meta-llama/Meta-Llama-3-8B
#allenai/tulu-2-7b
#deepseek-ai/deepseek-coder-7b-instruct-v1.5
#osunlp/TableLlama
export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_NAME=allenai/tulu-2-7b

python ../llama_training/my_lora_trainer.py \
    --model_name ${MODEL_NAME} \
    --dataset_path ./data/train_all.jsonl \
    --seq_length 1024 \
    --cache_dir /home/luxinyuan/.MODEL_CACHE \
    --load_in_4bit \
    --output_dir ./model_save/tulu2_7b_train_all \
    --log_with wandb \
    --wandb_project tart \
    --huggingface_token your_hk_token \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \