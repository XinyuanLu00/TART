export CUDA_VISIBLE_DEVICES=7

python ../llama_training/my_lora_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_path ./data/tabmwp_train.jsonl \
    --seq_length 1500 \
    --cache_dir /home/ubuntu/.MODEL_CACHE \
    --load_in_4bit \
    --output_dir ./model_save/llama2_train_all \
    --log_with wandb \
    --wandb_project llama \
    --huggingface_token your_hf_token \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \