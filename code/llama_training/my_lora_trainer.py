from datasets import load_dataset
from trl import SFTTrainer
import wandb
import argparse
import os

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--dataset_path", type=str, help="the dataset path where the json is stored")
    parser.add_argument("--dataset_text_field", type=str, default="text", help="the text field of the dataset")
    parser.add_argument("--seq_length", type=int, default=512, help="Input sequence length")
    # model arguments
    parser.add_argument("--model_name", type=str, help="the model name")
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache directory to save the model")
    parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision")
    parser.add_argument("--load_in_4bit", action='store_true', help="load the model in 4 bits precision")
    parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
    # parser.add_argument("--use_auth_token", type=bool, default=True, help="Use HF auth token to access the model")
    parser.add_argument("--huggingface_token", type=str, default=None, help="The HF auth token")
    # training arguments
    ## 1. saving
    parser.add_argument("--output_dir", type=str, default="output", help="the output directory")
    parser.add_argument("--save_steps", type=int, default=100, help="Number of updates steps before two checkpoint saves")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Limits total number of checkpoints.")
    parser.add_argument("--push_to_hub", action='store_true', help="Push the model to HF Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the model on HF Hub")
    ## 2. logging
    parser.add_argument("--log_with", type=str, default=None, help="use 'wandb' to log with wandb")
    parser.add_argument("--wandb_project", type=str, help="the wandb project name")
    parser.add_argument("--logging_steps", type=int, default=1, help="the number of logging steps")
    ## 3. learning
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="the learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="the batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="the number of gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="the number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="the number of training steps")
    ## 4. peft & lora args
    parser.add_argument("--use_peft", action='store_true', help="Wether to use PEFT or not to train adapters")
    parser.add_argument("--peft_lora_r", type=int, default=64, help="the r parameter of the LoRA adapters")
    parser.add_argument("--peft_lora_alpha", type=int, default=16, help="the alpha parameter of the LoRA adapters")

    args = parser.parse_args()
    return args

class LoraTrainer:
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset_path = args.dataset_path
        self.args = args
        self.load_data()
        self.load_model()

    def load_data(self):
        #train_file = os.path.join(self.dataset_path, 'train.jsonl')
        #eval_file = os.path.join(self.dataset_path, 'train.jsonl')
        train_file = self.dataset_path
        eval_file = self.dataset_path
        self.train_dataset = load_dataset('json', data_files=train_file, split='train', download_mode='force_redownload')  
        self.dev_dataset = load_dataset('json', data_files=eval_file, split='train', download_mode='force_redownload')

    def load_model(self):
        if self.args.load_in_8bit and self.args.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif self.args.load_in_8bit or self.args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.args.load_in_8bit, load_in_4bit=self.args.load_in_4bit
            )
            #device_map = {"": 0} # fit the entire model on the GPU:0
            device_map = "auto"
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=self.args.trust_remote_code,
            torch_dtype=torch_dtype,
            # use_auth_token=self.args.use_auth_token,
            token=self.args.huggingface_token,
            cache_dir=self.args.cache_dir,
        )
        print(f"Model {self.model_name} loaded")

    def train(self):
        # Step 1: Define the training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=self.args.logging_steps,
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            report_to=self.args.log_with,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            push_to_hub=self.args.push_to_hub,
            hub_model_id=self.args.hub_model_id
        )

        # Step 2: Define the LoraConfig
        if self.args.use_peft:
            peft_config = LoraConfig(
                r=self.args.peft_lora_r,
                lora_alpha=self.args.peft_lora_alpha,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            peft_config = None

        # Step 3: Define the Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            max_seq_length=self.args.seq_length,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            dataset_text_field=self.args.dataset_text_field,
            peft_config=peft_config,
        )

        trainer.train()

        # Step 4: Save the model
        trainer.save_model(self.args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    tqdm.pandas()
    wandb.init(project=args.wandb_project)
    trainer = LoraTrainer(args)
    trainer.train()