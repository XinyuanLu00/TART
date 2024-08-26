from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer  # Ensure your custom trainer can handle DDP if it's not a native PyTorch Trainer
import wandb

tqdm.pandas()
wandb.init(project="llama")

# Initialize DDP
import os
import torch.distributed as dist
dist.init_process_group(backend="nccl")

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """The name of the Casual LM model we wish to fine-tune with SFTTrainer."""
    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Whether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "Lora r value for PEFT"})
    peft_lora_alpha: Optional[float] = field(default=None, metadata={"help": "Lora alpha value for PEFT"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the model
model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
model = torch.nn.parallel.DistributedDataParallel(model.cuda())  # Ensure model is moved to GPU

# Load the dataset
dataset = load_dataset(script_args.dataset_name, split="train")

# Training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    num_train_epochs=script_args.num_train_epochs,
    logging_dir='./logs',
    logging_steps=100
)

# Define PEFT config if used
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
)

# Start training
trainer.train()

# Save the model
trainer.save_model(script_args.output_dir)
