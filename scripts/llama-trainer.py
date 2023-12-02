
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import numpy as np
import torch
import datasets

import deepspeed
import accelerate
import argparse

import bitsandbytes as bnb
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

TOKEN_QUESTION = "<question>"
TOKEN_END_QUESTION = "<question>"
TOKEN_CONTEXT = "<context>"
TOKEN_END_CONTEXT = "<context>"
TOKEN_ANSWER = "<answer>"
TOKEN_END_ANSWER = "<answer>"
HIGHLIGHT_ANSWER = "<hl>"
SPLIT_SEED = 42
NPROC = 32
HIGHLIGHT = True


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="llama-7B")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--ds", type=bool, default=False)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
LORA_R = args.lora_r
model_name = args.model_name

PRETRAINED_MODEL = "/storagenfs/l.miglior/hlt-project/llama/hf"



checkpoint_dir = f"./results/{model_name}"
log_dir = f"./logs/{model_name}"
save_dir = f"./models/{model_name}"

# print configuration
print("-" * 10 + "CONFIGURATION" + "-" * 10)
print(f"> Model name: {model_name}")
print(f"> Batch size: {BATCH_SIZE}")
print(f"> Epochs: {EPOCHS}")
print(f"> Lora alpha: {LORA_ALPHA}")
print(f"> Lora dropout: {LORA_DROPOUT}")
print(f"> Lora r: {LORA_R}")
print(f"> Checkpoint dir: {checkpoint_dir}")
print(f"> Log dir: {log_dir}")
print(f"> Save dir: {save_dir}")
print("-" * 9 + "END CONFIGURATION" + "-" * 9)

train_config = TrainingArguments(
    output_dir=checkpoint_dir,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    gradient_accumulation_steps=1,
    fp16=True,
    load_best_model_at_end=True,
    seed=42,
    greater_is_better=False,
    local_rank=0,
    report_to=["tensorboard"],
    metric_for_best_model="eval_loss",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    #deepspeed="./deepspeed_config.json",
)


def get_inputs_target(e):
    answer_start = e["answers"]["answer_start"][0]
    # add highlight token to context
    ans_len = len(e["answers"]["text"][0])

    e["context"] = (
        e["context"][:answer_start]
        + " "
        + HIGHLIGHT_ANSWER
        + " "
        + e["context"][answer_start : answer_start + ans_len]
        + " "
        + HIGHLIGHT_ANSWER
        + " "
        + e["context"][answer_start + ans_len :]
    )

    return {
        # answer + context + question for causal language modeling
        "text": f'<s> [INST] generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT} [/INST] {TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION} <s>',
    }


def preprocess_squad_dataset(dataset_name="squad", split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split)
    dataset = dataset.map(get_inputs_target, num_proc=NPROC)
    dataset = dataset.remove_columns(["answers", "context", "question"])
    return dataset


# load dataset
train_dataset = preprocess_squad_dataset(dataset_name="squad", split="train")
valid_dataset = preprocess_squad_dataset(dataset_name="squad", split="validation")


# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Quantize Model
peft_config = LoraConfig(
    r=LORA_R, 
    lora_alpha=LORA_ALPHA, 
    lora_dropout=LORA_DROPOUT, 
    task_type=TaskType.CAUSAL_LM
)

print("> Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL, 
    #quantization_config=quant_config, 
    load_in_4bit=True,
    low_cpu_mem_usage=True
)
print("> Model loaded!")

print("> Loading tokenizer...")
llama_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"
print("> Tokenizer loaded!")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset.shuffle(seed=SPLIT_SEED).select(range(40000)),
    eval_dataset=valid_dataset.select(range(1000)),
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_config,
    max_seq_length=64,
)

trainer.train()

trainer.save_model(save_dir)



