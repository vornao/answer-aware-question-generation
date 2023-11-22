"""Train T5 model on a given dataset, for answer aware question generation task. (Basically a sequence to sequence task)"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import gc
import argparse

# set torch not to use 0th GPU. use only 1st, 2nd, 3rd, GPU.
import datasets
import transformers
from transformers import Trainer
from peft import LoraConfig, TaskType, get_peft_model
import torch

DEEPSPEED_CONFIG = (
    "/storagenfs/l.miglior/answer-aware-question-generation/configs/t5deepspeed.json"
)

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

parser.add_argument("--model_name", type=str, default="t5-base")
parser.add_argument("--highlight", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=64)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_r", type=int, default=32)

args = parser.parse_args()

HIGHLIGHT = args.highlight
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
LORA_R = args.lora_r
model_name = args.model_name

if HIGHLIGHT:
    checkpoint_dir = f"./results/{model_name}-hl"
    log_dir = f"./logs/{model_name}-hl"
    save_dir = f"./models/{model_name}-hl"
else:
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
print(f"> Highlight: {HIGHLIGHT}")
print(f"> Checkpoint dir: {checkpoint_dir}")
print(f"> Log dir: {log_dir}")
print(f"> Save dir: {save_dir}")
print("-" * 9 + "END CONFIGURATION" + "-" * 9)


training_args = transformers.TrainingArguments(
    output_dir=checkpoint_dir,  # output directory
    num_train_epochs=EPOCHS,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=128,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # stderength of weight decay
    logging_dir=log_dir,  # directory for storing logs
    do_eval=True,  # do evaluation
    fp16=True,  # use mixed precision trainin
    report_to="tensorboard",
    logging_steps=10,
    eval_steps=200,
    evaluation_strategy="steps",
    optim="adamw_8bit",
    learning_rate=1e-4,
    save_strategy="epoch",
)

# this works fine with deepspeed.
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v"],
)

# Load Models from pretrained
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.T5ForConditionalGeneration.from_pretrained(
    model_name, load_in_4bit=True, low_cpu_mem_usage=True, device_map="auto"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
gc.collect()
torch.cuda.empty_cache()


def get_inputs_target(e):
    answer_start = e["answers"]["answer_start"][0]
    # add highlight token to context
    ans_len = len(e["answers"]["text"][0])

    if HIGHLIGHT:
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
        # answer + context
        "inputs": f'generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT}',
        # question
        "target": f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}',
    }


# set torch not to use 0th GPU. use only 1st, 2nd, 3rd, GPU.
def preprocess_squad_dataset(dataset_name="squad", split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split)
    # Add question, answer and context tokens to dataset in a new column named text
    dataset = dataset.map(
        lambda e: {
            # answer + context
            "inputs": f'generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT}',
            # question
            "target": f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}',
        },
        num_proc=NPROC,
    )

    # Remove unnecessary columns, leaving only the formatted_text column
    dataset = dataset.remove_columns(["answers", "context", "question"])
    return dataset


# load dataset
dataset = preprocess_squad_dataset(dataset_name="squad", split="train")
valid_dataset = preprocess_squad_dataset(dataset_name="squad", split="validation")
train, validation = dataset, valid_dataset


# create a tokenizer function
def tokenize_function(example, max_context_length=512, max_question_length=32):
    # Combine context and question
    # Tokenize input (context + answer)
    inputs = tokenizer(
        example["inputs"],
        max_length=(max_context_length),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        example["target"],
        max_length=max_question_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}


# tokenize dataset
tokenized_dataset_train = train.map(
    tokenize_function,
    batched=True,
    num_proc=32,
    remove_columns=["inputs", "target", "title", "id"],
)

tokenized_dataset_validation = validation.select(range(1000)).map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["inputs", "target", "title", "id"],
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_validation,
    # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
)

# save model
trainer.train()
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
