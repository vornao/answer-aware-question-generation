"""Train T5 model on a given dataset, for answer aware question generation task. (Basically a sequence to sequence task)"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import gc
# set torch not to use 0th GPU. use only 1st, 2nd, 3rd, GPU.
import datasets
import transformers
from transformers import Trainer
from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, TaskType, get_peft_model
import torch
import accelerate

DEEPSPEED_CONFIG = "/storagenfs/l.miglior/answer-aware-question-generation/configs/t5deepspeed.json"


TOKEN_QUESTION = '<question>'
TOKEN_END_QUESTION = '<question>'
TOKEN_CONTEXT = '<context>'
TOKEN_END_CONTEXT = '<context>'
TOKEN_ANSWER = '<answer>'
TOKEN_END_ANSWER = '<answer>'  
SPLIT_SEED = 42
NPROC = 32


model_name = "t5-base"
deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=1, offload_optimizer_device='cpu', offload_param_device='cpu')
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision='fp16')

training_args = transformers.TrainingArguments(
    output_dir=f'./results/{model_name}',       # output directory
    num_train_epochs=10,                        # total number of training epochs
    per_device_train_batch_size=16,             # batch size per device during training
    per_device_eval_batch_size=8,               # batch size for evaluation
    warmup_steps=128,                           # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                          # strength of weight decay
    logging_dir=f'./logs/{model_name}',         # directory for storing logs
    do_eval=True,                               # do evaluation
    fp16=True,                                  # use mixed precision trainin
    report_to='tensorboard',
    logging_steps=10,
    eval_steps=50,
    evaluation_strategy='steps',
    deepspeed=DEEPSPEED_CONFIG,
)

# this works fine with deepspeed.
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v"],
)

# Load Models from pretrained
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
gc.collect()
torch.cuda.empty_cache()


# set torch not to use 0th GPU. use only 1st, 2nd, 3rd, GPU.
def preprocess_squad_dataset(dataset_name='squad', split='train'):
    dataset = datasets.load_dataset(dataset_name, split=split)
    # Add question, answer and context tokens to dataset in a new column named text
    dataset = dataset.map(
        lambda e: {
            # answer + context
            'inputs': f'generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT}', 
            # question
            'target': f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}'},
            num_proc=NPROC
        )
    
    # Remove unnecessary columns, leaving only the formatted_text column
    dataset = dataset.remove_columns(['answers', 'context', 'question'])
    return dataset

# load dataset
dataset = preprocess_squad_dataset(dataset_name='squad', split='train')
valid_dataset = preprocess_squad_dataset(dataset_name='squad', split='validation')
train, validation = dataset, valid_dataset


# create a tokenizer function
def tokenize_function(example, max_context_length=512, max_question_length=32):
# Combine context and question
    # Tokenize input (context + answer)
    inputs = tokenizer(example['inputs'], max_length=(max_context_length), return_tensors="pt", padding="max_length", truncation=True)
    labels = tokenizer(example['target'], max_length=max_question_length, return_tensors="pt", padding="max_length", truncation=True)
    return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

# tokenize dataset
tokenized_dataset_train = train.map(
    tokenize_function,
    batched=True,
    num_proc=32,
    remove_columns=['inputs', 'target', 'title', 'id'],
)

tokenized_dataset_validation = validation.select(range(1000)).map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=['inputs', 'target', 'title', 'id'],
)


trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_validation,
    #callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
))

# save model
trainer.train()
model.save_pretrained(f"./models/{model_name}")
tokenizer.save_pretrained(f"./models/{model_name}")
