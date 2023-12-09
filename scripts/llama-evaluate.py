#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import evaluate
import transformers
import argparse
import torch
import json
import datasets

from numpy import mean
from tqdm import tqdm

TOKEN_QUESTION = "### Question:"
TOKEN_END_QUESTION = ""
TOKEN_CONTEXT = "### Context:"
TOKEN_END_CONTEXT = ""
TOKEN_ANSWER = "### Answer:"
TOKEN_END_ANSWER = ""
HIGHLIGHT_ANSWER = ""
SPLIT_SEED = 43
NPROC = 32
HIGHLIGHT = True
LLAMA_PATH = "./results/llama-7B/checkpoint-1000"


# In[3]:


bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")


# In[4]:


tokenizer = transformers.AutoTokenizer.from_pretrained(LLAMA_PATH)
model = transformers.AutoModelForCausalLM.from_pretrained(LLAMA_PATH, device_map="auto")


# In[5]:


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


# In[16]:


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
        + " "
        + TOKEN_ANSWER
        + " "
        + e["answers"]["text"][0]
        + " "
        + TOKEN_END_ANSWER
    )

    return {
        # answer + context + question for causal language modeling
        "text": f'<s> {e["context"]} {TOKEN_QUESTION}',
    }



def preprocess_squad_dataset(dataset_name="squad", split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split).select(range(1000,2000))
    dataset = dataset.map(get_inputs_target, num_proc=NPROC)
    return dataset


# load dataset
valid_dataset = preprocess_squad_dataset(dataset_name="squad", split="validation")

predictions = []
references = []
contexes = []

for examples in tqdm(valid_dataset.shuffle(SPLIT_SEED)):
    sequences = pipeline(
            examples['text'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_length=512
        )
    prediction = str([s["generated_text"] for s in sequences])
    predictions.append(prediction.split(TOKEN_QUESTION)[1].split('?')[0] + "?")
    references.append(examples["question"])
    contexes.append(examples["context"])
    print("> Prediction:", prediction.split(TOKEN_QUESTION)[1].split('?')[0], "\n" + "> Original:", examples["question"], "\n") 


# compute scores
bertscore.compute(predictions=predictions, references=references, lang="en")
rouge.compute(predictions=predictions, references=references, lang="en")

# save scores
metrics = {
    "bertscore": bertscore.compute(predictions=predictions, references=references, lang="en"),
    "rouge": rouge.compute(predictions=predictions, references=references, lang="en"),
    "predictions": predictions,
    "references": references,
    "contexes": contexes
}

with open(f"results/llama-7B/eval-{SPLIT_SEED}.json", "w") as f:
    json.dump(metrics, f)

