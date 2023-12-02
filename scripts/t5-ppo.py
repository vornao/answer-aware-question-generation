#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import argparse
import json
import gc
import datasets
import transformers
import torch
import evaluate
from tqdm import tqdm
import json
import numpy as np
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from peft import LoraConfig, TaskType, get_peft_model, PeftModelForSeq2SeqLM, PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""
Train fine tuned T5 model with Proximal Policy Optimization (PPO) algorithm.
"""
parser = argparse.ArgumentParser()
#parser.add_argument("--model_name", type=str, default="t5-small")
#parser.add_argument("--highlight", type=bool, default=True)
#args = parser.parse_args()


bertscore = evaluate.load("bertscore")
average_question_length = 10.0

HIGHLIGHT = True
TOKEN_QUESTION = "<question>"
TOKEN_END_QUESTION = "<question>"
TOKEN_CONTEXT = "<context>"
TOKEN_END_CONTEXT = "<context>"
TOKEN_ANSWER = "<answer>"
TOKEN_END_ANSWER = "<answer>"
HIGHLIGHT_ANSWER = "<hl>"
SPLIT_SEED = 42
NPROC = 32

model_name = "t5-base"
HIGHLIGHT = True
if HIGHLIGHT:
    model_name = f"{model_name}-hl"


# In[3]:


peft_config = LoraConfig(
    r=128,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)


#model = AutoModelForSeq2SeqLM.from_pretrained(f"./models/{model_name}/", device_map="auto")
#peft_model = PeftModelForSeq2SeqLM.from_pretrained(model, model_id=f"./models/{model_name}/", config=peft_config, device_map='auto', is_trainable=True)

ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(f"./models/{model_name}/", device_map="cuda:0")
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(f"./models/{model_name}/", device_map="cuda:0")
tokenizer = transformers.AutoTokenizer.from_pretrained(f"./models/{model_name}", model_max_length=512)
torch.cuda.empty_cache()


# In[4]:


tokenize_query = lambda e : tokenizer(e["query"], return_tensors='pt', padding=True, truncation=True).input_ids.squeeze().to('cuda:0')

def tokenize_query(e):
    return tokenizer(e["query"], return_tensors='pt', padding=True, truncation=True).input_ids.squeeze().to('cuda:0')

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

    e['query'] = e.pop('context')

    return {
        # answer + context
        "query": f'generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["query"]} {TOKEN_END_CONTEXT}',
        # question
        "target": f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}',
        "answer": e["answers"]["text"][0],
    }


def preprocess_squad_dataset(dataset_name="squad", split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split).shuffle(42).select(range(10000))  
    # Add question, answer and context tokens to dataset in a new column named text
    dataset = dataset.map(
        get_inputs_target,
        num_proc=16
    )
    return dataset

# Need to have training dataset aligned with PPO input format

train_dataset = preprocess_squad_dataset(dataset_name="squad", split="train") 


# In[5]:


def preprocess_prediction(example):
    """
    this function will preprocess the prediction
    """
    return example.replace("<pad>", "").replace("<unk>", "").replace("</s>", "").replace("question>", "").replace("<question>", "").replace('<', "").strip()

def reward_model(example):
    """
    this function will return a reward function for PPO
    """
    try:
        context = example["query"]
        target = example["target"]
        answer = example["answer"]
        prediction = example["prediction"]
        prediction = [preprocess_prediction(pred) for pred in prediction] if isinstance(prediction, list) else preprocess_prediction(prediction)
        

        if isinstance(target, list):
            target = [preprocess_prediction(ans) for ans in answer]
        else:
            target = preprocess_prediction(target)

        reward = bertscore.compute(
            predictions=[prediction] if isinstance(prediction, str) else prediction,
            references=[target] if isinstance(target, str) else target,
            lang="en",
            model_type="bert-base-uncased",
        )["f1"][0]

        prediction = prediction[0] if isinstance(prediction, list) else prediction
        target = target[0] if isinstance(target, list) else target
        

        repetition_penalty = -5.0 if answer.lower() in prediction.lower() else 1.0
        question_word_penalty = -0.5 if target.split()[0].lower() != prediction.split()[0].lower() else 0.5
        question_length_penalty = -0.5 if len(target.split()) > 20 else 0.5

        reward = reward + (repetition_penalty + question_word_penalty  + question_length_penalty)
        # make it between 0 and 1
        reward = torch.nn.Sigmoid()(torch.tensor(reward))
        print(f"> Ans: {answer}, Q: {prediction}, P: {target}, R: {reward}")
        return reward

    except Exception as e:
        print("WARNING", e)
        return torch.tensor(0.0)


# In[6]:


BATCH_SIZE = 16
config = PPOConfig(
    learning_rate=1e-5,
    log_with='tensorboard',
    project_kwargs={'logging_dir': f'./logs/{model_name}-ppo'},
    batch_size=BATCH_SIZE,
)

ppo_trainer = PPOTrainer(
    model=ppo_model,
    ref_model=ref_model,
    config=config,
    tokenizer=tokenizer,
)


batched_dataset = [[train_dataset[i] for i in range(j, min(j+BATCH_SIZE, len(train_dataset)))] for j in range(0, len(train_dataset), BATCH_SIZE)]
# remove last batch if it is not full
if len(batched_dataset[-1]) != BATCH_SIZE:
    batched_dataset.pop(-1)

for batch in tqdm(batched_dataset):
    query_tensors = [tokenizer(e["query"], return_tensors='pt', padding=True, truncation=True).input_ids.squeeze().to('cuda:0') for e in batch]

    response_tensors = ppo_trainer.generate(query_tensors, max_length=32, early_stopping=True)

    # put response_tensors into batch
    for i in range(len(batch)):
        batch[i]["prediction"] = tokenizer.decode(response_tensors[i], skip_special_tokens=True)

    pipe_outputs = [reward_model(e) for e in batch] 
    rewards = pipe_outputs

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    
    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')

    print('-'.join('' for x in range(100)))

# save model
ppo_trainer.save_pretrained(f"./models/{model_name}-ppo/")


# In[ ]:


# save model 

