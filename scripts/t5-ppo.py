import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
from trl import PPOConfig, PPOTrainer

"""
Train fine tuned T5 model with Proximal Policy Optimization (PPO) algorithm.
"""
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="t5-base")
parser.add_argument("--highlight", type=bool, default=True)

args = parser.parse_args()


bertscore = evaluate.load_metric("bertscore")
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



def reward_function(prediction, example):
    """
    this function will return a reward function for PPO
    """

    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0]

    reward = bertscore.compute(
        predictions=[prediction],
        references=[question],
        lang="en",
        model_type="bert-base-uncased",
    )["f1"].item()

    repetition_penalty = -1.0 if answer.lower() in prediction.lower() else 1.0
    question_word_penalty = -0.5 if question.split()[0].lower() != prediction.split()[0].lower() else 0.5
    ans_in_question_penalty = -1.0 if answer.lower() in question.lower() else 1.0
    question_length_penalty = -0.5 if len(question.split()) > average_question_length else 0.5

    reward += repetition_penalty + question_word_penalty + ans_in_question_penalty + question_length_penalty
    return reward

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


def preprocess_squad_dataset(dataset_name="squad", split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split)
    # Add question, answer and context tokens to dataset in a new column named text
    dataset = dataset.map(
        lambda e: {
            # answer + context
            # changed to 'query' for PPO
            "query": f'generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT}',
            # question
            "target": f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}',
        },
        num_proc=NPROC,
    )

    return dataset


HIGHLIGHT = args.highlight
model_name = args.model_name

if HIGHLIGHT:
    model_name = f"{model_name}-hl"


model = transformers.T5ForConditionalGeneration.from_pretrained(
    f"./models/{model_name}/", device_map="auto"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(f"./models/{model_name}")
torch.cuda.empty_cache()

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


config = PPOConfig(
    learning_rate=1e5,
    log_with='tensorboard'
)


ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)