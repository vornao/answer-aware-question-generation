import argparse
import json
import os
import sys


"""Train T5 model on a given dataset, for answer aware question generation task. (Basically a sequence to sequence task)"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import gc
import datasets
import transformers
import torch
import evaluate
from tqdm import tqdm
import json
import numpy as np

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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="t5-small-hl")
parser.add_argument("--highlight", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="squad")

args = parser.parse_args()

HIGHLIGHT = args.highlight
DATASET_NAME = args.dataset
model_name = args.model_name



model = transformers.T5ForConditionalGeneration.from_pretrained(
    f"./models/{model_name}/", device_map="cuda:0"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(f"./models/{model_name}")
torch.cuda.empty_cache()


def get_inputs_target(e):
    try:
        answer_start = e["answers"]["answer_start"][0]
    except IndexError:
        return {
            "inputs": None,
            "target": None,
        }
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
            + e["context"][answer_start + ans_len:]
        )

    return {
        # answer + context
        "inputs": f'generate question: {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT}',
        # question
        "target": f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}',
    }


def preprocess_squad_dataset(dataset_name=DATASET_NAME, split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split)
    dataset = dataset.map(get_inputs_target, num_proc=NPROC)
    # remove data points where "inputs" is None
    dataset = dataset.filter(lambda e: e["inputs"] is not None)
    dataset = dataset.remove_columns(["answers", "context", "question"])
    return dataset


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


# load dataset
#dataset = preprocess_squad_dataset(dataset_name=DATASET_NAME, split="train")
valid_dataset = preprocess_squad_dataset(dataset_name=DATASET_NAME, split="validation")
validation = valid_dataset.select(range(1000, 2000))

tokenized_dataset_validation = validation.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["inputs", "target", "title", "id"],
)

predictions = []
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def generate_question(example, max_length=32):
    ids = tokenizer.encode(example, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        input_ids=ids, max_length=max_length, num_beams=4, early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_question_generation(dataset, max_length=32):

    for example in tqdm(dataset):
        predictions.append(generate_question(example["inputs"], max_length=max_length))


evaluate_question_generation(validation, max_length=32)

predictions = [
    prediction.replace("question>", "").replace("<question>", "").replace("<hl>", "")
    for prediction in predictions
]
targets = [target.replace("<question>", "") for target in validation["target"]]

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore", device_map="cpu")

bscore = bertscore.compute(predictions=predictions, references=targets, lang="en")
rougescore = rouge.compute(predictions=predictions, references=targets)

print(f"BERTScore: {np.mean(bscore['f1'])}")
print(f"ROUGE: {rougescore['rouge1']}")

metrics = {
    "bertscore": bscore,
    "rouge": rougescore,
    "predictions": predictions,
    "targets": targets,
}

# save results in a json file into ./models/{model_name}/metrics.json

with open(f"./models/{model_name}/metrics-{DATASET_NAME}.json", "w") as f:
    json.dump(metrics, f)
