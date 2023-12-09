import evaluate
import transformers
import argparse
import torch
import os
import json
import datasets

from numpy import mean
from tqdm import tqdm

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


# clear console
os.system("cls" if os.name == "nt" else "clear")
HEADER = """
 _ _                       
| | |                      
| | | __ _ _ __ ___   __ _ 
| | |/ _` | '_ ` _ \ / _` |
| | | (_| | | | | | | (_| | 
|_|_|\__,_|_| |_| |_|\__,_|
"""
print(HEADER)
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")

parser = argparse.ArgumentParser(
    prog="Llama evaluation tool", description="Evaluate llama 2 with bertscore"
)
parser.add_argument("--model", default="./llama/huggingface")
parser.add_argument("--seed", default=42)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()



LLAMA_PATH = args.model
# print device
print(f"> Using device {args.device}")
print(f"> Loading model from {LLAMA_PATH}")
print("> Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(LLAMA_PATH)
print("> Tokenizer loaded.\n> Loading model...")
model = transformers.AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH, device_map=args.device
)
print("> Model Loaded.")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map={"": 0},
    tokenizer=tokenizer,
)

bert_precision = []
bert_recall = []
bert_f1 = []


outputs: list = []


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
        "text": f'<s> [INST] generate question: {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT} {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER} [/INST] {TOKEN_QUESTION}  <s>',
        "target": e["question"],
    }


def preprocess_squad_dataset(dataset_name="squad", split="train"):
    dataset = datasets.load_dataset(dataset_name, split=split).select(1000,2000)
    dataset = dataset.map(get_inputs_target, num_proc=NPROC)
    dataset = dataset.remove_columns(["answers", "context", "question"])
    return dataset


# load dataset
valid_dataset = preprocess_squad_dataset(dataset_name="squad", split="validation")

predictions = []
targets = []

# generate questions for each example
print("> Starting evaluation...")
with tqdm(total=100) as pbar:
    for example in valid_dataset.select(range(100)):
        q = example["target"]
        sequences = pipeline(
            example['text'],
            do_sample=True,
            top_k=10,
            temperature=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=64,
        )
        prediction = str([s["generated_text"] for s in sequences])
        print("> Prediction:", prediction)
        predictions.append(prediction)
        targets.append(q)


print("> Evaluating bertscore...")
bertscore = bertscore.compute(
    predictions=predictions,
    references=targets,
    lang="en",
    model_type="bert-large-uncased",
    device=args.device,
)

print("> Bertscore results:")
print(bertscore)

print("> Evaluating rouge...")
rouge = rouge.compute(
    predictions=predictions,
    references=targets,
    lang="en",
    model_type="bert-large-uncased",
    device=args.device,
)

print("> Rouge results:")
print(rouge)

print("> Done!")

metrics = {
    "bertscore": bertscore,
    "rouge": rouge,
    "predictions": predictions,
    "targets": targets,
}

with open("./models/llama-7B/metrics.json", "w") as f:
    json.dump(metrics, f)