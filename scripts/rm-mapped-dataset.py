# This script will be used to map a function that will convert our dataset for giving it to the reward model in the right format

import os
import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser

@dataclass
class ScriptArguments:
    dataset_name: str = field(default="../dataset/stackoverflow-paired/rm", metadata={"help": "the dataset name"})
    model_name: str = field(default="./results/sft-eval/checkpoint-7600", metadata={"help": "the model name"}) #modified
    
    DEBUG: bool = field(default=False, metadata={"help": "whether to visualize what's going on"})
    output_dir: str = field(default="./dataset/stackoverflow-paired/rm-tokenized", metadata={"help": "the output directory"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


dataset = load_from_disk(args.dataset_name)
original_columns = dataset.column_names
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset['train']
test_dataset =  dataset['test']


tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


def preprocess_dataset(examples):
    
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    
    for chosen, rejected in zip(examples["response_j"], examples["response_k"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        
        new_examples["input_ids_j"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_rejected["attention_mask"])
    
    return new_examples

def map_tokenize_save(example):

    # use preprocess_dataset to tokenize the data and set in the right format
    new_examples = preprocess_dataset(example)
    
    # save to disk the converted using HuggingFace
    new_examples = Dataset.from_dict(new_examples)
    new_examples.save_to_disk(args.output_dir)


if not args.DEBUG:

    map_tokenize_save(train_dataset) 
 
# This brokes the data... I don't know why
#
#    # map the function to the dataset
#    train_dataset = train_dataset.map(
#        map_tokenize_save,
#        batched=True,
#        num_proc = 16
#    )
#
#    test_dataset = test_dataset.map(
#        map_tokenize_save,
#        batched=True,
#        num_proc = 16,
#    )
#


if args.DEBUG:

    # load from disk
    new_examples = load_from_disk(args.output_dir)

    # check if data are good
    print(new_examples[0])