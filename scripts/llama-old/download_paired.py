# We will use this repo from hf
# https://huggingface.co/datasets/lvwerra/stack-exchange-paired

# Download the dataset splits

from datasets import load_dataset
from tqdm import tqdm

# val_set = load_dataset("lvwerra/stack-exchange-paired", split="validation")
# Loading training


# use tqdm to show progress
print("Loading training")
train_set = load_dataset("lvwerra/stack-exchange-paired", split="train")

print("Loading test")
test_set = load_dataset("lvwerra/stack-exchange-paired", split="test")
