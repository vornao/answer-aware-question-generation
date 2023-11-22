# Train LLama2 checkpoint as Reward Model with deepspeed/accellerate
import os
from dataclasses import dataclass, field
from typing import Optional, Dict

from torch import nn
from accelerate import Accelerator, DeepSpeedPlugin
from datasets import load_from_disk
from peft import LoraConfig, TaskType

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)
from trl import RewardConfig


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="./results/sft-eval/checkpoint-7600",
        metadata={"help": "the model name"},
    )  # modified
    log_with: Optional[str] = field(
        default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"}
    )  # modified
    dataset_name: Optional[str] = field(
        default="./dataset/stackoverflow-paired-11k/rm",
        metadata={"help": "the dataset name"},
    )  # modified
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(
        default=4000, metadata={"help": "the size of the validation set"}
    )
    streaming: Optional[bool] = field(
        default=True, metadata={"help": "whether to stream the dataset"}
    )
    shuffle_buffer: Optional[int] = field(
        default=5000, metadata={"help": "the shuffle buffer size"}
    )
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "the sequence length"}
    )
    num_workers: Optional[int] = field(
        default=4, metadata={"help": "the number of workers"}
    )
    group_by_length: Optional[bool] = field(
        default=True, metadata={"help": "whether to group by length"}
    )
    output_dir: Optional[str] = field(
        default="./results/rm-eval", metadata={"help": "the model checkpoint dir"}
    )
    log_dir: Optional[str] = field(
        default="./logs/rm-eval", metadata={"help": "the model logging dir"}
    )
    # no packing in RewardTrainer code

    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir="../results/rm-eval",
            per_device_train_batch_size=1,
            num_train_epochs=3,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",  # pagaed_adamw_32bit as the sft
            logging_steps=200,
            evaluation_strategy="no",
            logging_dir="./logs/rm-eval",
            max_length=512,
            do_eval=True,
            fp16=True,
        )
    )

    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        ),
    )


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,
    gradient_accumulation_steps=1,
    offload_optimizer_device="cpu",
    offload_param_device="cpu",
)
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")

# Load the dataset with j and k responses

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

dataset = load_from_disk(args.dataset_name)
original_columns = dataset.column_names
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

"""
Expected dataset format for RM training:

Therefore the final dataset object should contain two 4 entries at least if you use the default
RewardDataCollatorWithPadding data collator. The entries should be named:

    input_ids_chosen
    attention_mask_chosen
    input_ids_rejected
    attention_mask_rejected
    
"""
# This is costumizable depending on the dataset, we should change this but i don't know the dataset format


def preprocess_dataset(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    for chosen, rejected in zip(examples["response_j"], examples["response_k"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(
            tokenized_rejected["attention_mask"]
        )

    return new_examples


train_dataset = train_dataset.map(preprocess_dataset, batched=True, num_proc=16)

test_dataset = test_dataset.map(
    preprocess_dataset,
    batched=True,
    num_proc=16,
)


# remove this to take all of them without filtering by length
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)

test_dataset = test_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)


#### Actual RM part ####

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, num_labels=1
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)


class MyRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print(input_ids=inputs["input_ids_j"], input_ids=inputs["input_ids_k"])
        rewards_j = model(
            input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"]
        )[0]
        rewards_k = model(
            input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"]
        )[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


## usual deepspeed stuff
trainer = accelerator.prepare(
    MyRewardTrainer(
        model=model,  # or just model
        tokenizer=tokenizer,
        args=args.reward_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
)

trainer.train()
trainer.save_model(args.reward_config.output_dir)

output_dir = os.path.join(args.output_dir, "final_checkpoint_rm")
trainer.model.save_pretrained(output_dir)
