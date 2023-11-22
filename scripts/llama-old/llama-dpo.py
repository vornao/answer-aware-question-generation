# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import deepspeed
import torch

from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from datasets import load_dataset, load_from_disk  # modified
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    Trainer,
    AdamW,
    get_constant_schedule,
)

from trl import SFTTrainer, DPOTrainer
from trl.trainer import ConstantLengthDataset

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="results/sft-eval/checkpoint-7600/", metadata={"help": "the model name"}
    )  # modified
    log_with: Optional[str] = field(
        default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"}
    )  # modified

    dataset_name: Optional[str] = field(
        default="./dataset/stackoverflow-paired-11k/rlhf",
        metadata={"help": "the dataset name"},
    )  # modified
    subset: Optional[str] = field(
        default="data/finetune", metadata={"help": "the subset to use"}
    )
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
        default=768, metadata={"help": "the sequence length"}
    )
    num_workers: Optional[int] = field(
        default=32, metadata={"help": "the number of workers"}
    )
    local_rank: Optional[int] = field(
        default=0, metadata={"help": "the number of workers"}
    )
    log_dir: Optional[str] = field(
        default="logs/dpo-llama", metadata={"help": "the logging directory"}
    )
    epochs: Optional[int] = field(
        default=3, metadata={"help": "the epochs for training"}
    )
    deepspeed: Optional[str] = field(
        default="configs/deepspeed.json", metadata={"help": "the number of workers"}
    )

    max_steps: Optional[int] = field(
        default=500, metadata={"help": "the maximum number of sgd steps"}
    )
    logging_steps: Optional[int] = field(
        default=5, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=10, metadata={"help": "the saving frequency"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per device train batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per device eval batch size"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the gradient accumulation steps"}
    )

    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(
        default=False, metadata={"help": "whether to group by length"}
    )
    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )

    lora_alpha: Optional[float] = field(
        default=8, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(
        default=1e-4, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    num_warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "the weight decay for optimizer"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )

    output_dir: Optional[str] = field(
        default="./results/dpo-eval", metadata={"help": "the output directory"}
    )
    log_freq: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )

    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum length for generation"}
    )
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum length for generation"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    return (
        f"### Question: {example['question']}\n\n ### Answer: {example['response_j']}"
    )


# DEFINE THIS BEFORE CALLING base_model = AutoModelForCausalLM.from_pretrained(...)
# OTHERWISE DEEPSPEED WONT WORK!
# DAMN HUGGINGFACE
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    logging_steps=script_args.logging_steps,
    report_to=script_args.log_with,
    num_train_epochs=script_args.epochs,
    save_steps=400,
    save_strategy="steps",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=True,
    fp16=True,
    gradient_checkpointing=False,
    logging_dir=script_args.log_dir,
)


deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,
    gradient_accumulation_steps=1,
    offload_optimizer_device="cpu",
    offload_param_device="cpu",
)
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")


tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj",
        "fc_in",
        "fc_out",
        "wte",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)


def return_prompt_and_responses(samples):
    return {
        "prompt": [
            "### Question: " + question + "### Answer: "
            for question in samples["question"]
        ],
        "chosen": samples["response_j"],  # rated better than k
        "rejected": samples["response_k"],  # rated worse than j
    }


dataset = load_from_disk("dataset/stackoverflow-paired-11k/rlhf")

dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset, eval_dataset = dataset["train"], dataset["test"]
original_columns = train_dataset.column_names

train_dataset = train_dataset.map(
    return_prompt_and_responses, batched=True, remove_columns=original_columns
).filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.seq_length)

eval_dataset = eval_dataset.map(
    return_prompt_and_responses, batched=True, remove_columns=original_columns
).filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.seq_length)


base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name + "final_merged_checkpoint", torch_dtype=torch.float16
)
reference_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.float16
)


# Finally, call training.
dpo_trainer = accelerator.prepare(
    DPOTrainer(
        model=base_model,
        ref_model=reference_model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        tokenizer=tokenizer,
        args=training_args,
    )
)

dpo_trainer.train()
dpo_trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
