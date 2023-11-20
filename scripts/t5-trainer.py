"""Train T5 model on a given dataset, for answer aware question generation task. (Basically a sequence to sequence task)"""

import os
import argparse
import datasets
import transformers
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import BitsAndBytesConfig


TOKEN_QUESTION = '[QST]'
TOKEN_END_QUESTION = '[QST_END]'
TOKEN_CONTEXT = '[CTX]'
TOKEN_END_CONTEXT = '[CTX_END]'
TOKEN_ANSWER = '[ANS]'
TOKEN_END_ANSWER = '[ANS_END]'  
SPLIT_SEED = 42
NPROC = 32

from transformers import TrainerCallback



# set torch not to use 0th GPU. use only 1st, 2nd, 3rd, GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def preprocess_squad_dataset(dataset_name='squad', split='train'):
    """Preprocesses the dataset for the answer aware question generation task. 
    Returns a dataset with the following columns:
        
        formatted_text: The text with the question, answer and context tokens added.
        id: The id of the example.
        title: The title of the example.
        validation: The validation split of the dataset.
        train: The train split of the dataset.
    
    Args:
        dataset_name (str, optional): The name of the dataset to load. Defaults to 'squad'.
        split (str, optional): The split of the dataset to load. Defaults to 'train'.
    
    Returns:
        datasets.DatasetDict: The dataset with the formatted_text column added.
    """
    dataset = datasets.load_dataset(dataset_name, split=split)
    # Add question, answer and context tokens to dataset in a new column named text
    dataset = dataset.map(
        lambda e: {
            'inputs': f'{TOKEN_ANSWER}{e["answers"]["text"][0]} {TOKEN_END_ANSWER} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT}', 
            'target': f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION}'},
            num_proc=NPROC
        )
    
    # Remove unnecessary columns, leaving only the formatted_text column
    dataset = dataset.remove_columns(['answers', 'context', 'question'])
    return dataset


def train_model(dataset_name='squad', model_name='t5-small', batch_size=8, epochs=1, save_model=False, save_model_path=None):
    """Trains a T5 model on a given dataset, for answer aware question generation task. (Basically a sequence to sequence task)
    
    Args:
        dataset_name (str, optional): The name of the dataset to load. Defaults to 'squad'.
        model_name (str, optional): The name of the model to use. Defaults to 't5-small'.
        batch_size (int, optional): The batch size to use. Defaults to 8.
        epochs (int, optional): The number of epochs to train for. Defaults to 1.
        save_model (bool, optional): Whether to save the model or not. Defaults to False.
        save_model_path (str, optional): The path to save the model to. Defaults to None.
    """

    # this works fine with deepspeed.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
    )

    # check if dave model path is given if save_model is True
    if save_model and save_model_path is None:
        raise ValueError('save_model_path must be given if save_model is True.')
    
    # create save_model_path if it doesn't exist
    if save_model and not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    # load dataset
    if dataset_name == 'squad':
        dataset = preprocess_squad_dataset(dataset_name=dataset_name, split='train')
        valid_dataset = preprocess_squad_dataset(dataset_name=dataset_name, split='validation')
    else:
        raise ValueError('Only squad dataset is supported at the moment.')
    
    
    train, validation = dataset, valid_dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # create a tokenizer function
    def tokenize_function(example, max_context_length=300, max_answer_length=32):
    # Combine context and question
        # Tokenize input (context + answer)
        inputs = tokenizer(example['inputs'], max_length=(max_context_length), return_tensors="pt", truncation=True, padding=True)
        labels = tokenizer(example['target'], max_length=max_answer_length, return_tensors="pt", truncation=True, padding=True)
        return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}
    

    # tokenize dataset
    tokenized_dataset_train = train.map(
        tokenize_function,
        batched=True,
        num_proc=32,
     
    )

    tokenized_dataset_validation = validation.map(
        tokenize_function,
        batched=True,
        num_proc=32,
    ).select(range(1000))


    training_args = transformers.TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=128,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        do_eval=True,                    # do evaluation
        fp16=True,                       # use mixed precision trainin
        report_to='tensorboard',
        logging_steps=10,
        eval_steps=50,
        evaluation_strategy='steps',
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_validation,
        #callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(save_model_path)



def main():
    parser = argparse.ArgumentParser(description='Train T5 model on a given dataset, for answer aware question generation task. (Basically a sequence to sequence task)')
    parser.add_argument('--dataset_name', type=str, default='squad', help='The name of the dataset to load. Defaults to squad.')
    parser.add_argument('--model_name', type=str, default='t5-small', help='The name of the model to use. Defaults to t5-small.')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size to use. Defaults to 8.')
    parser.add_argument('--epochs', type=int, default=1, help='The number of epochs to train for. Defaults to 1.')
    parser.add_argument('--save_model', type=bool, default=False, help='Whether to save the model or not. Defaults to False.')
    parser.add_argument('--save_model_path', type=str, default='./models', help='The path to save the model to. Defaults to ./models.')
    args = parser.parse_args()

    train_model(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_model=args.save_model,
        save_model_path=args.save_model_path
    )

if __name__ == '__main__':
    main()