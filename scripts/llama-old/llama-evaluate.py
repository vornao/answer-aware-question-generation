import evaluate
import transformers
import argparse
import torch
import os

from datasets import load_from_disk
from numpy import mean
from tqdm import tqdm
#clear console
os.system('cls' if os.name == 'nt' else 'clear')
HEADER = """
 _ _                       
| | |                      
| | | __ _ _ __ ___   __ _ 
| | |/ _` | '_ ` _ \ / _` |
| | | (_| | | | | | | (_| | 
|_|_|\__,_|_| |_| |_|\__,_|
"""
print(HEADER)
bertscore = evaluate.load('bertscore')
dataset = load_from_disk('./dataset/stackoverflow-paired-11k/sft')

parser = argparse.ArgumentParser(prog="Llama evaluation tool", description="Evaluate llama 2 with bertscore")
parser.add_argument('--model', default='./llama/huggingface')
parser.add_argument('--seed', default=42)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

# create evaluation folder if not exists
if not os.path.exists('./evaluation'):
    print('> Creating evaluation folder...')
    os.mkdir('./evaluation')

# if no model is specified, use the default one and warn the user
if args.model == './llama/huggingface':
    print('> WARNING: No model specified, using default pretrained one.')
    print('> WARNING: You can specify a model with --model <path_to_model>')

nf4_config = transformers.BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

LLAMA_PATH = args.model
# print device
print(f'> Using device {args.device}')
print(f'> Loading model from {LLAMA_PATH}')
print("> Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(LLAMA_PATH)
print("> Tokenizer loaded.\n> Loading model...")
model = transformers.AutoModelForCausalLM.from_pretrained(LLAMA_PATH, device_map=args.device)
print("> Model Loaded.")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map={'':0},
    tokenizer=tokenizer,
)

bert_precision = []
bert_recall = []
bert_f1 = []


outputs: list = []

val_dataset = dataset.train_test_split(test_size=0.05, seed=args.seed)['test'].shuffle(args.seed)

print('> Starting evaluation...')
with tqdm(total=100) as pbar:
    for example in val_dataset.select(range(100)):
        q = example["question"]
        a = example["response_j"]
        sequences = pipeline(
            f'### Question: {q} ### Answer: ',
            do_sample=True,
            top_k=10,
            temperature=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=256
        )
        prediction = str([s['generated_text'] for s in sequences])
        try:
            prediction = [prediction.split('### Answer:')[1]]
        except:
            prediction = [prediction]
        finally:
            metric = bertscore.compute(predictions=prediction, references=[a], lang='en', model_type='bert-large-uncased', device=args.device)
            bert_precision.append([metric['precision']])
            bert_recall.append([metric['recall']])
            bert_f1.append([metric['f1']])
            pbar.set_postfix(Metric=mean(bert_f1))
            pbar.update(1)
            torch.cuda.empty_cache()
        
        # append output to outputs list in the form of a dict with question, prediction, reference
        outputs.append({
            'question': q,
            'prediction': prediction,
            'reference': a
        })

  
# write results to file.
with open(f'./evaluate/bertscore-{LLAMA_PATH}.json', 'w') as f:
    f.write(f'{{"model":{LLAMA_PATH}, "precision": {mean(bert_precision)}, "recall": {mean(bert_recall)}, "f1": {mean(bert_f1)}}},')
    f.write('\n')
    # write outputs to file
    f.write(str(outputs))


