from datasets import load_dataset
import random
import argparse

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', type=str, default='./dataset/stackoverflow-paired', help='saved dataset output dir')
parser.add_argument('-p', '--num_proc', type=int, default=32, help='numbers of processors to use')
parser.add_argument('-s', '--partition_size', default=100000, help='size of each set ', type=int)

args = parser.parse_args()
CPUS = args.num_proc
OUTPUT_DIR = args.output_dir
SIZE = int(args.partition_size)

print(f'Output dir set as{OUTPUT_DIR}, using {CPUS} processors')

try:
    train_set = load_dataset("lvwerra/stack-exchange-paired", split="train")
    test_set = load_dataset("lvwerra/stack-exchange-paired", split="test")
except:
    print("download the StackExchange-Paired dataset from HuggingFace!")
    exit()

# shuffle the dataset
train_set = train_set.shuffle(seed=42)
test_set = test_set.shuffle(seed=42)
try:
    train_set_filtered = train_set.filter(lambda s: 'Stackoverflow' in s['metadata'][0], num_proc=CPUS)
    test_set_filtered = test_set.filter(lambda s: 'Stackoverflow' in s['metadata'][0], num_proc=CPUS)
    train_set_filtered = train_set_filtered.remove_columns(['metadata', 'date'])
    test_set_filtered  = test_set_filtered.remove_columns(['metadata', 'date'])
except Exception as e:
    print(e)
    exit()

rm_set = train_set_filtered.select([i for i in range(SIZE)])
sft_set = train_set_filtered.select([i for i in range(SIZE, SIZE*2)])
rlhf_set = train_set_filtered.select([i for i in range(SIZE*2, SIZE*3)])
test_set = test_set_filtered.select([i for i in range(SIZE)])

# saving to disk
train_set_filtered.select([i for i in range(300000)]).save_to_disk(f'{OUTPUT_DIR}/train', num_proc=CPUS)
rm_set.save_to_disk(f'{OUTPUT_DIR}/rm', num_proc=CPUS)
sft_set.save_to_disk(f'{OUTPUT_DIR}/sft', num_proc=CPUS)
rlhf_set.save_to_disk(f'{OUTPUT_DIR}/rlhf', num_proc=CPUS)
test_set.save_to_disk(f'{OUTPUT_DIR}/test', num_proc=CPUS)

# push to huggingface hub


