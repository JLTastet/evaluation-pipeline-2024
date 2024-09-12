#!/usr/bin/env python
# coding=utf-8

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='Path to the model to fine-tune')
parser.add_argument('--seed', type=int, default=12, help='Random seed')
parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')
args = parser.parse_args()

if args.model_path is None:
    raise(ValueError('Please specify the path to the model using the --model_path option'))

MODEL_PATH = Path(args.model_path)
SEED = args.seed
MAX_SEQ_LENGTH = args.max_seq_length

TASKS = ['boolq', 'cola', 'mnli', 'mnli-mm', 'mrpc', 'multirc', 'qnli', 'qqp', 'rte', 'sst2', 'wsc']

HYPERPARAMETERS = {
    'cola':    {'learning_rate': 4e-5, 'batch_size': 64, 'max_epochs':  3, 'patience':   10, 'eval_every':   20, 'seed': SEED},
    'sst2':    {'learning_rate': 5e-5, 'batch_size': 64, 'max_epochs':  6, 'patience':   10, 'eval_every':  200, 'seed': SEED},
    'mrpc':    {'learning_rate': 3e-5, 'batch_size': 64, 'max_epochs':  3, 'patience':   10, 'eval_every':   20, 'seed': SEED},
    'qqp':     {'learning_rate': 4e-5, 'batch_size': 64, 'max_epochs': 10, 'patience':   10, 'eval_every': 1000, 'seed': SEED},
    'mnli':    {'learning_rate': 5e-5, 'batch_size': 64, 'max_epochs':  6, 'patience':   10, 'eval_every':  200, 'seed': SEED},
    'mnli-mm': {'learning_rate': 5e-5, 'batch_size': 64, 'max_epochs':  6, 'patience':   10, 'eval_every':  200, 'seed': SEED},
    'qnli':    {'learning_rate': 5e-5, 'batch_size': 64, 'max_epochs':  6, 'patience':   10, 'eval_every':  200, 'seed': SEED},
    'rte':     {'learning_rate': 5e-5, 'batch_size': 64, 'max_epochs':  6, 'patience':   10, 'eval_every':  200, 'seed': SEED},
    'boolq':   {'learning_rate': 3e-4, 'batch_size': 16, 'max_epochs': 10, 'patience':   10, 'eval_every':   10, 'seed': SEED},
    'multirc': {'learning_rate': 1e-4, 'batch_size': 64, 'max_epochs':  7, 'patience':   10, 'eval_every': 1000, 'seed': SEED},
    'wsc':     {'learning_rate': 5e-7, 'batch_size':  1, 'max_epochs': 10, 'patience': 1000, 'eval_every': 2000, 'seed': SEED},
}

# Sanity check before starting to fine-tune
for task in TASKS:
    assert task in HYPERPARAMETERS.keys()

for task in TASKS:
    task_path = MODEL_PATH / 'results' / 'finetune' / task
    task_path.mkdir(parents=True, exist_ok=True)

    hp = HYPERPARAMETERS[task]
    
    if task == 'mnli-mm': # Continue from mnli
        train_name = 'mnli'
        valid_name = 'mnli-mm'
        do_train = False
        model_path_full = MODEL_PATH / 'results' / 'finetune' / train_name
    else:
        train_name = task
        valid_name = task
        do_train = True
        model_path_full = MODEL_PATH

    cmdline = f'''\
python finetune_classification.py \\
    --model_name_or_path "{model_path_full}" \\
    --output_dir "{task_path}" \\
    --train_file evaluation_data/glue_filtered/{train_name}.train.jsonl \\
    --validation_file evaluation_data/glue_filtered/{valid_name}.valid.jsonl \\
    --do_train {do_train} \\
    --do_eval \\
    --do_predict \\
    --max_seq_length {MAX_SEQ_LENGTH} \\
    --per_device_train_batch_size {hp['batch_size']} \\
    --learning_rate {hp['learning_rate']} \\
    --num_train_epochs {hp['max_epochs']} \\
    --patience {hp['patience']} \\
    --evaluation_strategy epoch \\
    --save_strategy epoch \\
    --overwrite_output_dir \\
    --seed {hp['seed']}\
'''

    if args.trust_remote_code:
        cmdline += ' \\\n    --trust_remote_code'

    subprocess.run(cmdline, text=True, shell=True)
