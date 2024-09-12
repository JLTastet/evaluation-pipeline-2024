#!/usr/bin/env python
# coding=utf-8

import argparse
from pathlib import Path

print('NOTE: some choices are hardcoded in this script, so make sure to review it.')

if not Path('finetune_classification.py').exists():
    raise(ValueError('This file must be executed within the same directory as finetune_classification.py'))

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='Path to the model to fine-tune')
parser.add_argument('--output', type=str, default='./slurm', help='Path where the batch scripts will be stored')
parser.add_argument('--seed', type=int, default=12, help='Random seed')
parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')
parser.add_argument('--job_name', type=str, default='finetune', help='SLURM job name')
parser.add_argument('--partition', type=str, default='gr10_gpu,page', help='SLURM partition(s) to use')
parser.add_argument('--max_memory', type=int, default=28000, help='Allocated RAM for each job')
parser.add_argument('--time_limit', type=str, default='36:00:00', help='Timeout for the jobs')
parser.add_argument('--nice', type=int, default=0, help='Base nice value for the jobs (default 0)')
args = parser.parse_args()

if args.model_path is None:
    raise(ValueError('Please specify the path to the model using the --model_path option'))

MODEL_PATH = Path(args.model_path).resolve()
OUTPUT_PATH = Path(args.output).resolve()
OUTPUT_PATH.mkdir(exist_ok=True)
SEED = args.seed
MAX_SEQ_LENGTH = args.max_seq_length

TASKS = ['boolq', 'cola', 'mnli', 'mnli-mm', 'mrpc', 'multirc', 'qnli', 'qqp', 'rte', 'sst2', 'wsc']

# Suggested hyperparameters for the challenge (same for all tasks), but with a lower batch size
DEFAULT_HYPERPARAMETERS = {
    task: {'learning_rate': 5e-5, 'batch_size': 32, 'max_epochs': 10, 'patience': 3, 'eval_every': 1, 'seed': SEED}
    for task in TASKS
}

# The hyperparameters used for the 58M model in the 2023 edition
HYPERPARAMETERS_2023 = {
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

HYPERPARAMETERS = DEFAULT_HYPERPARAMETERS
#HYPERPARAMETERS = HYPERPARAMETERS_2023 # For last year’s submission

# For pyenv
ENV_ACTIVATION = '''\
source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment\
'''
# Add your activation for conda if needed

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
        nice_value = args.nice + 10 # Lower priority, since it depends on mnli
        print('Remember to set mnli as a dependency of mnli-mm using: sbatch --dependency=afterok:<MNLI-JOB-ID> path/to/slurm-finetune-mnli-mm.sh')
    else:
        train_name = task
        valid_name = task
        do_train = True
        model_path_full = MODEL_PATH
        nice_value = args.nice

    executable_path = Path('.').resolve() / 'finetune_classification.py'
    train_file = Path('.').resolve() / 'evaluation_data' / 'glue_filtered'/ f'{train_name}.train.jsonl'
    validation_file = Path('.').resolve() / 'evaluation_data' / 'glue_filtered' / f'{valid_name}.valid.jsonl'

    cmdline = f'''\
python "{executable_path}" \\
    --model_name_or_path "{model_path_full}" \\
    --output_dir "{task_path}" \\
    --train_file "{train_file}" \\
    --validation_file "{validation_file}" \\
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

    partition_lines = '\n'.join([f'#SBATCH --partition={part}' for part in args.partition.split(',')])

    batch_script = f'''\
#!/bin/bash
#SBATCH --job-name={args.job_name}
#SBATCH --output=../logs/{args.job_name}-%j.log
{partition_lines}
#SBATCH --gres=gpu:1
#SBATCH --nice={nice_value}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={args.max_memory}
#SBATCH --time={args.time_limit}

{ENV_ACTIVATION}

{cmdline}
'''

    batch_script_path = OUTPUT_PATH / f'slurm-finetune-{task}.sh'
    batch_script_path.write_text(batch_script)

print(f'The batch scripts have been generated in {args.output}. Please review and modify them as needed.')
