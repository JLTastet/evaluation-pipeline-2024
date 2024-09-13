import argparse
from pathlib import Path
from uuid import uuid4
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_train_epochs', type=int)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--lr_scheduler_type', type=str)
parser.add_argument('--warmup_steps', type=int)
parser.add_argument('--trust_remote_code', action='store_true')
parser.add_argument('--seed', type=int)
parser.add_argument('--trained_mnli_path', type=str, default=None)
args = parser.parse_args()

unique_id = str(uuid4())

model_path = Path(args.model_path)
run_path = Path(args.output_dir) / 'finetune' / args.task / unique_id
run_path.mkdir(parents=True, exist_ok=True)

if args.task == 'mnli-mm': # Continue from mnli
    train_name = 'mnli'
    valid_name = 'mnli-mm'
    do_train = False
    model_path_full = args.trained_mnli_path
    if model_path_full is None:
        raise(ValueError('To run MNLI-mm, you need to specify the path to a fine-tuned MNLI checkpoint using --trained_mnli_path'))
else:
    train_name = args.task
    valid_name = args.task
    do_train = True
    model_path_full = model_path

executable_path = Path('.').resolve() / 'finetune_classification.py'
train_file = Path('.').resolve() / 'evaluation_data' / 'glue_filtered'/ f'{train_name}.train.jsonl'
validation_file = Path('.').resolve() / 'evaluation_data' / 'glue_filtered' / f'{valid_name}.valid.jsonl'

cmdline = f'''\
python "{executable_path}" \\
--model_name_or_path "{model_path_full}" \\
--output_dir "{run_path}" \\
--train_file "{train_file}" \\
--validation_file "{validation_file}" \\
--do_train {do_train} \\
--do_eval \\
--do_predict \\
--max_seq_length 128 \\
--bf16 True \\
--evaluation_strategy epoch \\
--save_strategy epoch \\
--learning_rate {args.learning_rate} \\
--per_device_train_batch_size {args.batch_size} \\
--num_train_epochs {args.num_train_epochs} \\
--patience {args.num_train_epochs+1} \\
--weight_decay {args.weight_decay} \\
--lr_scheduler_type {args.lr_scheduler_type} \\
--warmup_steps {args.warmup_steps} \\
--seed {args.seed}\
'''

if args.trust_remote_code:
    cmdline += ' \\\n    --trust_remote_code'

subprocess.run(cmdline, shell=True, text=True)