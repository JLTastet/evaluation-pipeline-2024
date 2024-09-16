#!/bin/bash
#SBATCH --job-name=mnli-mm
#SBATCH --output=sweeps/logs/mnli-mm-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=1:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

python finetune_classification.py \
    --model_name_or_path ../baby-llama2/results/SmolLlama-345M-2_teachers/results/finetune/mnli/1e-5 \
    --output_dir ../baby-llama2/results/SmolLlama-345M-2_teachers/results/finetune/mnli-mm/1e-5 \
    --train_file evaluation_data/glue_filtered/mnli.train.jsonl \
    --validation_file evaluation_data/glue_filtered/mnli-mm.valid.jsonl \
    --do_train False \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --bf16 True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 6 \
    --patience 7\
    --weight_decay 1.0 \
    --lr_scheduler_type linear \
    --warmup_steps 500 \
    --seed 12
