#!/bin/bash
#SBATCH --job-name=wsc-final
#SBATCH --output=sweeps/logs/wsc-final-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=2:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# WSC
#wandb agent --count 1 polargeese/babylm2-finetune-sweeps/7ft96ztf

# Final run with the optimal parameters
python finetune_classification.py \
    --model_name_or_path ../baby-llama2/results/SmolLlama-345M-2_teachers \
    --output_dir ../baby-llama2/results/SmolLlama-345M-2_teachers/results/finetune/wsc \
    --train_file evaluation_data/glue_filtered/wsc.train.jsonl \
    --validation_file evaluation_data/glue_filtered/wsc.valid.jsonl \
    --do_train True \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --bf16 True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 24 \
    --patience 25 \
    --weight_decay 0.4 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --seed 12
