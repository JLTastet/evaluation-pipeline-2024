#!/bin/bash
#SBATCH --job-name=sst2-final
#SBATCH --output=sweeps/logs/sst2-final-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=4:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# SST-2
#wandb agent --count 1 polargeese/babylm2-finetune-sweeps/okf11v09

# Final run with the optimal parameters
python finetune_classification.py \
    --model_name_or_path ../baby-llama2/results/SmolLlama-345M-2_teachers \
    --output_dir ../baby-llama2/results/SmolLlama-345M-2_teachers/results/finetune/sst2 \
    --train_file evaluation_data/glue_filtered/sst2.train.jsonl \
    --validation_file evaluation_data/glue_filtered/sst2.valid.jsonl \
    --do_train True \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --bf16 True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 24 \
    --num_train_epochs 2 \
    --patience 3 \
    --weight_decay 5.0 \
    --lr_scheduler_type constant \
    --warmup_steps 200 \
    --seed 12
