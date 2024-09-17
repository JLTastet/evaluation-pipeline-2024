#!/bin/bash
#SBATCH --job-name=qnli-final
#SBATCH --output=sweeps/logs/qnli-final-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=12:00:00
#SBATCH --nice=0

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# QNLI
#wandb agent --count 1 polargeese/babylm2-finetune-sweeps/mzphwgvd

# Final run with the optimal parameters
#MODEL_PATH=../baby-llama2/results/SmolLlama-345M-2_teachers
#MODEL_PATH=../baby-llama2/models/SmolLlama-345M/47108234
#MODEL_PATH=../baby-llama2/results/SmolLlama-345M-sweep/best-240911-1207
MODEL_PATH=../baby-llama2/models/SmolLlama-345M-2_teachers/47109224
python finetune_classification.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$MODEL_PATH"/results/finetune/qnli \
    --train_file evaluation_data/glue_filtered/qnli.train.jsonl \
    --validation_file evaluation_data/glue_filtered/qnli.valid.jsonl \
    --do_train True \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --bf16 True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 2 \
    --patience 3 \
    --weight_decay 0.3 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
    --seed 12
