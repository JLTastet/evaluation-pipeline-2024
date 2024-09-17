#!/bin/bash
#SBATCH --job-name=zeroshot
#SBATCH --output=results/logs/zeroshot-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=2:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# Edit the arguments below
#MODEL_PATH=../baby-llama2/results/baby-llama-58m/
#MODEL_PATH=../baby-llama2/models/SmolLlama-345M/47108234/
MODEL_PATH=../baby-llama2/models/SmolLlama-345M-2_teachers/47109224
BATCH_SIZE=64
RUN_BLIMP=true
RUN_EWOK=true

MODEL_PATH=$(realpath "$MODEL_PATH")

if $RUN_EWOK; then
    python -m lm_eval --model hf \
        --model_args pretrained="$MODEL_PATH",backend="causal" \
        --tasks ewok_filtered \
        --device cuda:0 \
        --batch_size $BATCH_SIZE \
        --log_samples \
        --trust_remote_code \
        --output_path "$MODEL_PATH"/results/ewok/ewok_results.json
fi

if $RUN_BLIMP; then
    python -m lm_eval --model hf \
        --model_args pretrained="$MODEL_PATH",backend="causal" \
        --tasks blimp_filtered,blimp_supplement \
        --device cuda:0 \
        --batch_size $BATCH_SIZE \
        --log_samples \
        --trust_remote_code \
        --output_path "$MODEL_PATH"/results/blimp/blimp_results.json
fi
