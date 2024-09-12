#!/bin/bash
#SBATCH --job-name=zeroshot
#SBATCH --output=../logs/zeroshot-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28000
#SBATCH --time=2:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# Edit the arguments below
MODEL_PATH=../baby-llama2/results/baby-llama-58m/
BATCH_SIZE=128

MODEL_PATH=$(realpath "$MODEL_PATH")

python -m lm_eval --model hf \
    --model_args pretrained="$MODEL_PATH",backend="causal" \
    --tasks ewok_filtered \
    --device cuda:0 \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --trust_remote_code \
    --output_path "$MODEL_PATH"/results/ewok/ewok_results.json

python -m lm_eval --model hf \
    --model_args pretrained="$MODEL_PATH",backend="causal" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --trust_remote_code \
    --output_path "$MODEL_PATH"/results/blimp/blimp_results.json
