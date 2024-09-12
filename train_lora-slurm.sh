#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --output=../logs/lora-%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28000
#SBATCH --time=24:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

./my_train_lora.sh ../baby-llama2/results/SmolLlama-345M-best_teacher
