#!/bin/bash
#SBATCH --job-name=qnli
#SBATCH --output=sweeps/logs/qnli-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=12:00:00
#SBATCH --nice=10

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# QNLI
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/mzphwgvd
