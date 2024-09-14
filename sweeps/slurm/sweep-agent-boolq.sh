#!/bin/bash
#SBATCH --job-name=boolq
#SBATCH --output=sweeps/logs/boolq-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=2:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# BoolQ
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/hvdg52f7
