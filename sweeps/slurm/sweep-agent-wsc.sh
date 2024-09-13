#!/bin/bash
#SBATCH --job-name=wsc
#SBATCH --output=sweeps/logs/wsc-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28000
#SBATCH --time=2:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# WSC
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/7ft96ztf
