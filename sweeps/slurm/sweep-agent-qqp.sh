#!/bin/bash
#SBATCH --job-name=qqp
#SBATCH --output=sweeps/logs/qqp-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=4:00:00
#SBATCH --nice=10

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# QQP
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/ijerenc8
