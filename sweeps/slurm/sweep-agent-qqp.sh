#!/bin/bash
#SBATCH --job-name=qqp
#SBATCH --output=sweeps/logs/qqp-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=8:00:00
#SBATCH --nice=10

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# QQP (initial sweep)
#wandb agent --count 1 polargeese/babylm2-finetune-sweeps/ijerenc8

# QQP (extended)
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/sbr9siw7
