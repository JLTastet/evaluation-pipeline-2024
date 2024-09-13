#!/bin/bash
#SBATCH --job-name=rte
#SBATCH --output=sweeps/logs/rte-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28000
#SBATCH --time=2:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# RTE
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/svmjaz0r
