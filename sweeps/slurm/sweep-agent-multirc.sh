#!/bin/bash
#SBATCH --job-name=multirc
#SBATCH --output=sweeps/logs/multirc-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=6:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# MultiRC
wandb agent --config 1 polargeese/babylm2-finetune-sweeps/u58fjevx
