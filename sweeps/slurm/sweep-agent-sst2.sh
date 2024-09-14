#!/bin/bash
#SBATCH --job-name=sst2
#SBATCH --output=sweeps/logs/sst2-%j.log
#SBATCH --partition=page
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28000
#SBATCH --time=4:00:00

source "$HOME"/.bashrc
pyenv activate babylm  # activate virtual environment

# SST-2
wandb agent --count 1 polargeese/babylm2-finetune-sweeps/okf11v09
