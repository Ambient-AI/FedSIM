#!/bin/bash


#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
source /home/${USER}/miniconda3/bin/activate
conda activate tff

srun python main.py -ds stackoverflow -sd 0 -uc False -cr 500