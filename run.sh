#!/bin/bash


#SBATCH --job-name=train35
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

SERVER=35
source /home/${USER}/.bashrc
source /home/${USER}/miniconda3/bin/activate
conda activate tff

srun python main.py --server-partition $SERVER