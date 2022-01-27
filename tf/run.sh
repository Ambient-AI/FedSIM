#!/bin/bash

#SBATCH --job-name=fedlearn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH ==time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8


METHOD=$1
DATASET=$2
COMMS=${3}

srun python ~/Development/HF2-Meta/main.py \
    --method $METHOD \
    --dataset $DATASET \
    --comm_round $COMMS \
    --verbose 1