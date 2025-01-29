#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "SAKTEHPOPT"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100|H100|L40S"
module load python/3.12.7
module load cuda
source ~/myenvs/cuda-torch/bin/activate
python3 ./tuning_sakt.py E