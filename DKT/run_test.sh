#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "DKTHPOPT"
#SBATCH -p short
#SBATCH -t 24:00
#SBATCH --gres=gpu:1
module load cuda
source ~/myenvs/cuda-torch/bin/activate
python3 ./main_DKT.py
