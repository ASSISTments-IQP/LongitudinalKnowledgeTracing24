#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "SAKTHPOPT"
#SBATCH -p long
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:4
module load python/3.12.7
module load cuda
source ~/myenvs/cuda-torch/bin/activate
python3 ./tuning_sakt.py