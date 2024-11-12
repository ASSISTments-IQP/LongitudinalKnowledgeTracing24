#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "DKTHPOPT"
#SBATCH -p academic
#SBATCH -t 01:00
#SBATCH --gres=gpu:1
module load python/3.12.7
module load cuda
source cuda/bin/activate
python3 ./test_share.py