#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "DKTHPOPT"
#SBATCH -p short
#SBATCH -t 01:00
#SBATCH --gres=gpu:2
module load python/3.12.7
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ./gpu_cuda_test.py