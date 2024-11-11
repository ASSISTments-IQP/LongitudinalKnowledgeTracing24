#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "DKTHPOPT"
#SBATCH -p academic
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:1
module load python/3.12.7
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ./tuning_dkt.py