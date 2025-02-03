#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "SAKTTEST"
#SBATCH -p short
#SBATCH -t 24:00
#SBATCH --gres=gpu:1
module load python/3.12.7
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ./test_sakt.py
