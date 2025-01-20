#!/bin/bash
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --mem=256g
#SBATCH -J "MODEL_TYPECYYEAR"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:10
module load python/3.12.7
module load cuda/11.8.0
source ~/myenvs/cuda-torch/bin/activate
python3 ../run_cy_one_year.py MODEL_TYPE YEAR
