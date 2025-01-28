#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "MODEL_TYPECYYEAR-SAMPLE_NUM"
#SBATCH -o "MODEL_TYPECYYEAR-SAMPLE_NUM.out"
#SBATCH -e "BIG_ERROR.out"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100|H100"
module load cuda
source ~/myenvs/cuda-torch/bin/activate
python3 ../run_cy_deep_one_fold.py MODEL_TYPE YEAR SAMPLE_NUM