#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "MODEL_TYPEWYYEAR-FOLD_NUM"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "H200"
#SBATCH -o "MODEL_TYPEWYYEAR-FOLD_NUM.out"
#SBATCH -e "BIG_ERROR.out"
module load cuda
source ~/myenvs/cuda-torch/bin/activate
python3 ../run_wy_deep_one_fold.py MODEL_TYPE YEAR FOLD_NUM