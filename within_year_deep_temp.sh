#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=256g
#SBATCH -J "MODEL_TYPEWYYEAR"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:5
module load python
module load cuda
source ~/myenvs/cuda-torch/bin/activate
python3 ../run_wy_one_year.py MODEL_TYPE YEAR
