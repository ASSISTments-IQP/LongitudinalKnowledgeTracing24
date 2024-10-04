#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -J "CY DKTFOLD_NUM"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
module load python
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ../run_deep_model_cross_year_one_fold.py DKT 20-21 FOLD_NUM
