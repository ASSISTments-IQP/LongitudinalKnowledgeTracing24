#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "DKTWY22-234"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
module load python
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ../run_deep_model_within_year_one_fold.py DKT 22-23 4 
