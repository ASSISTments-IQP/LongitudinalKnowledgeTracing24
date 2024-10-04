#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -J "WY SAKT1"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
module load python
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ../run_deep_model_within_year_one_fold.py SAKT 20-21 1 
