#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128g
#SBATCH -J "MODEL_TYPECYYEARSAMPLE_NUM"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
module load python
module load cuda
source ~/myenvs/lkt-env/bin/activate
python3 ../run_deep_model_cross_year_one_fold.py MODEL_TYPE YEAR SAMPLE_NUM
