#!/bin/bash
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --mem=128g
#SBATCH -J "MODEL_TYPECYYEAR"
#SBATCH -o "MODEL_TYPECYYEAR.out"
#SBATCH -e "BIG_ERROR.out"
#SBATCH -p short
#SBATCH -t 24:00:00
module load python
source ~/myenvs/lkt-env/bin/activate
python3 ../run_cy_shallow_one_year.py MODEL_TYPE YEAR
