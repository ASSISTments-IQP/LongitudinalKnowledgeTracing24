#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./DKT*.sh; do
  sbatch "$fn"
  done
