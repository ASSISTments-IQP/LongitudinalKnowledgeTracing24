#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./DKT*; do
  sbatch "$fn"
  done
