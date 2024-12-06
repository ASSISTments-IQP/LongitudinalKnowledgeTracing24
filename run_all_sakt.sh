#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./SAKT*; do
  sbatch "$fn"
  done
