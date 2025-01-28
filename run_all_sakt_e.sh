#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./SAKT-E*.sh; do
  sbatch "$fn"
  done
