#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./SAKT-KC; do
  sbatch "$fn"
  done
