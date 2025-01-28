#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./SAKT-E; do
  sbatch "$fn"
  done
