#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./*; do
  sbatch "$fn"
  done
