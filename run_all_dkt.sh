#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./DKT-KC*; do
  sbatch "$fn"
  done
