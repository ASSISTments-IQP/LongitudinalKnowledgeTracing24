#!/bin/bash
cd ./job_start_scripts
pwd

for fn in ./BKT*; do
  sbatch "$fn"
  done

for fn in ./PFA*; do
  sbatch "$fn"
  done