#!/bin/bash

#SBATCH -n 250
#SBATCH --mem-per-cpu=15000
#SBATCH --time=2:00:00
#SBATCH --job-name=expes_pre_last
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0-5

srun -W 7200 -n 250 python3.10 learn_compute_entropy_binary_rl_dt_trees.py --expe_id=${SLURM_ARRAY_TASK_ID}