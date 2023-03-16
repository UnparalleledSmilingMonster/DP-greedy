#!/bin/bash

#SBATCH -n 300
#SBATCH --mem-per-cpu=12000
#SBATCH --time=1:30:00
#SBATCH --job-name=expes_pre_last
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=2-3

srun -W 4800 -n 300 python3.10 learn_compute_entropy_binary_rl_dt_trees.py --expe_id=${SLURM_ARRAY_TASK_ID}