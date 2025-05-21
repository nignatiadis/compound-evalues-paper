#!/bin/bash

#SBATCH --job-name=compound
#SBATCH --array=1-52
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000
#SBATCH --account=pi-ignat
#SBATCH --partition=caslake

# Execute the Julia script, passing the SLURM_ARRAY_TASK_ID as an argument
module load gurobi/11.0
julia run_simulation.jl $SLURM_ARRAY_TASK_ID
