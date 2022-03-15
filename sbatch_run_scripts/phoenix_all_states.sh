#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@adelaide.edu.au
#SBATCH --ntasks-per-core=1
#SBATCH --array=0-7

# module load arch/haswell (older versions of Julia here)
module load arch/skylake
# load latest Julia module
module load Julia/1.6.0

states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")

DATADATE=$1

julia generative_model/run_forecasts_single_state.jl $DATADATE ${states[$SLURM_ARRAY_TASK_ID]}