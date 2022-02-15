#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:30:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@adelaide.edu.au

# module load arch/haswell (older versions of Julia here)
module load arch/skylake
# load latest Julia module
module load Julia/1.6.0

DATADATE=$1

julia generative_model/run_forecast_plotting.jl $DATADATE