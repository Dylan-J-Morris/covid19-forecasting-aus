#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@adelaide.edu.au

module load arch/haswell
module load Python/3.6.1-foss-2016b
source ../virtualenvs/bin/activate



python analysis/cprs/generate_posterior.py $1 #comment out argument for cprs
python analysis/cprs/generate_RL_forecasts.py $1 #as above

deactivate
