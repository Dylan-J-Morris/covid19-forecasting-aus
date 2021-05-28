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


DATADATE = $1
python model/cprs/generate_posterior.py $DATADATE #comment out argument for cprs
python model/cprs/generate_RL_forecasts.py $DATADATE #as above

deactivate
