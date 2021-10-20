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
source /hpcfs/users/$USER/local/virtualenvs/bin/activate

DATADATE=$1

python model/fitting_and_forecasting/generate_posterior.py $DATADATE

deactivate