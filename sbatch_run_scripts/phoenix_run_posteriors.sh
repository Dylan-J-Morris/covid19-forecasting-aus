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
module load arch/skylake
module load Python/3.8.6
source ../virtualenvs3.8/bin/activate

DATADATE=$1
SCENARIO=$2 # Optional flag to allow for scenario modelling. Not used in normal forecast.
SCENARIODATE=$3

python model/cprs/generate_posterior.py $DATADATE
python model/cprs/generate_RL_forecasts.py $DATADATE $SCENARIO $SCENARIODATE

deactivate