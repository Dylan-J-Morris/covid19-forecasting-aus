#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --time=1-00:00:00
#SBATCH --mem=60GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@adelaide.edu.au

module load arch/haswell
module load arch/skylake
module load Python/3.8.6
source ../virtualenvs3.8/bin/activate

STATE=$1 # Pre-pass the single state
NSIMS=$2
DATADATE=$3
VOCFLAG=$4 # Optional VoC Name
SCENARIO=$5 # Optional scenario modelling flag


python model/run_state.py $NSIMS $DATADATE $STATE $VOCFLAG $SCENARIO

deactivate
