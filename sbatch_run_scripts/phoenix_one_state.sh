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
module load Python/3.6.1-foss-2016b
source ../virtualenvs/bin/activate

STATE=$1 # Pre-pass the single state
NSIMS=$2
NDAYS=$3
DATADATE=$4
STARTDATE=$5
VOCFLAG=$6 # Optional VoC Name
SCENARIO=$7 # Optional scenario modelling flag


python model/run_state.py $NSIMS $NDAYS $DATADATE $STATE $STARTDATE $VOCFLAG $SCENARIO

deactivate
