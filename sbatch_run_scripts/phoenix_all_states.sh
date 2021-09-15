#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --time=1-00:00:00
#SBATCH --mem=60GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@adelaide.edu.au
#SBATCH --array=0-7


module load arch/haswell
module load arch/skylake
module load Python/3.8.6
source ../virtualenvs3.8/bin/activate

states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")

NSIMS=$1
DATADATE=$2
VOCFLAG=$3 # Optional VoC Name
SCENARIO=$4 # Optional scenario modelling flag

python model/run_state.py $NSIMS $DATADATE ${states[$SLURM_ARRAY_TASK_ID]} $VOCFLAG $SCENARIO

deactivate
