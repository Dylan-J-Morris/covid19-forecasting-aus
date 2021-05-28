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
module load Python/3.6.1-foss-2016b
source ../virtualenvs/bin/activate

states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")

python run_state.py $1 $2 $3 ${states[$SLURM_ARRAY_TASK_ID]} $4 p $5

deactivate
