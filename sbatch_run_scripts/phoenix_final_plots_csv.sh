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

module load arch/haswell
module load Python/3.6.1-foss-2016b
source ../virtualenvs/bin/activate

NSIMS = $1
NDAYS = $2
DATADATE = $3
STARTDATE = $4
VOCFLAG = $5

python model/collate_states.py $NSIMS $NDAYS $DATADATE $STARTDATE $VOCFLAG 
python model/record_to_csv.py $NSIMS $NDAYS R_L $DATADATE $STARTDATE $VOCFLAG 

deactivate
