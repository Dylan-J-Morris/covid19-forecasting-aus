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
module load arch/skylake
module load Python/3.8.6
source ../virtualenvs3.8/bin/activate

NSIMS=$1
DATADATE=$2
VOCFLAG=$3
SCENARIO=$4

python model/collate_states.py $NSIMS $DATADATE $VOCFLAG $SCENARIO
python model/record_to_csv.py $NSIMS $DATADATE $VOCFLAG $SCENARIO

deactivate
