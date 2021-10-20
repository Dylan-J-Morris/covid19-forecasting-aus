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
source /hpcfs/users/$USER/local/virtualenvs/bin/activate

NSIMS=$1
DATADATE=$2
VOCFLAG=$3
SCENARIO=$4

python model/record_sim_results/collate_states.py $NSIMS $DATADATE $VOCFLAG $SCENARIO
python model/record_sim_results/record_to_csv.py $NSIMS $DATADATE $VOCFLAG $SCENARIO

deactivate
