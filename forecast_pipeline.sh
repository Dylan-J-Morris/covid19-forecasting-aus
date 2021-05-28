#!/bin/bash

STARTDATE=$1 # Start date of forecast
DATADATE=$2 # Date of NNDSS data file
NDAYS=$3 # Number of day from start date to forecast
NSIMS=$4 # Total number of simulations to run

jid1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1 sbatch_run_scripts/run_posteriors.sh ${DATADATE})


jid4=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1,$jid2 sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE})
jid5=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid4 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE})
echo "Normal Run:", $jid4, $jid5

jid6=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid5 sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} UK)

jid7=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid6 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} UK)
echo "VoC Run:", $jid6, $jid7
