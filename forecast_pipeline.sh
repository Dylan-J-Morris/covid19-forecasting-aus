#!/bin/bash

STARTDATE=$1 # Start date of forecast
DATADATE=$2 # Date of NNDSS data file
NDAYS=$3 # Number of days after data date to forecast
NSIMS=$4 # Total number of simulations to run

jid_estimator=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
echo "Fitting", $jid_estimator, $jid_posteriors

# Delta simulations
jid_delta1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta)
jid_delta2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_delta1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta)
echo "Delta", $jid_delta1, $jid_delta2


# Alpha simulations
jid_alpha1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Alpha)
jid_alpha2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_alpha1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Alpha)
echo "Alpha", $jid_alpha1, $jid_alpha2


# Base simulations
jid_base1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_posteriors,$jid_delta1 sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE})
jid_base2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_base1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE})
echo "Base", $jid_base1, $jid_base2

# Single state run (example)
# jid_single=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_one_state.sh "TAS" ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE})
