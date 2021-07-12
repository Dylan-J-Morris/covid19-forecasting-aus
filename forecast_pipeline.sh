#!/bin/bash

DATADATE=$1 # Date of NNDSS data file
NSIMS=$3 # Total number of simulations to run

jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
echo "Fitting", $jid_estimator, $jid_posteriors

# Delta simulations
jid_delta1=$(sbatch --parsable --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} Delta)
jid_delta2=$(sbatch --parsable --dependency=afterok:$jid_delta1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} Delta)
echo "Delta", $jid_delta1, $jid_delta2


# Alpha simulations
jid_alpha1=$(sbatch --parsable --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} Alpha)
jid_alpha2=$(sbatch --parsable --dependency=afterok:$jid_alpha1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} Alpha)
echo "Alpha", $jid_alpha1, $jid_alpha2


# Base simulations
jid_base1=$(sbatch --parsable --dependency=afterok:$jid_posteriors,$jid_delta1 sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE})
jid_base2=$(sbatch --parsable --dependency=afterok:$jid_base1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE})
echo "Base", $jid_base1, $jid_base2

# Single state run (example)
# jid_single=$(sbatch --parsable --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_one_state.sh "TAS" ${NSIMS} ${NDAYS} ${DATADATE})
