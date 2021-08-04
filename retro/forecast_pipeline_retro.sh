#!/bin/bash

DATADATE=$1 # Date of NNDSS data file
NSIMS=$2 # Total number of simulations to run

jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
echo "Fitting", $jid_estimator, $jid_posteriors

# Delta simulations
jid_delta1=$(sbatch --parsable --dependency=afterok:$jid_posteriors sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE} )
jid_delta2=$(sbatch --parsable --dependency=afterok:$jid_delta1 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE} )
echo "Running", $jid_delta1, $jid_delta2