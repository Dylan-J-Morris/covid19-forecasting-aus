#!/bin/bash

DATADATE=$1 # Date of NNDSS data file
NSIMS=$2 # Total number of simulations to run
SCENARIO='no_reversion'
SCENARIODATE='2021-07-05' # This doesn't matter for a no-reversion scenario

# Assumes you've already run an EpyReff for the date. If not, uncomment the following line.
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})

# We split the scenario params into the type and the date. It will apply the sec
jid_posteriors_a=$(sbatch --parsable sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE} ${SCENARIO} ${SCENARIODATE})

# Here the scenario parameter is just a filename extention.
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")

# You need to make sure every state has a results/STATEDATE_sim_R_L_daysDelta[SCENARIO].parquet file. You can just rename parquets for states that aren't relevant. collate states doesn't work with a single state.
jid_savefigs_and_csv=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")