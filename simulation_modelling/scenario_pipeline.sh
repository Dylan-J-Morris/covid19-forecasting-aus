#!/bin/bash

STARTDATE=$1 # Start date of forecast
DATADATE=$2 # Date of NNDSS data file
NDAYS=$3 # Number of days after data date to forecast
NSIMS=$4 # Total number of simulations to run

# Assumes you've already run an EpyReff for the date.


# We split the scenario params into the type and the date. It will apply the sec
jid_posteriors_a=$(sbatch --parsable simulation_modelling/phoenix_scenario_R_L.sh ${DATADATE} "half_reversion" "2021-07-09")
jid_posteriors_b=$(sbatch --parsable simulation_modelling/phoenix_scenario_R_L.sh ${DATADATE} "half_reversion" "2021-07-16")

# Here the scenario parameter is just a filename extention. We can either run a single state for speed and later rename other parquets to include the scenario name for forecast plots or we can run all of the states.
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_one_state.sh "NSW" ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta "half_reversion2021-07-09")

jid_simulate_b=$(sbatch --parsable --dependency=afterok:$jid_posteriors_b sbatch_run_scripts/phoenix_one_state.sh "NSW" ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta "half_reversion2021-07-16")

# or

jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta "half_reversion2021-07-09")

jid_simulate_b=$(sbatch --parsable --dependency=afterok:$jid_posteriors_b sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta "half_reversion2021-07-16")


# You need to make sure every state has a results/STATEDATE_sim_R_L_daysDelta[SCENARIO].parquet file. You cna just rename parquets for states that aren't relevant. collate states doesn't work with a single state.
jid_savefigs_and_csv=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta "half_reversion2021-07-09")

jid_savefigs_and_csv=$(sbatch --parsable --dependency=afterok:$jid_simulate_b sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} ${STARTDATE} Delta "half_reversion2021-07-16")
