#!/bin/bash

DATE=$1
NDAYS=$2

jid1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1 sbatch_run_scripts/run_posteriors.sh ${DATADATE})


jid4=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1,$jid2 sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATADATE})
jid5=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid4 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE})
echo "Normal Run:", $jid4, $jid5

jid6=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid5 sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${NDAYS} ${DATE} None UK)

jid7=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid6 sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${NDAYS} ${DATADATE} UK)
echo "VoC Run:", $jid6, $jid7
