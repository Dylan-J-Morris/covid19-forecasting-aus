# Running workflows
## What you need to know
There are two ways to run the UoA Covid-19 forecasting model built into the codebase: 
- locally
- on a cluster (HPC) that uses slurm
  
In this markdown document we outline the requirements for both and provide the straightforward approaches for scenario modelling (the main source of interest currently).

**Note:** For normal (no scenario) modelling, you do not supply the scenario or scenario date to the functions/sbatch scripts. 

## Compiling the model code
The simulation model is written using Cython which means that in order to compile the model a C-compiler is needed. To compile `sim_class_cython.pyx` run,
```
python model/sim_class_cython_setup.py
```
which creates a shared object and this is what is referenced in `run_state.py`. This builds the shared object and stores it in `/model` (Note that there will be some warnings when building this but they relate to building Numpy under cython and can be ignored. There will also be some additional files produced but they are just the compiled c-code). The model in `sim_class_cython.pyx` is mostly written in python and should be relatively straightforward to understand. The real performance gains come from using Cython on the `generate_cases` function which results in an approximate 4x speedup over base Python implementation. 

## Data
1. In the covid forecasting directory (from github) create a data folder called `data`. 
2. Create folder for the microdistancing surveys called `md`. This needs to contain `Barometer wave XX compliance.csv` files up to the current wave. 
3. Download latest NNDSS data or linelist from Mediaflux. Put in `/data`.
4. Put `vaccine_effect_timeseries.csv` in `/data`.
5. Download Google mobility data from https://www.google.com/covid19/mobility/ and put in `/data`.

## What you need installed
Need `pip install matplotlib pandas numpy arviz pystan pyarrow fastparquet seaborn tables tqdm scipy`.

*Note*: I think this is all the dependencies but there might be one or two more. Might be best to wait till you see that each part is running before leaving it alone. Particularly relevant for the simulation part.

## Model options
There are some options used within the model that are not passed as parameters. These are all found in the `model/params.py` file. Additionally, options/assumptions have been made during the fitting in `model/cprs/generate_posterior.py`. Before running either workflow, ensure the flags in the `model/params.py` file are set accordingly. Typically this will involve setting `on_phoenix=True` to either true (if using HPC) of `False` (if running locally), setting `run_inference=True`, `testing_inference=False` and `run_inference_only=False`. The latter two flags are used to save time by not plotting in the stan fitting part of `generate_posterior.py`. 

The `on_phoenix` flag tells the model to use a slightly older version of Pystan and Python. This does not influence any results but latter versions of Pystan required slightly more postprocessing of the file. This is implemented if `on_phoenix=False`.

Run these at the command line. Number of sims is used to name some of the files. These lines provide the VoC flag as well as the scenario. Note that scenario date is only of importance for particular situations and acts only as an identifier for a no-reversion to baseline scenario. 
```
DATADATE='2021-10-05'   # Date of NNDSS data file
NSIMS=20000             # Total number of simulations to run should be > 5000
VOCFLAG='Delta'
SCENARIO='no_reversion'
# set date of scenario. Does not matter for no-reversion and is just used to name files. 
SCENARIODATE='2021-10-05'       
```

```
python model/EpyReff/run_estimator.py $DATADATE
python model/cprs/generate_posterior.py $DATADATE 
python model/cprs/generate_RL_forecasts.py $DATADATE $SCENARIO $SCENARIODATE
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for STATE in "${states[@]}"
do
    python model/run_state.py $NSIMS $DATADATE $STATE $VOCFLAG "${SCENARIO}${SCENARIODATE}"
done
python model/collate_states.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
```

## Running the model locally 
1. Run EpyReff. If using the linelist you need to set `use_linelist=True` in `params.py`.
```
python model/EpyReff/run_estimator.py $DATADATE
```
2. Run the stan model to link parameters with the EpyReff estimates.
```
python model/cprs/generate_posterior.py $DATADATE 
```
3. Uses the posterior sample to generate $R_L$ forecasts. 
```
python model/cprs/generate_RL_forecasts.py $DATADATE $SCENARIO $SCENARIODATE
```
4. Now we loop over each state and simulate forward. 
```
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for STATE in "${states[@]}"
do
    python model/run_state.py $NSIMS $DATADATE $STATE $VOCFLAG "${SCENARIO}${SCENARIODATE}"
done
```
5. Now we use the outputs and produce all the forecast plots. 
```
python model/collate_states.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
```
6. Finally we record the csv to supply for the ensemble. 
```
python model/record_to_csv.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
```

## Running the model on a HPC that uses slurm
1. Run EpyReff.
```
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
```
2. Run the stan fitting and then generate the TP forecasts from that. 
```
jid_posteriors_a=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE} ${SCENARIO} ${SCENARIODATE})
```
3. This step is the meaty part of the forecasts. This will run the branching process simulation for each state.
```
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")
```
4. Once the simulations have completed we process the output files, plot the resulting forecasts and produce a csv to go into the ensemble.
```
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")
```

```
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors_a=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE} ${SCENARIO} ${SCENARIODATE})
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")
```

```
one_state=$(sbatch --parsable sbatch_run_scripts/phoenix_one_state.sh ${STATE} ${NSIMS} ${DATADATE} Delta "${SCENARIO}${SCENARIODATE}")
```


