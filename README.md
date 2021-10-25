# COVID-19 Forecasting for Australia
University of Adelaide model used to forecast COVID-19 cases in Australia for the Australian Health Protection Principal Committee (AHPPC). This model uses a hierarchical bayesian approach to state level to inference of effective reproductions numbers over time using marco and micro social distancing data. These inferred temporal $R_{eff}$ values are then used in a branching process model to select valid $R_{eff}$ trajectories and produce a forecast distribution of cases over an arbitrary forecast period.

### Data needed
In the `data` folder, ensure you have the latest:
* Case data (NNDSS)
* Up to date microdistancing survey files titled `Barometer wave XX compliance.csv` saved in the `data/md/` folder. All files up to the current wave need to be included.

Extra data:
* [Google mobility indices](https://www.google.com/covid19/mobility/); use the Global CSV with the file named `Global_Mobility_Report.csv`. This is automatically downloaded when `download_google_automatically` in `params.py` is set to True.
* Vaccination coverage data by state. This is required for later forecasts and when `use_vaccine_effect` in `params.py` is set to True.

These data will need to be updated every week. 

#### Internal options
There are some options used within the model that are not passed as parameters. These are all found in the `model/params.py` file. Additionally, options/assumptions have been made during the fitting in `model/fitting_and_forecasting/generate_posterior.py`. 
# Running workflows
## What you need to know
There are two ways to run the UoA Covid-19 forecasting model built into the codebase: 
- locally
- on a cluster (HPC) that uses slurm
  
In this markdown document we outline the requirements for both and provide the straightforward approaches for scenario modelling (the main source of interest currently).

**Note:** For normal (no scenario) modelling, you do not supply the scenario or scenario date to the functions/sbatch scripts. 

## Compiling the model code
The simulation model is written using Cython which means that in order to compile the model a C-compiler is needed. To compile `sim_class_cython.pyx` locally,
```
python model/sim_model/sim_class_cython_setup.py
```
and on HPC,
```
module load arch/haswell
module load Python/3.6.1-foss-2016b
source /hpcfs/users/$USER/local/virtualenvs/bin/activate
python model/sim_model/sim_class_cython_setup.py
```
which creates a shared object and this is what is referenced in `run_state.py`. This builds the shared object and stores it in `/model/sim_model/` (Note that there will be some warnings when building this but they relate to building Numpy under cython and can be ignored. There will also be some additional files produced but they are just the compiled C-code). The model in `sim_class_cython.pyx` is mostly written in python and should be relatively straightforward to understand. The real performance gains come from using Cython on the `generate_cases` function which results in an approximate 4x speedup over base Python implementation. 

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
There are some options used within the model that are not passed as parameters. These are all found in the `model/params.py` file. Additionally, options/assumptions have been made during the fitting in `model/fitting_and_forecasting/generate_posterior.py`. Before running either workflow, ensure the flags in the `model/params.py` file are set accordingly. Typically this will involve setting `on_phoenix=True` to either true (if using HPC) of `False` (if running locally), setting `run_inference=True`, `testing_inference=False` and `run_inference_only=False`. The latter two flags are used to save time by not plotting in the stan fitting part of `generate_posterior.py`. 

The `on_phoenix` flag tells the model to use a slightly older version of Pystan and Python. This does not influence any results but latter versions of Pystan required slightly more postprocessing of the file. This is implemented if `on_phoenix=False`.

Run these at the command line. Number of sims is used to name some of the files. These lines provide the VoC flag as well as the scenario. Note that scenario date is only of importance for particular situations and acts only as an identifier for a no-reversion to baseline scenario. 
## Required arguments
```
DATADATE='2021-10-25'   # Date of NNDSS data file
NSIMS=100000               # Total number of simulations to run should be > 5000
```

## Quick run: Local
```
python model/EpyReff/run_estimator.py $DATADATE
python model/fitting_and_forecasting/generate_posterior.py $DATADATE 
python model/fitting_and_forecasting/forecast_TP.py $DATADATE
states=("NSW" "VIC")
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for STATE in "${states[@]}"
do
    python model/sim_model/run_state.py $NSIMS $DATADATE $STATE
done
python model/record_sim_results/collate_states.py ${NSIMS} ${DATADATE}
python model/record_sim_results/record_to_csv.py ${NSIMS} ${DATADATE}
```

## Quick run: Phoenix
```
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors_a=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
jid_TP_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_TP_forecasting.sh ${DATADATE})
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_TP_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE})
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
```

## Running the model locally 
1. Run EpyReff. If using the linelist you need to set `use_linelist=True` in `params.py`.
```
python model/EpyReff/run_estimator.py $DATADATE
```
2. Run the stan model to link parameters with the EpyReff estimates.
```
python model/fitting_and_forecasting/generate_posterior.py $DATADATE 
```
3. Uses the posterior sample to generate $R_L$ forecasts. 
```
python model/fitting_and_forecasting/forecast_TP.py $DATADATE
```
4. Now we loop over each state and simulate forward. 
```
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for STATE in "${states[@]}"
do
    python model/sim_model/run_state.py $NSIMS $DATADATE $STATE 
done
```
5. Now we use the outputs and produce all the forecast plots. 
```
python model/record_sim_results/collate_states.py ${NSIMS} ${DATADATE} 
```
6. Finally we record the csv to supply for the ensemble. 
```
python model/record_sim_results/record_to_csv.py ${NSIMS} ${DATADATE}
```

## Running the model on a HPC that uses slurm
1. Run EpyReff.
```
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
```
2. Run the stan fitting. This links the mobility and microdistancing measures with the EpyReff TP estimates.
```
jid_posteriors_a=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
```
3. Using the output from the stan fitting, we forecast the TP's forward. This is done by forecasting each mobility measure forward based on a particular mobility scenario. 
```
jid_TP_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_TP_forecasting.sh ${DATADATE})
```
4. This step is the meaty part of the forecasts. This will run the branching process simulation for each state.
```
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_TP_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE})
```
5. Once the simulations have completed we process the output files, plot the resulting forecasts and produce a csv to go into the ensemble.
```
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
```

## Tips and tricks 

### Merging scenario forecast files
If you have different scenarios for different states, provided there is the same number of sims in each file, you can rename the scenario such that they are all the same and this allows you to collate them and produce a single plot. 

### Running just simulations 
If running **JUST** the sims.
```
jid_simulate_b=$(sbatch --parsable sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE})
jid_savefigs_and_csv_b=$(sbatch --parsable --dependency=afterok:$jid_simulate_b sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
```

### Running a single state
```
one_state=$(sbatch --parsable sbatch_run_scripts/phoenix_one_state.sh ${STATE} ${NSIMS} ${DATADATE})
```

## Original Code
Earlier version of this code are available at [https://github.com/tobinsouth/covid19-forecasting-aus](https://github.com/tobinsouth/covid19-forecasting-aus). For the original codebase, see [https://github.com/tdennisliu/covid19-forecasting-aus](https://github.com/tdennisliu/covid19-forecasting-aus). This code has been restructured and deprecated functions and files have been removed. For older code check the other repositories.  
