# COVID-19 Forecasting for Australia
University of Adelaide model used to forecast COVID-19 cases in Australia for the Australian Health Protection Principal Committee (AHPPC). This model uses a hierarchical bayesian approach to state level to inference of effective reproductions numbers over time using marco and micro social distancing data. These inferred temporal $R_{eff}$ values are then used in a branching process model to select valid $R_{eff}$ trajectories and produce a forecast distribution of cases over an arbitrary forecast period.

### Data needed
In the `data` folder, ensure you have the latest:
* Case data (NNDSS)
* Up to date microdistancing survey files titled `Barometer wave XX compliance.csv` saved in the `data/md/` folder. All files up to the current wave need to be included.
* [Google mobility indices](https://www.google.com/covid19/mobility/); use the Global CSV with the file named `Global_Mobility_Report.csv`. This is automatically downloaded when `download_google_automatically` in `params.py` is set to True.
* Vaccination coverage data by state. This is required for forecasts using third wave data and onwards. This is a time series of the (multiplicative) reduction in $R_\text{eff}$ as a result of vaccination. **Note:** that this file will most often need the date to be adjusted due to a small offset. This adjustment is just renaming the file with the same date as per NNDSS/interim linelist. 

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
## Data
1. In the covid forecasting directory (from github) create a data folder called `data`. 
2. Create folder for the microdistancing surveys called `md`. This needs to contain `Barometer wave XX compliance.csv` files up to the current wave. 
3. Download latest NNDSS data or linelist from Mediaflux. Put in `/data`.
4. Put `vaccine_effect_timeseries.csv` in `/data`.
5. Download Google mobility data from https://www.google.com/covid19/mobility/ and put in `/data`.

## What you need installed
Need `matplotlib pandas numpy arviz pystan pyarrow fastparquet seaborn tables tqdm scipy`.

**Note:** Think this is all the dependencies but there might be one or two more. Might be best to wait till you see that each part is running before leaving it alone. Particularly relevant for the simulation part.

## Model options
There are some options used within the model that are not passed as parameters. These are all found in the `model/params.py` file. Additionally, options/assumptions have been made during the fitting in `model/fitting_and_forecasting/generate_posterior.py`. Before running either workflow, ensure the flags in the `model/params.py` file are set accordingly. Typically this will involve setting `on_phoenix=True` to either true (if using HPC) of `False` (if running locally), setting `run_inference=True`, `testing_inference=False` and `run_inference_only=False`. The latter two flags are used to save time by not plotting in the stan fitting part of `generate_posterior.py`. 

The `on_phoenix` flag tells the model to use a slightly older version of Pystan and Python. This does not influence any results but latter versions of Pystan required slightly more postprocessing of the file. This is implemented if `on_phoenix=False`.

Run these at the command line. Number of sims is used to name some of the files. These lines provide the VoC flag as well as the scenario. Note that scenario date is only of importance for particular situations and acts only as an identifier for a no-reversion to baseline scenario. 
## Required arguments
```
DATADATE='2021-11-01'   # Date of NNDSS data file
NSIMS=100000               # Total number of simulations to run should be > 5000
```

## Scenario modelling
Scenario modelling in the context of this model relates to the assumptions we apply to the forecasting of mobility and microdistancing measures, as well as vaccination. Different scenarios allow us to capture different behaviours of the populations under study and enable a more realistic outlook of the epidemic over the forecast horizon. These scenarios are implemented in `TP_forecasting.py` and can be easily extended. The scenarios themselves are set in `scenarios.py`. This file enables us to set jurisdiction based scenarios as well as separate dates to apply these scenarios. If the scenario and scenario dates are left empty then standard forecasting occurs. 

## Quick run: Local
This is a quick start to run the model locally. 
```
python model/EpyReff/run_estimator.py $DATADATE
python model/fitting_and_forecasting/generate_posterior.py $DATADATE 
python model/fitting_and_forecasting/forecast_TP.py $DATADATE
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for STATE in "${states[@]}"
do
    python model/sim_model/run_state.py $NSIMS $DATADATE $STATE
done
python model/record_sim_results/collate_states.py ${NSIMS} ${DATADATE}
python model/record_sim_results/record_to_csv.py ${NSIMS} ${DATADATE}
```

## Quick run: HPC using SLURM
This is the quick start to run the model on a HPC that uses slurm.
```
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors_a=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
jid_TP_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_TP_forecasting.sh ${DATADATE})
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_TP_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE})
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
```

## Detailed steps for running the model locally 
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
1. Using the posterior sample, we forecast the TP's forward. This is done by forecasting each mobility measure, the microdistancing survey, and the vaccination effects forward based on a particular social mobility scenario. 
```
jid_TP_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_TP_forecasting.sh ${DATADATE})
```
4. This step is the meaty part of the forecasts. This will run the branching process simulation for each state. An a-priori way to tune the runtime is to run a test batch of 10000 sims and then determine the acceptance rate. This can then be extrapolated to the required number of sims to acheive a representative sample of 2000 accepted simulations. 
```
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_TP_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE})
```
5. Once the simulations have completed we process the output files, plot the resulting forecasts and produce a summary csv to go into the ensemble forecast.
```
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
```

## Tips and tricks 
### Merging forecasts
If you have different number of simulations for different states, the number of sims part of the parquet file name can be renamed. This allows all files to be read in and processed accordingly during simulation.

### Running a single state
Something that occurs occasionally is that a single state needs to be run. This can be due to an incorrect scenario or a low acceptance rate for the simulations. This provides functionality to run a particular state `STATE` in such a situation. 
```
one_state=$(sbatch --parsable sbatch_run_scripts/phoenix_one_state.sh ${STATE} ${NSIMS} ${DATADATE})
```

## Original Code
An earlier version of this code is available at [https://github.com/tobinsouth/covid19-forecasting-aus](https://github.com/tobinsouth/covid19-forecasting-aus). For the original codebase, see [https://github.com/tdennisliu/covid19-forecasting-aus](https://github.com/tdennisliu/covid19-forecasting-aus). This code has been restructured and deprecated functions and files have been removed. There is also a number of elements in the current model which are implemented in a different way or non-existent in the previous code bases. For older code check the other repositories.  
