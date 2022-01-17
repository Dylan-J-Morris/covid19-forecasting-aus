# COVID-19 Forecasting for Australia
University of Adelaide model used to forecast COVID-19 cases in Australia for the Australian Health Protection Principal Committee (AHPPC). This model uses a hierarchical bayesian approach to state level to inference of effective reproductions numbers over time using macro and micro social distancing data. The model based estimates of $R_\text{eff}$ are referred to as Transmission Potential (TP) which are then used in a branching process model to select valid TP trajectories and produce a forecast distribution of cases over an arbitrary forecast period.

## Data needed
In the `data` folder, ensure you have the latest:
* Case data (NINDSS) (**Note:** as of 6/1/2022, the file in Mediaflux is in a different format and is large enough that we cannot use the preexisting code for reading the raw NINDSS file. This will need to be fixed at some point. )
* Up to date microdistancing survey files titled `Barometer wave XX compliance.csv` saved in the `data/md/` folder. All files up to the current wave need to be included.
* Up to date mask wearing survey files titled `face_covering_XX_.csv` saved in the `data/face_coverings/` folder. All files up to the current wave need to be included.
* [Google mobility indices](https://www.google.com/covid19/mobility/); use the Global CSV with the file named `Global_Mobility_Report.csv`. This is automatically downloaded when `download_google_automatically` in `params.py` is set to `True` (**Note:** this is not normally a good idea to leave on as it slows things down considerably).
* Vaccination coverage data by state. This is required for forecasts using third wave data and onwards. This is a time series of the (multiplicative) reduction in $R_\text{eff}$ as a result of vaccination. **Note:** that this file will most often need the date to be adjusted due to a small offset. This adjustment is just renaming the file with the same date as per NNDSS/interim linelist. 

These data will need to be updated every week. 

## Outline of the model 
The model can be broken down into two components 
1. The TP model fitting and forecasting found in `TP_model`; and, 
2. the branching process simulation found in `generative_model`.

`TP_model` contains Python scripts for running EpyReff to get an estimate of the time-varying reproduction number. This directory also contains code for fitting the TP model to the inferred $R_\text{eff}$. There is also code for forecasting mobility, micro-distancing, mask-wearing and vaccination forwards in time and combining the posterior draws with these forecasted estimates to obtain a forecast of the local (and import) TP. 

`generative_model` features Julia scripts for running the branching process. Using Julia provides noticeable improvements for runtime compared to Python implementations (used previously for this project and are featured in the previous repositories linked at the bottom of this .md file). 
## Model setup and options 

### Data
1. In the covid forecasting directory (from github) create a data folder called `data`. 
2. Create folder for the microdistancing surveys called `md`. This needs to contain `Barometer wave XX compliance.csv` files up to the current wave. 
3. Create folder for the mask wearing surveys called `face_coverings`. This needs to contain `face_covering_XX_.csv` files up to the current wave. 
4. Download latest NNDSS data or linelist from Mediaflux. Put in `/data`.
5. Put `vaccine_effect_timeseries_xxxx-xx-xx.csv` in `/data`. Rename this to have the same file date as the NNDSS data.
6. Download Google mobility data from https://www.google.com/covid19/mobility/ and put in `/data`.

### Required Python/Julia packages
To run the TP model component of the code you will need `matplotlib pandas numpy arviz pystan pyarrow fastparquet seaborn tables tqdm scipy pytables` installed in Python. For older versions of Python, you can use `stan` instead of `pystan`. This can be triggered by setting `on_phoenix=True` in `params.py`.

For the generative model, you will need to have Julia installed. The packages used can all be installed by running:
```
julia generative_model/install_pkgs.jl
```

### Model options
There are some options used within the model that are not passed as parameters. These are all found in the `TP_model/params.py` file. Additionally, options/assumptions have been made during the fitting in `TP_model/fitting_and_forecasting/generate_posterior.py`. Before running either workflow, ensure the flags in the `model/params.py` file are set accordingly. Typically this will involve setting `on_phoenix=True` to either true (if using HPC) of `False` (if running locally), setting `run_inference=True`, `testing_inference=False` and `run_inference_only=False`. The latter two flags are used to save time by not plotting in the stan fitting part of `generate_posterior.py`. 

The `on_phoenix` flag tells the model to use a slightly older version of Pystan and Python. This does not influence any results but latter versions of Pystan required slightly more postprocessing of the file. This is implemented if `on_phoenix=False`.

For the generative model, the assumptions are stored in `generative_model/assumptions.jl`. 

### Scenario modelling
Scenario modelling in the context of this model relates to the assumptions we apply to the forecasting of mobility and microdistancing measures, as well as vaccination. Different scenarios allow us to capture different behaviours of the populations under study and enable a more realistic outlook of the epidemic over the forecast horizon. These scenarios are implemented in `TP_forecasting.py` and can be easily extended. The scenarios themselves are set in `scenarios.py`. This file enables us to set jurisdiction based scenarios as well as separate dates to apply these scenarios. If the scenario and scenario dates are left empty then standard forecasting occurs. 

## Running forecasts

Run these at the command line. Number of sims is used to name some of the files. These lines provide the VoC flag as well as the scenario. Note that scenario date is only of importance for particular situations and acts only as an identifier for a no-reversion to baseline scenario. 
```
DATADATE='2022-01-11'   # Date of NNDSS data file
NUM_THREADS=4
NSIMS=1000               # Total number of simulations to run should be > 5000
```

Then to run the pipeline
```
python TP_model/EpyReff/run_estimator.py $DATADATE
python TP_model/fit_and_forecast/generate_posterior.py $DATADATE
python TP_model/fit_and_forecast/forecast_TP.py $DATADATE
julia -t $NUM_THREADS generative_model/run_forecasts_all_states.jl $DATADATE $NSIMS
```

## Running on a HPC that uses slurm (**Note:** currently untested for the julia model )
This is the quick start to run the model on a HPC that uses slurm.
```
jid_estimator=$(sbatch --parsable sbatch_run_scripts/phoenix_run_estimator.sh ${DATADATE})
jid_posteriors_a=$(sbatch --parsable --dependency=afterok:$jid_estimator sbatch_run_scripts/phoenix_run_posteriors.sh ${DATADATE})
jid_TP_a=$(sbatch --parsable --dependency=afterok:$jid_posteriors_a sbatch_run_scripts/phoenix_TP_forecasting.sh ${DATADATE})
jid_simulate_a_seeding=$(sbatch --parsable --dependency=afterok:$jid_TP_a sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE} 'False')
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a_seeding sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
jid_simulate_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a_seeding sbatch_run_scripts/phoenix_all_states.sh ${NSIMS} ${DATADATE} 'True')
jid_savefigs_and_csv_a=$(sbatch --parsable --dependency=afterok:$jid_simulate_a sbatch_run_scripts/phoenix_final_plots_csv.sh ${NSIMS} ${DATADATE})
```

## Original Code
An earlier version of this code is available at [https://github.com/tobinsouth/covid19-forecasting-aus](https://github.com/tobinsouth/covid19-forecasting-aus). For the original codebase, see [https://github.com/tdennisliu/covid19-forecasting-aus](https://github.com/tdennisliu/covid19-forecasting-aus). This code has been restructured and deprecated functions and files have been removed. There are also some functionalities in the current version of the code which are implemented in very different ways and/or non-existent in the previous code bases. Naming conventions have also changed rather dramatically. For historic versions of the code check the previous repositories.  
