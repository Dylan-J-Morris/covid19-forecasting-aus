# COVID-19 Forecasting for Australia
University of Adelaide model used to forecast COVID-19 cases in Australia for the Australian Health Protection Principal Committee (AHPPC). This model uses a hierarchical bayesian approach to state level to inference of effective reproductions numbers over time using macro and micro social distancing data. The model based estimates of $R_\text{eff}$ are referred to as Transmission Potential (TP) which are then used in a branching process model to select valid TP trajectories and produce a forecast distribution of cases over an arbitrary forecast period.

## Data needed
In the `data` folder, ensure you have the latest:
* Case data (NINDSS).
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

`generative_model` features Julia scripts for running the branching process. Using Julia provides noticeable improvements for runtime compared to Python implementations (used previously for this project and are featured in the previous repositories linked at the bottom of this file). 
## Model setup and options 

### Data
1. In the covid forecasting directory (from github) create a data folder called `data`. 
2. Create folder for the microdistancing surveys called `md`. This needs to contain `Barometer wave XX compliance.csv` files up to the current wave. 
3. Create folder for the mask wearing surveys called `face_coverings`. This needs to contain `face_covering_XX_.csv` files up to the current wave. 
4. Download latest NNDSS data or linelist from Mediaflux. Put in `/data`.
5. Put `vaccine_effect_timeseries_xxxx-xx-xx.csv` in `/data`. Rename this to have the same file date as the NNDSS data.
6. Download Google mobility data from https://www.google.com/covid19/mobility/ and put in `/data`.

### Required Python/Julia packages
To run the TP model component of the code you will need `matplotlib pandas numpy arviz cmdstanpy pyarrow fastparquet seaborn tables tqdm scipy pytables` installed in Python. The model uses Stan in the `cmdstanpy` framework. The previous implementation of the code used Pystan but due to some recent changes in the Stan ecosystem and for consistency across platforms, we've moved to a more stable framework. Cmdstanpy (> v2.28) offers across chain parralelism resulting in dramatic runtime improvements over the Pystan implementation. Upon installation of `cmdstanpy`, from within Python, run:
```
cmdstanpy.install_cmdstan()
```
This can take some time to run as it installs and links all C++ libraries required for the latest stable Cmdstan build. On MacOSX whenever the command line tools are updated (Xcode) then Cmdstan will fail to build. In these cases Cmdstanpy should prompt the appropriate direction but running (inside Python):
```
cmdstanpy.rebuild_cmdstan()
```
will relink all C++ libraries and ensure Cmdstan is working as intended. 

For the generative model, you will need to have Julia installed. Julia is available at:
https://julialang.org/downloads/
and the code is currently implemented using v1.7.1 (which has some improvements in random number generation). Install is easy to handle and the advantage of Julia is that the code can be run effortlessly across systems. The packages used in the model can all be installed by running:
```
julia generative_model/install_pkgs.jl
```
In the future we may package the generative model code but for ease of use we currently supply a very naieve approach for installation.

### Model options
There are some options used within the model that are not passed as parameters. These are all found in the `TP_model/params.py` file. Additionally, options/assumptions have been made during the fitting in `TP_model/fitting_and_forecasting/generate_posterior.py`. 

For the generative model, the assumptions are stored in `generative_model/forecast_types.jl`. This file features custom types for various components of the model. See this for details. 

### Scenario modelling
Scenario modelling in the context of this model relates to the assumptions we apply to the forecasting of mobility and microdistancing measures, as well as vaccination. Different scenarios allow us to capture different behaviours of the populations under study and enable a more realistic outlook of the epidemic over the forecast horizon. These scenarios are implemented in `TP_forecasting.py` and can be easily extended. The scenarios themselves are set in `scenarios.py`. This file enables us to set jurisdiction based scenarios as well as separate dates to apply these scenarios. If the scenario and scenario dates are left empty then standard forecasting occurs. 

## Running forecasts

These are the options required for routine running of the forecasts:
```
DATADATE='2022-02-01'   # Date of NNDSS data file
NUM_THREADS=4
NSIMS=1000               # Total number of simulations to run should be > 5000
POST_RUN_FLAG=1
```
`POST_RUN_FLAG` enables us to run the full inference setup, only the inference itself, or just the plotting by setting to 1, 2, or 3 respectively.

To run the pipeline:
```
python TP_model/EpyReff/run_estimator.py $DATADATE 
python TP_model/fit_and_forecast/generate_posterior.py $DATADATE $POST_RUN_FLAG
python TP_model/fit_and_forecast/forecast_TP.py $DATADATE
julia -t $NUM_THREADS generative_model/run_forecasts_all_states.jl $DATADATE $NSIMS
```

## Running on a HPC that uses slurm
Currently the model is not tested on the UoA HPC (Slurm). This is due to a recent suite of changes made to facilitate the increased case loads. We are currently working on getting the components of the model back up and appropriately running on a cluster. 

## Original Code
An earlier version of this code is available at [https://github.com/tobinsouth/covid19-forecasting-aus](https://github.com/tobinsouth/covid19-forecasting-aus). For the original codebase, see [https://github.com/tdennisliu/covid19-forecasting-aus](https://github.com/tdennisliu/covid19-forecasting-aus). This code has been restructured and deprecated functions and files have been removed. There are also some functionalities in the current version of the code which are implemented in very different ways and/or non-existent in the previous code bases. Naming conventions have also noticeably changed so direct comparisons are difficult. For historic versions of the code check the previous repositories.  
