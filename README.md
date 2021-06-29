# COVID-19 Forecasting for Australia
University of Adelaide model used to forecast COVID-19 cases in Australia for the Australian Health Protection Principal Committee (AHPPC). This model uses a hierarchical bayesian approach to state level to inference of effective reproductions numbers over time using marco and micro social distancing data. These inferred temporal $R_{eff}$ values are then used in a branching process model to select valid $R_{eff}$ trajectories and produce a forecast distribution of cases over an arbitrary forecast period.

## Quickstart: Using HPC and slurm
If you have access to HPC (High performance cluster) that uses slurm, then you can use the following bash script to run the full pipeline, provided your data is stored correctly.

### Data needed
In the `data` folder, ensure you have the latest:
* Case data (NNDSS)
* [Google mobility indices](https://www.google.com/covid19/mobility/); use the Global CSV with the file named `Global_Mobility_Report.csv`
* Up to date microdistancing survey files titled `Barometer wave XX compliance.csv` saved in the `data/md/` folder. All files up to current wave need to be included.

These will need to be updated every week. 

### Running model on slurm
Once all the data are in their corresponding folders, you can run this command to run the full pipeline on HPC:

```
STARTDATE='2020-12-01' # Start date of forecast
DATADATE='2021-06-28'  # Date of NNDSS data file
NDAYS=35 # Number of days after data date to forecast (usually 35)
NSIMS=20000 # Total number of simulations to run

bash forecast_pipeline.sh ${STARTDATE} ${DATADATE} ${NDAYS} ${NSIMS}
```

## Step-by-step workflow and relevant scripts
Below is a breakdown of the pipeline from case line list data to producing forecasts using this repository. This can be used if you don't have access to a slurm HPC.

1. Cori et al. (2013) and Thompson et al. (2019) method to infer an effective reproduction number $R_{eff}$ from case data. Requires:
    * case data in data folder
   ```
   python model/EpyReff/run_estimator.py $DATADATE
   ```
2. Inference of parameters to produce an effective reproduction number locally $R_L$ from $R_{eff}$ and case data. Requires:
    * Google mobility indices
    * Micro-distancing surveys
    * Cori et al. (2013) $R_{eff}$ estimates from 1.
    * case data
   ```
    python model/cprs/generate_posterior.py $DATADATE 
   ```
3. Forecasting Google mobility indices and microdistancing trends. Requires:
   * Google mobility indices
   * Micro-distancing surveys
   * Posterior samples from 2.
    ```
    python model/cprs/generate_RL_forecasts.py $DATADATE
    ```
4.  Simulate cases from $R_L$. Code base lives in `sim_class.py`, but executed by scripts listed below. Requires:
    * case data
    * $R_L$ distribution file from 3.
    ```
    states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
    for state in "${states[@]}"
    do
        python model/run_state.py $NSIMS $NDAYS $DATADATE $state $STARTDATE $VOCFLAG 
    done
    ```

* For a single state (eg. VIC)
    ```
    python model/run_state.py $NSIMS $NDAYS $DATADATE <state-initials> $STARTDATE $VOCFLAG  
    ```

5.  Examine simulation of cases and generate figures. 
    * case data
    * simulation files of all states from 4, saved in `results/`.
    
    ```
    python model/collate_states.py $NSIMS $NDAYS $DATADATE $STARTDATE $VOCFLAG 
    ```

6.  Record results into csv file for UoM ensemble model.
    ```
    python model/record_to_csv.py $NSIMS $NDAYS R_L $DATADATE $STARTDATE $VOCFLAG 
    ```

### Variant of Concern Changes
The model can run with a optional Variant of Concern (VoC) flag, which increases the $R_{eff}$ starting from the forecast date. Currently only the B117 (UK) variant is implemented. This increased model is enabled by passing `UK` as the final parameter to `phoenix_all_states.sh` or `phoenix_final_plots_csv.sh`. This is done automatically by `forecast_pipeline.sh`.


### Internal options
Some things don't quite deserve to being a bash params, but you may still want to change. Here are some notes in case they are important.
- In the `read_in_NNDSS` data inside `model/helper_functions.py` you can set the `use_linelist` option to True to replace the NNDSS data with the imputed linelist of cases used elsewhere in the Aus forecasting pipeline.
- In `generate_RL_forecasts.py` there is a optional second argument that can be passed to allow for modelling of spread in different outbreak simulations during a lockdown (e.g. one could compare the outbreak if lockdown was stopped vs when a lockdown continues at a constant rate). This is not used or called during the normal forecasting (and the code snippet may be commented out when not in use).

### Original Code
An earlier version of this code is available at [https://github.com/tdennisliu/covid19-forecasting-aus](https://github.com/tdennisliu/covid19-forecasting-aus). This code has been restructured and deprecated functions and files have been removed. For older code check the other repository. 
