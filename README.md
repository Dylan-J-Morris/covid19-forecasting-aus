# COVID-19 Forecasting for Australia
University of Adelaide model used to forecast COVID-19 cases in Australia for the Australian Health Protection Principal Committee (AHPPC). This model uses a hierarchical bayesian approach to state level to inference of effective reproductions numbers over time using marco and micro social distancing data. These inferred temporal $R_{eff}$ values are then used in a branching process model to select valid $R_{eff}$ trajectories and produce a forecast distribution of cases over an arbitrary forecast period.

## Quickstart: Using HPC and slurm
If you have access to HPC (High performance cluster) that uses slurm, then you can use the following bash script to run the full pipeline, provided your data is stored correctly.

### Data needed
In the `data` folder, ensure you have the latest:
* Case data (NNDSS)
* Up to date microdistancing survey files titled `Barometer wave XX compliance.csv` saved in the `data/md/` folder. All files up to the current wave need to be included.

Extra data:
* [Google mobility indices](https://www.google.com/covid19/mobility/); use the Global CSV with the file named `Global_Mobility_Report.csv`. This is automatically downloaded when `download_google_automatically` in `params.py` is set to True.
* Vaccination coverage data by state. This is required for later forecasts and when `use_vaccine_effect` in `params.py` is set to True.

These data will need to be updated every week. 

### Running model on slurm
Once all the data are in their corresponding folders, you can run this command to run the full pipeline on HPC:

```
DATADATE='2021-08-09'  # Date of NNDSS data file
NSIMS=20000 # Total number of simulations to run

bash forecast_pipeline.sh ${DATADATE} ${NSIMS}
```

#### Internal options
There are some options used within the model that are not passed as parameters. These are all found in the `model/params.py` file. Additionally, options/assumptions have been made during the fitting in `model/cprs/generate_posterior.py`. 


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
The model can run with a optional Variant of Concern (VoC) flag, which increases the $R_{eff}$ starting from the forecast date. This increased model is enabled by passing the WHO name as a parameter to `phoenix_all_states.sh` or `phoenix_final_plots_csv.sh`. This is done automatically by `forecast_pipeline.sh`.


### Original Code
Earlier versions of this code are available at [https://github.com/tobinsouth/covid19-forecasting-aus](https://github.com/tobinsouth/covid19-forecasting-aus) and [https://github.com/tdennisliu/covid19-forecasting-aus](https://github.com/tdennisliu/covid19-forecasting-aus). This code has been restructured and deprecated functions and files have been removed. For older code check the other repository. 
