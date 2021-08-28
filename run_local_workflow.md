# Running the local workflow

## 0. Data
### Step 0.1
In the covid forecasting directory (from github) create a data folder called `data`. 
### Step 0.2
Create folder for the microdistancing surveys called `md`. This contains `Barometer wave XX compliance.csv` files up to the current wave. 
### Step 0.3
Download latest NNDSS data or linelist from Mediaflux. Put in `\data`.
### Step 0.4
Put `vaccine_effect_timeseries.csv` in `\data`.
### Step 0.5
Download Google mobility data from https://www.google.com/covid19/mobility/ and put in `\data`.

## 1. Terminal

Run these in terminal. Number of sims is used to name some of the files.

These lines provide the VoC flag as well as the scenario and are run at the command line. 
```
DATADATE='2021-08-23'   # Date of NNDSS data file
NSIMS=1000             # Total number of simulations to run should be > 5000
VOCFLAG='Delta'
SCENARIO='no_reversion'
# set date of scenario. Does not matter for no-reversion and is just used to name files. 
SCENARIODATE='2021-08-16'       
```

## 2. Running the Python code
### Step 2.1

Easiest to run in a virtual environment of Python3. Otherwise a standard Python3 install should work fine. 

Need to `pip install matplotlib pandas numpy arviz pystan pyarrow fastparquet seaborn tables tqdm scipy`.

*Note*: I think this is all the dependencies but there might be one or two more. Might be best to wait till you see that each part is running before leaving it alone. Particularly relevant for the simulation part (Step 2.5).

Each of these lines needs to be run one at a time. Just be in the covid forecasting directory and run each one. 
### Step 2.2

Run EpyReff. If using the linelist you need to set `use_linelist=True` in `params.py`.
```
python model/EpyReff/run_estimator.py $DATADATE
```
### Step 2.3
Run the stan model to link parameters with the EpyReff estimates.
```
python model/cprs/generate_posterior.py $DATADATE 
```

### Step 2.4
Uses the posterior sample to generate $R_L$ forecasts. 
```
python model/cprs/generate_RL_forecasts.py $DATADATE $SCENARIO $SCENARIODATE
```

### Step 2.5
Now we loop over each state and simulate forward. 
```
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
states=("NSW" "VIC" "QLD")
states=("TAS" "WA" "ACT" "NT")
for state in "${states[@]}"
do
    python model/run_state.py $NSIMS $DATADATE $state $VOCFLAG "${SCENARIO}${SCENARIODATE}"
done
```

### Step 2.6
Now we use the outputs and produce all the forecast plots. 
```
python model/collate_states.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
```

### Step 2.7
Finally we record the csv to supply for the ensemble. 

*Note*: the record to csv bit is still buggered and I am trying to fix it. 
```
python model/record_to_csv.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
```

