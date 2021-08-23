# Running the local workflow

```
DATADATE='2021-08-16'  # Date of NNDSS data file
NSIMS=1000 # Total number of simulations to run
```

```
VOCFLAG='Delta'
SCENARIO='no_reversion'
```

```
python model/EpyReff/run_estimator.py $DATADATE
python model/cprs/generate_posterior.py $DATADATE 
python model/cprs/generate_RL_forecasts.py $DATADATE $SCENARIO $SCENARIODATE
states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for state in "${states[@]}"
do
    python model/run_state.py $NSIMS $DATADATE $state $VOCFLAG "${SCENARIO}${SCENARIODATE}"
done
python model/collate_states.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
python model/record_to_csv.py ${NSIMS} ${DATADATE} $VOCFLAG "${SCENARIO}${SCENARIODATE}"
```

