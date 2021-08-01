# Simulation modelling

Simulation modelling is when we use the forecasting code but fix the mobility forecasts to levels that would be chosen by future policy. 

Edit the `scenario_pipeline.sh` file with the scenarios and make sure that `model/cprs/generate_RL_forecasts.py` has the correct conditions. You can then run the scenario via,
```bash simulation_modelling/scenario_pipeline.sh ${DATADATE} ${NSIMS}```