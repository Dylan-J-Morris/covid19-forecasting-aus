import sys
sys.path.insert(0,'model')
from params import start_date, num_forecast_days

import pandas as pd
from sys import argv
import json

states = ['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA', 'ACT', 'NT']
# states = ['NSW', 'VIC']

n_sims = int(argv[1])  # number of sims

end_date = pd.to_datetime(argv[2], format="%Y-%m-%d") + pd.Timedelta(days=num_forecast_days)
days = (end_date - pd.to_datetime(start_date, format="%Y-%m-%d")).days

forecast_type = 'R_L'

dic_states = {
    'state': [],
    'date': [],
    'type': [],
    'bottom': [],
    'lower': [],
    'lower10': [],
    'upper10': [],
    'lower15': [],
    'upper15': [],
    'lower20': [],
    'upper20': [],
    'median': [],
    'upper': [],
    'top': [],
}
dates = pd.date_range(start=start_date, periods=days)
vars_l = ['symp_inci_obs', 'imports_inci_obs', 'asymp_inci_obs',
          'symp_inci', 'asymp_inci', 'imports_inci', 'total_inci', 'total_inci_obs']
good_sims_by_state = {}
for state in states:

    try: 
        df_file = pd.read_parquet(
            "./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"_seed_days_"+str(days)+".parquet")
    except:
        df_file = pd.read_parquet(
            "./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"days_"+str(days)+".parquet")
        
    # take only the good sims for plotting
    df = df_file.loc[df_file.bad_sim == 0]
    df = df_file[[col.strftime('%Y-%m-%d') for col in dates]]
    
    for var in vars_l:
        quantiles = df.loc[var].quantile(
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95], axis=0)
        dic_states['state'].extend([state]*len(dates))
        dic_states['date'].extend(df.columns)
        dic_states['type'].extend([var]*len(dates))
        dic_states['bottom'].extend(quantiles.loc[0.05])
        dic_states['lower10'].extend(quantiles.loc[0.1])
        dic_states['upper10'].extend(quantiles.loc[0.9])
        dic_states['lower15'].extend(quantiles.loc[0.15])
        dic_states['upper15'].extend(quantiles.loc[0.85])
        dic_states['lower20'].extend(quantiles.loc[0.2])
        dic_states['upper20'].extend(quantiles.loc[0.8])
        dic_states['lower'].extend(quantiles.loc[0.25])
        dic_states['median'].extend(quantiles.loc[0.50])
        dic_states['upper'].extend(quantiles.loc[0.75])
        dic_states['top'].extend(quantiles.loc[0.95])

    # grab sim numbers of good sims
    good_sims_by_state[state] = df_file.loc[df_file.bad_sim == 0].index.get_level_values("sim").unique().tolist()
    
plots = pd.DataFrame.from_dict(dic_states)
plots.to_parquet('./results/quantiles'+forecast_type+start_date + "sim_"+str(n_sims)+"days_"+str(days)+".parquet")

with open("./results/good_sims"+str(n_sims)+"days_"+str(days)+".json", 'w') as file:
    json.dump(good_sims_by_state, file)

import forecast_plots