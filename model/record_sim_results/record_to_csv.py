import sys
sys.path.insert(0,'model')
from params import start_date, num_forecast_days
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sys import argv

states = ['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA', 'ACT', 'NT']
n_sims = int(argv[1])

forecast_type = 'R_L'  # default None
forecast_date = argv[2]  # format should be '%Y-%m-%d'
end_date = pd.to_datetime(forecast_date, format="%Y-%m-%d") + \
    pd.Timedelta(days=num_forecast_days)
days = (end_date - pd.to_datetime(start_date, format="%Y-%m-%d")).days

end_date = pd.to_datetime(
    start_date, format='%Y-%m-%d') + timedelta(days=days-1)
sims_dict = {
    'state': [],
    'onset date': [],
}


for n in range(n_sims):
    if n < 2000:
        sims_dict['sim'+str(n)] = []

print("forecast up to: {}".format(end_date))
date_col = [day.strftime('%Y-%m-%d')
            for day in pd.date_range(start_date, end_date)]

for i, state in enumerate(states):

    # try to read the data with seeding first and then read the non-seeding forecasts 
    try: 
        df_results = pd.read_parquet(
            "./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"days_"+
            str(days)+".parquet", columns=date_col
        )
    except:
        df_results = pd.read_parquet(
            "./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"_seed_days_"+
            str(days)+".parquet", columns=date_col
        )

    df_local = df_results.loc['total_inci_obs']

    sims_dict['onset date'].extend(date_col)
    sims_dict['state'].extend([state]*len(date_col))
    n = 0
    print(state)
    for index, row in df_local.iterrows():
        if n == 2000:
            break
        # if index>=2000:
        #    continue
        # else:
        if np.all(row.isna()):
            continue
        else:
            sims_dict['sim'+str(n)].extend(row.values)
            n += 1
    print(n)
    while n < 2000:
        print("Resampling")
        for index, row in df_local.iterrows():
            if n == 2000:
                break
            if np.all(row.isna()):
                continue
            else:
                sims_dict['sim'+str(n)].extend(row.values)
                n += 1


df = pd.DataFrame.from_dict(sims_dict)
df["data date"] = forecast_date

key = 'local_obs'
df[df.select_dtypes(float).columns] = df.select_dtypes(float).astype(int)
df.to_csv('./results/UoA_'+forecast_date+str(key)+'.csv')
