
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sys import argv

states = ['NSW','QLD','SA','TAS','VIC','WA','ACT','NT']
n_sims = int(argv[1])
start_date = '2020-12-01'
days = int(argv[2])
forecast_type = argv[3] #default None

try:
    forecast_date = argv[4] #format should be '%Y-%m-%d'
except:
    forecast_date = datetime.strftime(datetime.today(),format='%Y-%m-%d')
end_date = pd.to_datetime(start_date,format='%Y-%m-%d') + timedelta(days=days-1)
sims_dict={
    'state': [],
    'onset date':[],
}

# If no VoC specified, code will run without alterations.
VoC_name_flag = ''
if len(argv)>5:
    if argv[5] == 'UK':
        VoC_name_flag = 'VoC'
        print('VoC being used in record_to_csv.py')

for n in range(n_sims):
    if n <2000:
        sims_dict['sim'+str(n)] = []

print("forecast up to: {}".format(end_date))
date_col = [day.strftime('%Y-%m-%d') for day in pd.date_range(start_date,end_date)]

for i,state in enumerate(states):
    
    df_results = pd.read_parquet("./results/"+state+start_date+"sim_"+forecast_type+str(n_sims)+"days_"+str(days)+VoC_name_flag+".parquet",columns=date_col)
    
    df_local = df_results.loc['total_inci_obs']

    sims_dict['onset date'].extend(date_col)
    sims_dict['state'].extend([state]*len(date_col))
    n=0
    print(state)
    for index, row in df_local.iterrows():
        if n==2000:
            break
        #if index>=2000:
        #    continue
        #else:
        if np.all(row.isna()):
            continue
        else:
            sims_dict['sim'+str(n)].extend(row.values)
            n +=1
    print(n)
    while n < 2000:
        print("Resampling")
        for index, row in df_local.iterrows():
            if n==2000:
                break
            if np.all(row.isna()):
                continue
            else:
                sims_dict['sim'+str(n)].extend(row.values)
                n +=1


df = pd.DataFrame.from_dict(sims_dict)
df["data date"] = forecast_date

key ='local_obs'
df[df.select_dtypes(float).columns] = df.select_dtypes(float).astype(int)
df.to_csv('./analysis/UoA_'+forecast_date+str(key)+VoC_name_flag+'.csv')