import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-poster")
from sys import argv

import os,csv
def read_in_cases(cases_file_date=None):
    """
    Read in NNDSS case file data
        
    """
    import pandas as pd
    from datetime import timedelta
    import glob

    if cases_file_date is None:
        import glob, os

        list_of_files = glob.glob("data/"+'COVID-19 UoM*.xlsx') 
        path = max(list_of_files, key=os.path.getctime)
        print("Using file "+path)
    else:
        path = "data/"+"COVID-19 UoM "+cases_file_date+"*.xlsx"

    for file in glob.glob(path):
        df = pd.read_excel(file,
                   parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'],
                   dtype= {'PLACE_OF_ACQUISITION':str})
    if len(glob.glob(path))!=1:
        print("There are %i files with the same date" %len(glob.glob(path)))

        if len(glob.glob(path)) >1:
            print("Using an arbritary file")
    print(path)
    
    df.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) #Fill blanks with simply unknown

    df['date_inferred'] = df.TRUE_ONSET_DATE
    #missing_cases = df.groupby('STATE').TRUE_ONSET_DATE.agg(lambda x: sum(x.isna()))
    #print("Unknown Symptom onset dates")
    #display(missing_cases)
    df.loc[df.TRUE_ONSET_DATE.isna(),'date_inferred'] = df.loc[df.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
    df.loc[df.date_inferred.isna(),'date_inferred'] = df.loc[df.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)

    df['imported'] = df.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[:4]!='1101' else 0)
    df['local'] = 1 - df.imported
    
    
    df_cases_state_time = df.groupby(['STATE','date_inferred'])[['imported','local']].sum()
    df_cases_state_time.reset_index(inplace=True)
    
    df_cases_state_time['cum_imported'] = df_cases_state_time.groupby('STATE').imported.transform(pd.Series.cumsum)
    df_cases_state_time['cum_local'] = df_cases_state_time.groupby('STATE').local.transform(pd.Series.cumsum)

    return df_cases_state_time

def calculate_scores(observations,forecasts, axis=1):
    """
    Given an array of forecasts (rows = samples, columns = dates), 
    generate the crps metric
    """
    import properscoring as ps

    crps_score = ps.crps_ensemble(
        observations,
        forecasts,
        axis=axis
    ) 
    #brier_score = ps.brier_score(
    #    observations,
    #    forecasts
    #)
    return crps_score#, brier_score

#inputs
data_date = pd.to_datetime(argv[1])
forecast_length = 28
train_periods =60
n_samples=1000
states = ['NSW','QLD','SA','TAS','VIC','WA','ACT','NT']

data = read_in_cases(data_date.strftime("%d%b"))
df_14days = read_in_cases(cases_file_date=(data_date+pd.Timedelta(days=14)).strftime("%d%b"))
df_future = read_in_cases("02Nov")
fig, ax_array = plt.subplots(figsize=(12,18), nrows=len(states)//2, ncols=2, sharex=True)

for i, state in enumerate(states):
    row = i//2
    col = i%2
    ax = ax_array[row, col]
    df_state = data.set_index(['STATE','date_inferred']).loc[state]

    #fill out all the values
    df_state = df_state.sort_index()
    df_state = df_state.reindex(
        pd.date_range(
            start=df_state.index.values[0], 
            end = data_date
            ),
        fill_value=0
        )
    #filter to relevant time period
    df_train = df_state.loc[pd.date_range(end=data_date+pd.Timedelta(days=-10), 
    periods=train_periods)]
    
    #training
    diff_array = np.diff(df_train.local)

    mu = np.mean(diff_array)
    std = np.std(diff_array)

    #predictions
    predictions = np.zeros(shape=(forecast_length,n_samples))
    for i in range(n_samples):
        predictions[:,i] = np.maximum(
            0,
            np.mean( df_train.local[-7:]) + np.cumsum(
                np.random.normal(mu, std, size=(forecast_length))
            )
        )
    
    quantiles  =np.quantile(predictions,
        [0.05,0.1,0.15,0.2,0.25,0.5,0.75,0.8,0.85,0.9,0.95],
        axis=1
    )
    lower10, lower20, lower30, lower40, lower50, median,\
        upper50,upper40, upper30, upper20, upper10 = quantiles
    
    forecast_dates = pd.date_range(
        start=data_date+pd.Timedelta(days=1),
        periods=forecast_length) 

    ax.plot(forecast_dates, median,'C1')

    ax.fill_between(forecast_dates, lower10, upper10, alpha=0.2,color='C1')
    ax.fill_between(forecast_dates, lower20, upper20, alpha=0.2,color='C1')
    ax.fill_between(forecast_dates, lower30, upper30, alpha=0.2,color='C1')
    ax.fill_between(forecast_dates, lower40, upper40, alpha=0.2,color='C1')
    ax.fill_between(forecast_dates, lower50, upper50, alpha=0.2,color='C1')

    ax.set_title(state)
    plot_start= forecast_dates[0] + pd.Timedelta(days=-28)
    end_date = forecast_dates[-1]
    for j,df in enumerate((data, df_14days, df_future)):
        dfplot = df.loc[(df.STATE==state) &
           (df.date_inferred >=plot_start) 
            & (df.date_inferred <=end_date)]
        ax.bar(dfplot.date_inferred,dfplot.local, label='Actual',color='grey', alpha=1/(j+1)**1.5)
    ax.set_xticks([data_date],minor=True)
    ax.xaxis.grid(which='minor', linestyle='--',alpha=0.6, color='black')

    ax.set_ylim(bottom=0)
    ax.tick_params('x',rotation=45)
    if col==0:
        ax.set_ylabel("Local cases")

    ## calculate score
    df_obs = df_future.loc[
        (df_future.STATE==state) 
        & (df_future.date_inferred >=forecast_dates[0]) 
            & (df_future.date_inferred <=forecast_dates[-1])
        ]
    df_obs =df_obs.set_index(['date_inferred'])
    df_obs = df_obs.sort_index()
    df_obs = df_obs.reindex(forecast_dates, fill_value=0)

    crps_scores = calculate_scores(
         df_obs.local.values,
         predictions.T, 
         axis=0)
    
    str_output = ['random_walk']
    str_output.extend([str(state), data_date])
    str_output.extend(crps_scores)

    os.makedirs("./analysis/comparison_models/results/", exist_ok=True)
    with open("./analysis/comparison_models/results/scores"+data_date.strftime("%d%b")+".csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(str_output)

plt.tight_layout()
plt.savefig("figs/retro/random_walk"+data_date.strftime("%Y-%m-%d")+".png",dpi=300)


