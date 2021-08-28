print('Generating R_L forecasts using mobility data.')
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

from scipy.special import expit
from datetime import timedelta, datetime
from sys import argv

from Reff_constants import *
from Reff_functions import *

import sys
sys.path.insert(0, '../')

# Define inputs
from params import num_forecast_days, VoC_start_date, apply_voc_to_R_L_hats, vaccination_start_date, apply_vacc_to_R_L_hats

num_forecast_days = num_forecast_days+3 # Add 3 days buffer to mobility forecast
data_date = pd.to_datetime(argv[1])
print("Using data from", data_date)
start_date = '2020-03-01'

# Scenario modelling
scenario = '' # Assume no scenario
scenario_date = '' 
if len(argv) > 2:
    scenario = argv[2]
    if len(argv) > 3:
        scenario_date = argv[3]
    print('Using scenario', scenario, 'with date', 'None' if scenario_date=='' else scenario_date)


# Get Google Data
df_google_all = read_in_google(Aus_only=True,moving=True,local=True)

# reading in vaccination data
# Load in vaccination data by state and date
if apply_vacc_to_R_L_hats:
    vaccination_by_state = pd.read_csv('data/vaccine_effect_timeseries.csv', parse_dates=['date'])
    vaccination_by_state = vaccination_by_state[['state', 'date','effect']]

    third_end_date = pd.to_datetime(data_date) - pd.Timedelta(days=10)
    third_start_date = '2021-06-04'
    # vaccination_by_state = vaccination_by_state[(vaccination_by_state.date > third_start_date) & (vaccination_by_state.date < third_end_date)] # Get only the dates we need.

    vaccination_by_state = vaccination_by_state.pivot(index='state', columns='date', values='effect') # Convert to matrix form

    # If we are missing recent vaccination data, fill it in with the most recent available data.
    latest_vacc_data = vaccination_by_state.columns[-1]
    if latest_vacc_data < pd.to_datetime(third_end_date):
        vaccination_by_state = pd.concat([vaccination_by_state]+[pd.Series(vaccination_by_state[latest_vacc_data], name=day) for day in pd.date_range(start=latest_vacc_data,end=third_end_date)], axis = 1)

    # Convert to simple array for indexing by state initials 
    vaccination_by_state_array = vaccination_by_state.to_numpy()

# Get survey data
surveys = pd.DataFrame()
path = "data/md/Barometer wave*.csv"
for file in glob.glob(path):
    surveys = surveys.append(pd.read_csv(file,parse_dates=['date']))
surveys = surveys.sort_values(by='date')

surveys.loc[surveys.state!='ACT','state'] = surveys.loc[surveys.state!='ACT','state'].map(states_initials).fillna(surveys.loc[surveys.state!='ACT','state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

always =surveys.loc[surveys.response=='Always'].set_index(["state",'date'])
always = always.unstack(['state'])

#fill in date range
idx = pd.date_range('2020-03-01',pd.to_datetime("today"))
always = always.reindex(idx, fill_value=np.nan)
always.index.name = 'date'
always =always.fillna(method='bfill')
always = always.stack(['state'])

#Zero out before first survey 20th March
always = always.reset_index().set_index('date')
always.loc[:'2020-03-20','count'] =0
always.loc[:'2020-03-20','respondents'] =0
always.loc[:'2020-03-20','proportion'] =0

always = always.reset_index().set_index(['state','date'])

survey_X = pd.pivot_table(data=always,
                          index='date',columns='state',values='proportion')
prop_all = survey_X

# Get posterior
df_samples = read_in_posterior(date = data_date.strftime("%Y-%m-%d"))

states = ['NSW','QLD','SA','VIC','TAS','WA','ACT','NT']
plot_states = states.copy()

one_month = data_date + timedelta(days=num_forecast_days)
days_from_March = (one_month - pd.to_datetime(start_date)).days

##filter out future info
prop = prop_all.loc[:data_date]
df_google = df_google_all.loc[df_google_all.date<=data_date]

# use this trick of saving the google data and then reloading it to kill 
# the date time values
df_google.to_csv("results/test_google_data.csv")
df_google = pd.read_csv("results/test_google_data.csv")

# Simple interpolation for missing vlaues in Google data
df_google = df_google.interpolate(method='linear', axis=0)
df_google.date = pd.to_datetime(df_google.date)

#forecast time parameters
today = data_date.strftime('%Y-%m-%d')
if df_google.date.values[-1] < data_date:
    #check if google has dates up to now
    # df_google is sorted by date
    # if not add days to the forecast
    n_forecast = num_forecast_days + (data_date- df_google.date.values[-1]).days
else:
    n_forecast = num_forecast_days

training_start_date = datetime(2020,3,1,0,0)
print("Forecast ends at {} days after 1st March".format(
    (pd.to_datetime(today) - pd.to_datetime(training_start_date)).days + num_forecast_days))
print("Final date is {}".format(pd.to_datetime(today) + timedelta(days=num_forecast_days)))
df_google = df_google.loc[df_google.date>= training_start_date]
outdata = {
    'date': [],
    'type': [],
    'state': [],
    'mean': [],
    'std' : [],
    }
predictors = mov_values.copy()
predictors.remove('residential_7days')

# Setup Figures
axes = []
figs = []
for var in predictors:
    fig, ax_states = plt.subplots(figsize=(7,8),nrows=4, ncols=2, sharex=True)
    axes.append(ax_states)
    #fig.suptitle(var)
    figs.append(fig)
##extra fig for microdistancing
var='Proportion people always microdistancing'
fig, ax_states = plt.subplots(figsize=(7,8),nrows=4, ncols=2, sharex=True)
axes.append(ax_states)
figs.append(fig)
if apply_vacc_to_R_L_hats:
    var='Reduction in Reff due to vaccination'
    fig, ax_states = plt.subplots(figsize=(7,8),nrows=4, ncols=2, sharex=True)
    axes.append(ax_states)
    figs.append(fig)

# Forecasting Params
mob_samples = 1000
n_training = 14 #  Period to examine trend
n_baseline = 91 # Period to create baseline

# Loop through states and run forecasting.
state_Rmed = {}
state_sims = {}
for i,state in enumerate(states):

    rownum = int(i/2.)
    colnum = np.mod(i,2)

    rows = df_google.loc[df_google.state==state].shape[0]
    #Rmed currently a list, needs to be a matrix
    Rmed_array = np.zeros(shape=(rows,len(predictors), mob_samples))
    for j, var in enumerate(predictors):
        for n in range(mob_samples):
            Rmed_array[:,j,n] = df_google[df_google['state'] == state][var].values.T + np.random.normal(loc=0,
                scale = df_google[df_google['state'] == state][var+'_std'],
            )
    dates = df_google[df_google['state'] == state]['date']

    #cap min and max at historical or (-50,0)
    minRmed_array = np.minimum(-50,np.amin(Rmed_array, axis = 0)) #1 by predictors by mob_samples size
    maxRmed_array = np.maximum(0,np.amax(Rmed_array, axis=0))

    sims  =  np.zeros(shape=(n_forecast,len(predictors),mob_samples)) # days by predictors by samples
    for n in range(mob_samples): # Loop through simulations
        Rmed = Rmed_array[:,:,n]
        minRmed = minRmed_array[:,n]
        maxRmed = maxRmed_array[:,n]

        R_baseline_mean = np.mean(Rmed[-n_baseline:,:], axis=0)
        R_diffs = np.diff(Rmed[-n_training:,:], axis=0)
        mu = np.mean(R_diffs, axis=0)
        cov = np.cov(R_diffs, rowvar=False) #columns are vars, rows are obs

        # Forecast mobility forward sequentially by day.
        current  = np.mean(Rmed[-5:,:],axis=0) # Start from last valid days
        for i in range(n_forecast):

            # Proportion of trend_force to regression_to_baseline_force
            # if state == 'WA':  # Force return to baseline immediately
            #     p_force = 0 
            # else:
            p_force = (n_forecast-i)/(n_forecast)

            trend_force = np.random.multivariate_normal(mu, cov) # Generate a single forward realisation of trend
            regression_to_baseline_force = np.random.multivariate_normal(0.05*(R_baseline_mean - current), cov) # Generate a single forward realisation of baseline regression
                
            new_forcast_points = current+p_force*trend_force +(1-p_force)*regression_to_baseline_force # Find overall simulation step
            current = new_forcast_points

            # Apply minimum and maximum
            new_forcast_points = np.maximum(minRmed, new_forcast_points)
            new_forcast_points = np.minimum(maxRmed, new_forcast_points)

            # ## SCENARIO MODELLING
            # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
            if len(argv)>2:
                cov_baseline = np.cov(Rmed[-42:-28,:], rowvar=False) # Make baseline cov for generating points
                mu_current = Rmed[-1,:]
                mu_victoria = np.array([-55.35057887, -22.80891056, -46.59531636, -75.99942378, -44.71119293])

                mu_baseline = np.mean(Rmed[-42:-28,:], axis = 0)

                if scenario_date != '': 
                    scenario_change_point = (pd.to_datetime(scenario_date) - data_date).days + (n_forecast-42)

                # Constant Lockdown
                if scenario[:12] == "no_reversion":
                    new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 

                # No Lockdown
                elif scenario == "full_reversion":  
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 
                    else:
                        # Revert to values the week before lockdown started
                        new_forcast_points = np.random.multivariate_normal(mu_baseline, cov_baseline) 

                # Temporary Lockdown
                elif scenario == "half_reversion": 
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 
                    else:
                        new_forcast_points = np.random.multivariate_normal((mu_current + mu_baseline)/2, cov_baseline) 

                # Stage 4
                if scenario == "stage4":
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 
                    else:
                        new_forcast_points = np.random.multivariate_normal(mu_victoria, cov_baseline) 

            sims[i,:,n] = new_forcast_points # Set this day in this simulation to the forecast realisation

    dd = [dates.tolist()[-1] + timedelta(days=x) for x in range(1,n_forecast+1)]


    sims_med = np.median(sims,axis=2) #N by predictors
    sims_q25 = np.percentile(sims,25,axis=2)
    sims_q75 = np.percentile(sims,75,axis=2)


    ##forecast mircodistancing
    mu_overall = np.mean(prop[state].values[-n_baseline:]) # Get a baseline value of microdistancing
    md_diffs = np.diff(prop[state].values[-n_training:])
    mu_diffs = np.mean(md_diffs)
    std_diffs = np.std(md_diffs)

    extra_days_md = (pd.to_datetime(df_google.date.values[-1]) - pd.to_datetime(prop[state].index.values[-1])).days

    current = [prop[state].values[-1]] * 1000 # Set all values to current value.
    new_md_forecast = []
    # Forecast mobility forward sequentially by day.
    for i in range(n_forecast + extra_days_md):
        p_force = (n_forecast+extra_days_md-i)/(n_forecast+extra_days_md) # Proportion of trend_force to regression_to_baseline_force
        trend_force = np.random.normal(mu_diffs, std_diffs, size=1000) # Generate step realisations in training trend direction
        regression_to_baseline_force = np.random.normal(0.05*(mu_overall - current), std_diffs)  # Generate realisations that draw closer to baseline
        current = current+p_force*trend_force +(1-p_force)*regression_to_baseline_force # Balance forces

        ## SCENARIO MODELLING
        # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
        if len(argv)>2:
            std_baseline = np.std(prop[state].values[-42:-28]) # Make baseline cov for generating points
            mu_baseline = np.mean(prop[state].values[-42:-28], axis =0)
            mu_current =prop[state].values[-1]

            if scenario_date != '': 
                scenario_change_point = (pd.to_datetime(scenario_date) - data_date).days + extra_days_md

            # Constant Lockdown
            if scenario == "no_reversion":
                current = np.random.normal(mu_current, std_baseline) 

            # No Lockdown
            elif scenario == "full_reversion":  
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline) 
                else:
                    # Revert to values the week before lockdown started
                    current = np.random.normal(mu_baseline, std_baseline) 

            # Temporary Lockdown
            elif scenario == "half_reversion":  # No Lockdown
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline) 
                else:
                     # Revert to values halfway between the before and after
                    current = np.random.normal((mu_current + mu_baseline)/2, std_baseline) 

            # # Stage 4
            # Not yet implemented

        new_md_forecast.append(current)

    md_sims = np.vstack(new_md_forecast) # Put forecast days together
    md_sims = np.minimum(1, md_sims)
    md_sims = np.maximum(0, md_sims)

    dd_md = [prop[state].index[-1] + timedelta(days=x) for x in range(1,n_forecast+extra_days_md+1)]

    if apply_vacc_to_R_L_hats:
        ## Forecasting vaccine effect -- might need small refactoring going forward to avoid using .loc 
        mu_overall = np.mean(vaccination_by_state.loc[state].values[-n_baseline:]) # Get a baseline value of vaccination
        vacc_diffs = np.diff(vaccination_by_state.loc[state].values[-n_training:])
        mu_diffs = np.mean(vacc_diffs)
        std_diffs = np.std(vacc_diffs)

        extra_days_vacc = (pd.to_datetime(df_google.date.values[-1]) - pd.to_datetime(vaccination_by_state.loc[state].index.values[-1])).days

        current = [vaccination_by_state.loc[state].values[-1]] * 1000 # Set all values to current value.
        new_vacc_forecast = []
        # Forecast mobility forward sequentially by day.
        for i in range(n_forecast + extra_days_vacc):
            trend_force = np.random.normal(mu_diffs, std_diffs, size=1000) # Generate step realisations in training trend direction
            current = current+trend_force       # no regression to baseline for vaccination or scenario modelling yet
            
            new_vacc_forecast.append(current)

        vacc_sims = np.vstack(new_vacc_forecast) # Put forecast days together
        vacc_sims = np.minimum(1, vacc_sims)
        vacc_sims = np.maximum(0, vacc_sims)

        #get dates
        dd_vacc = [vaccination_by_state.loc[state].index[-1] + timedelta(days=x) for x in range(1,n_forecast+extra_days_vacc+1)]

    for j, var in enumerate(predictors+['md_prop']+['vaccination']):
        #Record data
        axs=axes[j]
        if (state=='AUS') and (var=='md_prop'):
            continue

        if var == 'md_prop':
            outdata['type'].extend([var]*len(dd_md))
            outdata['state'].extend([state]*len(dd_md))
            outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd_md])
            outdata['mean'].extend(np.mean(md_sims,axis=1))
            outdata['std'].extend(np.std(md_sims,axis=1))

        elif var == 'vaccination':
            outdata['type'].extend([var]*len(dd_vacc))
            outdata['state'].extend([state]*len(dd_vacc))
            outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd_vacc])
            outdata['mean'].extend(np.mean(vacc_sims,axis=1))
            outdata['std'].extend(np.std(vacc_sims,axis=1))

        else:
            outdata['type'].extend([var]*len(dd))
            outdata['state'].extend([state]*len(dd))
            outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd])
            outdata['mean'].extend(np.mean(sims[:,j,:],axis=1))
            outdata['std'].extend(np.std(sims[:,j,:],axis=1))

        if state in plot_states:

            if var == 'md_prop':
                ## md plot
                axs[rownum,colnum].plot(prop[state].index,prop[state].values, lw=1)
                #axs[rownum,colnum].fill_between(dates,
                #                                np.percentile(Rmed_array[:,j,:], 25, axis =1),
                #                                np.percentile(Rmed_array[:,j,:], 75, axis =1),
                #                               alpha=0.5)

                axs[rownum,colnum].plot(dd_md,np.median(md_sims,axis=1),'k', lw=1)
                axs[rownum,colnum].fill_between(dd_md, np.quantile(md_sims,0.25, axis=1),
                                                np.quantile(md_sims,0.75,axis=1), color='k',alpha = 0.1)

            elif var == 'vaccination':
                ## vaccination plot
                axs[rownum,colnum].plot(vaccination_by_state.loc[state].index,vaccination_by_state.loc[state].values, lw=1)
                #axs[rownum,colnum].fill_between(dates,
                #                                np.percentile(Rmed_array[:,j,:], 25, axis =1),
                #                                np.percentile(Rmed_array[:,j,:], 75, axis =1),
                #                               alpha=0.5)

                axs[rownum,colnum].plot(dd_vacc,np.median(vacc_sims,axis=1),'k', lw=1)
                axs[rownum,colnum].fill_between(dd_vacc, np.quantile(vacc_sims,0.25, axis=1),
                                                np.quantile(vacc_sims,0.75,axis=1), color='k',alpha = 0.1)

            else: 
                ## all other predictors
                axs[rownum,colnum].plot(dates,df_google[df_google['state'] == state][var].values, lw=1)
                axs[rownum,colnum].fill_between(dates,
                                                np.percentile(Rmed_array[:,j,:], 25, axis =1),
                                                np.percentile(Rmed_array[:,j,:], 75, axis =1),
                                            alpha=0.5)

                axs[rownum,colnum].plot(dd,sims_med[:,j],'k', lw=1)
                axs[rownum,colnum].fill_between(dd, sims_q25[:,j], sims_q75[:,j], color='k',alpha = 0.1)

            axs[rownum,colnum].set_title(state)
            axs[rownum,colnum].axhline(1,ls = '--', c = 'k', lw=1)
            axs[rownum,colnum].set_title(state)
            axs[rownum,colnum].tick_params('x',rotation=90)
            axs[rownum,colnum].tick_params('both',labelsize=8)
            if j<len(predictors):
                axs[rownum,colnum].set_ylabel(predictors[j].replace('_',' ')[:-5], fontsize=7)
            elif var == 'md_prop':
                axs[rownum,colnum].set_ylabel('Proportion of respondents\n micro-distancing', fontsize=7)  
            elif var == 'vaccination':
                axs[rownum,colnum].set_ylabel('Vaccine reduction in Reff', fontsize=7)  
    state_Rmed[state] = Rmed_array
    state_sims[state] = sims
    
os.makedirs("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date, exist_ok=True)
for i,fig in enumerate(figs):
    fig.text(0.5, 0.02, 
    'Date', 
    ha='center', va='center',
    fontsize=15)

    
    if i<len(predictors):       # this plots the google mobility forecasts

        # fig.text(0.03, 0.5, 
        # predictors[i].replace('_',' ')[:-5], 
        # ha='center', va='center',
        # rotation='vertical',
        # fontsize=15)
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date+"/"+str(predictors[i])+scenario+scenario_date+".png",dpi=400)


    elif i == len(predictors):      # this plots the microdistancing forecasts
        # fig.text(0.03, 0.5, 
        # 'Proportion of respondents\n micro-distancing', 
        # ha='center', va='center',
        # rotation='vertical',
        # fontsize=15)
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date+"/micro_dist.png",dpi=400)

    else:       # finally this plots the vaccination forecasts 

        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date+"/vaccination.png",dpi=400)
        

df_out = pd.DataFrame.from_dict(outdata)

df_md = df_out.loc[df_out.type=='md_prop']
if apply_vacc_to_R_L_hats:
    df_vaccination = df_out.loc[df_out.type=='vaccination']
    df_out = df_out.loc[df_out.type!='vaccination']

# pull out md and vaccination in 2 steps cause not sure how to do it by a list
df_out = df_out.loc[df_out.type!='md_prop']

df_forecast = pd.pivot_table(df_out, columns=['type'],index=['date','state'],values=['mean'])
df_std = pd.pivot_table(df_out, columns=['type'],index=['date','state'],values=['std'])

df_forecast_md = pd.pivot_table(df_md, columns=['state'],index=['date'],values=['mean'])
df_forecast_md_std = pd.pivot_table(df_md, columns=['state'],index=['date'],values=['std'])

#align with google order in columns
df_forecast = df_forecast.reindex([('mean',val) for val in predictors],axis=1)
df_std = df_std.reindex([('std',val) for val in predictors],axis=1)
df_forecast.columns = predictors  #remove the tuple name of columns
df_std.columns = predictors

df_forecast_md = df_forecast_md.reindex([('mean',state) for state in states],axis=1)
df_forecast_md_std = df_forecast_md_std.reindex([('std',state) for state in states],axis=1)

df_forecast_md.columns = states
df_forecast_md_std.columns = states

if apply_vacc_to_R_L_hats:
    df_forecast_vaccination = pd.pivot_table(df_vaccination, columns=['state'],index=['date'],values=['mean'])
    df_forecast_vaccination_std = pd.pivot_table(df_vaccination, columns=['state'],index=['date'],values=['std'])
    df_forecast_vaccination = df_forecast_vaccination.reindex([('mean',state) for state in states],axis=1)
    df_forecast_vaccination_std = df_forecast_vaccination_std.reindex([('std',state) for state in states],axis=1)
    df_forecast_vaccination.columns = states
    df_forecast_vaccination_std.columns = states
    df_forecast_vaccination = df_forecast_vaccination.reset_index()
    df_forecast_vaccination_std = df_forecast_vaccination_std.reset_index()
    df_forecast_vaccination.date = pd.to_datetime(df_forecast_vaccination.date)
    df_forecast_vaccination_std.date = pd.to_datetime(df_forecast_vaccination_std.date)

df_forecast = df_forecast.reset_index()
df_std = df_std.reset_index()

df_forecast_md = df_forecast_md.reset_index()
df_forecast_md_std = df_forecast_md_std.reset_index()

df_forecast.date = pd.to_datetime(df_forecast.date)
df_std.date = pd.to_datetime(df_std.date)

df_forecast_md.date = pd.to_datetime(df_forecast_md.date)
df_forecast_md_std.date = pd.to_datetime(df_forecast_md_std.date)

df_R = df_google[['date','state']+mov_values + [val+'_std' for val in mov_values]]
df_R = pd.concat([df_R,df_forecast],ignore_index=True,sort=False)
df_R['policy'] = (df_R.date>='2020-03-20').astype('int8')

df_md = pd.concat([prop,df_forecast_md.set_index('date')])

if apply_vacc_to_R_L_hats:
    ############# now we read in the vaccine time series again...
    vaccination_by_state = pd.read_csv('data/vaccine_effect_timeseries.csv', parse_dates=['date'])
    vaccination_by_state = vaccination_by_state[['state', 'date','effect']]

    third_start_date = '2021-06-04'
    third_end_date = pd.to_datetime(data_date) - pd.Timedelta(days=10)
    # vaccination_by_state = vaccination_by_state[(vaccination_by_state.date > third_start_date) & (vaccination_by_state.date < third_end_date)] # Get only the dates we need.
    # vaccination_by_state = vaccination_by_state[vaccination_by_state['state'].isin(third_states)] # Isolate fitting states
    vaccination_by_state = vaccination_by_state.pivot(index='state', columns='date', values='effect') # Convert to matrix form

    # If we are missing recent vaccination data, fill it in with the most recent available data.
    latest_vacc_data = vaccination_by_state.columns[-1]
    if latest_vacc_data < pd.to_datetime(third_end_date):
        vaccination_by_state = pd.concat([vaccination_by_state]+[pd.Series(vaccination_by_state[latest_vacc_data], name=day) for day in pd.date_range(start=latest_vacc_data,end=third_end_date)], axis = 1)

    # the above part only deals with data after the vaccination program begins -- we also need to account 
    # for a fixed effect of 1.0 before that
    start_date = '2020-03-01'
    before_vacc_dates = pd.date_range(start_date, vaccination_by_state.columns[0]- timedelta(days=1), freq='d')

    # this is just a df of ones with all the missing dates as indices
    before_vacc_Reff_reduction = pd.DataFrame(np.ones((8,len(before_vacc_dates))))
    before_vacc_Reff_reduction.columns = before_vacc_dates
    before_vacc_Reff_reduction.index = vaccination_by_state.index

    # merge the vaccine data and the 1's dataframes
    vacc_df = pd.concat([before_vacc_Reff_reduction.T, vaccination_by_state.T])

    # merge the dfs of the past and forecasted values on the date 
    df_vaccination = pd.concat([vacc_df,df_forecast_vaccination.set_index('date')])
    # save the forecasted vaccination line
    df_vaccination.to_csv("results/forecasted_vaccination.csv")

#prop_std = pd.DataFrame(np.random.beta(1+survey_counts, 1+survey_respond), columns = survey_counts.columns, index = prop.index)
#df_md_std = pd.concat([prop_std,df_forecast_md_std.set_index('date')])

expo_decay=True
theta_md = np.tile(df_samples['theta_md'].values, (df_md['NSW'].shape[0],1))

fig, ax = plt.subplots(figsize=(12,9), nrows=4,ncols=2,sharex=True, sharey=True)

for i,state in enumerate(plot_states):
    prop_sim= df_md[state].values#np.random.normal(df_md[state].values, df_md_std.values)
    if expo_decay:
        md = ((1+theta_md).T**(-1* prop_sim)).T
    else:
        md = (2*expit(-1*theta_md*prop_sim[:,np.newaxis]))


    row = i//2
    col = i%2

    ax[row,col].plot(df_md[state].index, np.median(md,axis=1 ), label='Microdistancing')
    ax[row,col].fill_between(df_md[state].index, np.quantile(md,0.25,axis=1 ),np.quantile(md,0.75,axis=1 ),
                            label='Microdistancing',
                            alpha=0.4,
                            color='C0')
    ax[row,col].fill_between(df_md[state].index, np.quantile(md,0.05,axis=1 ),np.quantile(md,0.95,axis=1 ),
                            label='Microdistancing',
                            alpha=0.4,
                            color='C0')
    ax[row,col].set_title(state)
    ax[row,col].tick_params('x',rotation=45)

    ax[row,col].set_xticks([df_md[state].index.values[-n_forecast-extra_days_md]],minor=True,)
    ax[row,col].xaxis.grid(which='minor', linestyle='-.',color='grey', linewidth=1)

fig.text(0.03, 0.5, 
    'Multiplicative effect \n of micro-distancing $M_d$', 
    ha='center', va='center', rotation='vertical',
    fontsize=20)

fig.text(0.5, 0.04, 
    'Date', 
    ha='center', va='center',
    fontsize=20)
plt.tight_layout(rect=[0.05,0.04,1,1])
fig.savefig("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date+"/md_factor.png",dpi=144)


df_R = df_R.sort_values('date')
n_samples = 100
samples = df_samples.sample(n_samples) #test on sample of 2
forecast_type = ['R_L','R_L0']
state_Rs = {
    'state':[],
    'date':[],
    'type':[],
    'median':[],
    'lower':[],
    'upper':[],
    'bottom':[],
    'top':[],
    'mean':[],
    'std':[],
}
ban = '2020-03-20'
new_pol = '2020-06-01' #VIC and NSW allow gatherings of up to 20 people, other jurisdictions allow for

expo_decay=True

# start and end date for the third wave 
third_start_date = '2021-06-04'
third_end_date = third_end_date = data_date - pd.Timedelta(days=10) # Subtract 10 days to avoid right truncation

typ_state_R={}
mob_forecast_date = df_forecast.date.min()
mob_samples = 100

state_key = {
    'NSW':'1',
    'QLD':'2',
    'SA':'3',
    'TAS':'4',
    'VIC':'5',
    'WA':'6',
}

# this key is just used so that the right vaccination multiplier gets used in the calculation
vacc_state_key = {
    'NSW':'1',
    'QLD':'2',
    'VIC':'3',
}

for typ in forecast_type:
    state_R={}
    for state in states:
    #sort df_R by date so that rows are dates

        #rows are dates, columns are predictors
        df_state = df_R.loc[df_R.state==state]
        dd = df_state.date
        post_values = samples[predictors].values.T
        prop_sim = df_md[state].values
        
        if apply_vacc_to_R_L_hats:
            vacc_sim = df_vaccination[state].values

        #take right size of md to be N by N
        theta_md = np.tile(samples['theta_md'].values, (df_state.shape[0],mob_samples))
        if expo_decay:
            md = ((1+theta_md).T**(-1* prop_sim)).T
        #else:
        #    md = (2*expit(-1*theta_md*prop_sim[:,np.newaxis]))
        
        if apply_vacc_to_R_L_hats:
            # only have fit the effect to NSW, QLD and VIC
            if state in {'NSW', 'QLD', 'VIC'}:
                # transposing the posterior vacc effect before we multiply it by the forecasted values
                vacc_post = np.tile(samples['vacc_effect_third_wave['+vacc_state_key[state]+']'], (df_state.shape[0],mob_samples)).T * vacc_sim
            else:
                # if we don't fit to a state, for now we just use the data -- will want to sample from the hierarchical part 
                # going forwards
                vacc_post = np.tile([1.0]*samples['vacc_effect_third_wave[1]'].shape[0], (df_state.shape[0],mob_samples)).T * vacc_sim
                
            # last thing to do is modify the 
            for ii in range(vacc_post.shape[0]):
                if ii < df_state.loc[df_state.date<vaccination_start_date].shape[0]:
                    vacc_post[ii] = 1.0

        for n in range(mob_samples):
            #add gaussian noise to predictors before forecast
            df_state.loc[df_state.date<mob_forecast_date,predictors] = state_Rmed[state][:,:,n]/100#df_state.loc[
                #df_state.date<mob_forecast_date,predictors]/100 + np.random.normal(
                #loc= 0, scale = df_state.loc[
                #    (df_state.date<mob_forecast_date),
                #    [val+'_std' for val in predictors]].values/100)


            #add gaussian noise to predictors after forecast
            df_state.loc[df_state.date>=mob_forecast_date,predictors] = state_sims[state][:,:,n]/100
            #df_state.loc[
            #   df_state.date>=mob_forecast_date,predictors]/100 + np.random.normal(
            #   loc= 0, scale = df_std.loc[(df_std.state==state,predictors)].values/100)


            #dd = df_state.date

            df1 =df_state.loc[df_state.date<=ban]
            X1 = df1[predictors] #N by K

            #sample the right R_L
            if state in ("ACT","NT","TAS"):
                sim_R = np.tile(samples.R_L.values, (df_state.shape[0],mob_samples))
            else:
                #if state =='VIC':
                #    sim_R = np.tile(
                #        samples['R_Li['+state_key[state]+']'].values + samples['R_temp'].values,
                #         (df_state.shape[0],mob_samples)
                #         )
                #else:
                sim_R = np.tile(samples['R_Li['+state_key[state]+']'].values, (df_state.shape[0],mob_samples))


            #set initial pre ban values of md to 1
            md[:X1.shape[0],:] = 1

            if n==0:
                #initialise arrays (loggodds)
                logodds = X1 @ post_values # N by K times (Nsamples by K )^T = Ndate by Nsamples

                if typ =='R_L':
                    df2 = df_state.loc[(df_state.date>ban) & (df_state.date<new_pol)]
                    df3 = df_state.loc[df_state.date>=new_pol]
                    X2 = df2[predictors]
                    X3 = df3[predictors]

                    #halve effect of md
                    #md[(X1.shape[0]+df2.shape[0]):,:] = 1- 0.5 *( 1 - md[(X1.shape[0]+df2.shape[0]):,:])

                    logodds = np.append(logodds,X2 @ post_values,axis=0)
                    logodds = np.append(logodds,X3 @ post_values,axis=0)

                    #md = np.append(md, ((1+theta_md).T**(-1* prop2)).T, axis=0)
                    #md = np.append(md, ((1+theta_md).T**(-1* prop3)).T, axis=0)

                elif typ=='R_L0':
                    df2 = df_state.loc[(df_state.date>ban) & (df_state.date<new_pol)]
                    df3 = df_state.loc[df_state.date>=new_pol]
                    X2 = df2[predictors]
                    X3 = np.zeros_like(df3[predictors])

                    #social mobility all at baseline implies R_l = R_L0

                    #md has no effect after June 1st
                    md[(X1.shape[0]+df2.shape[0]):,:] = 1

                    logodds = np.append(logodds,X2 @ post_values,axis=0)
                    logodds = np.append(logodds,X3 @ post_values,axis=0)


                else:
                    #forecast as before, no changes to md
                    df2 = df_state.loc[df_state.date>ban]
                    X2 = df2[predictors]

                    logodds = np.append(logodds,X2 @ post_values,axis=0)
                                    #df_state.loc[df_state.date>'2020-03-15',predictors].values/100 @ samples[predictors].values.T, axis = 0)

            else:
                #concatenate to pre-existing logodds martrix
                logodds1 = X1 @ post_values

                if typ =='R_L':
                    df2 = df_state.loc[(df_state.date>ban) & (df_state.date<new_pol)]
                    df3 = df_state.loc[df_state.date>=new_pol]
                    X2 = df2[predictors]
                    X3 = df3[predictors]

                    prop2 = df_md.loc[ban:new_pol,state].values
                    prop3 = df_md.loc[new_pol:,state].values

                    #halve effect of md
                    #md[(X1.shape[0]+df2.shape[0]):,:] = 1- 0.5 *( 1 - md[(X1.shape[0]+df2.shape[0]):,:])

                    logodds2 = X2 @ post_values
                    logodds3 = X3 @ post_values

                    logodds_sample = np.append(logodds1, logodds2, axis=0)
                    logodds_sample = np.append(logodds_sample, logodds3, axis=0)

                elif typ=='R_L0':
                    df2 = df_state.loc[(df_state.date>ban) & (df_state.date<new_pol)]
                    df3 = df_state.loc[df_state.date>=new_pol]
                    X2 = df2[predictors]
                    X3 = np.zeros_like(df3[predictors])

                    #social mobility all at baseline implies R_l = R_L0

                    #md has no effect after June 1st

                    md[(X1.shape[0]+df2.shape[0]):,:] = 1

                    logodds2 = X2 @ post_values
                    logodds3 = X3 @ post_values

                    logodds_sample = np.append(logodds1, logodds2, axis=0)
                    logodds_sample = np.append(logodds_sample, logodds3, axis=0)


                else:
                    #forecast as before, no changes to md
                    df2 = df_state.loc[df_state.date>ban]
                    X2 = df2[predictors]

                    logodds2 = X2 @ post_values

                    logodds_sample = np.append(logodds1, logodds2, axis=0)

                ##concatenate to previous
                logodds = np.concatenate((logodds, logodds_sample ), axis =1)


        # calculate the forecasted R_L values
        if apply_vacc_to_R_L_hats:
            R_L = 2 * md * sim_R * expit( logodds ) * vacc_post.T
        else: 
            R_L = 2 * md * sim_R * expit( logodds )
        
        # print("Including the voc effects into the R_L forecasts")
        # create an matrix of mob_samples realisations which is an indicator of the voc (delta right now) 
        # which will be 1 up until the voc_start_date and then it will be values from the posterior sample
        voc_multiplier = np.tile(samples['voc_effect_third_wave'].values, (df_state.shape[0],mob_samples))
        # now we just modify the values before the introduction of the voc to be 1.0
        if apply_voc_to_R_L_hats:
            for ii in range(voc_multiplier.shape[0]):
                if ii < df_state.loc[df_state.date<VoC_start_date].shape[0]:
                    voc_multiplier[ii] = 1.0

            R_L *= voc_multiplier

        R_L_lower = np.percentile(R_L,25,axis=1)
        R_L_upper = np.percentile(R_L,75,axis=1)
        R_L_bottom = np.percentile(R_L,5,axis=1)
        R_L_top = np.percentile(R_L,95,axis=1)
        R_L_med = np.median(R_L,axis=1)

        #R_L
        state_Rs['state'].extend([state]*df_state.shape[0])
        state_Rs['type'].extend([typ]*df_state.shape[0])
        state_Rs['date'].extend(dd.values) #repeat n_samples times?
        state_Rs['lower'].extend(R_L_lower)
        state_Rs['median'].extend(R_L_med)
        state_Rs['upper'].extend(R_L_upper)
        state_Rs['top'].extend(R_L_top)
        state_Rs['bottom'].extend(R_L_bottom)
        state_Rs['mean'].extend(np.mean(R_L,axis=1))
        state_Rs['std'].extend(np.std(R_L,axis=1))

        state_R[state] = R_L
    typ_state_R[typ] = state_R



for state in states:
    #R_I
    R_I = samples['R_I'].values[:df_state.shape[0]]


    state_Rs['state'].extend([state]*df_state.shape[0])
    state_Rs['type'].extend(['R_I']*df_state.shape[0])
    state_Rs['date'].extend(dd.values)
    state_Rs['lower'].extend(np.repeat(np.percentile(R_I,25),df_state.shape[0]))
    state_Rs['median'].extend(np.repeat(np.median(R_I),df_state.shape[0]))
    state_Rs['upper'].extend(np.repeat(np.percentile(R_I,75),df_state.shape[0]))
    state_Rs['top'].extend(np.repeat(np.percentile(R_I,95),df_state.shape[0]))
    state_Rs['bottom'].extend(np.repeat(np.percentile(R_I,5),df_state.shape[0]))
    state_Rs['mean'].extend(np.repeat(np.mean(R_I),df_state.shape[0]))
    state_Rs['std'].extend(np.repeat(np.std(R_I),df_state.shape[0]))

df_Rhats = pd.DataFrame().from_dict(state_Rs)
df_Rhats = df_Rhats.set_index(['state','date','type'])

d = pd.DataFrame()
for state in states:
    for i,typ in enumerate(forecast_type):
        if i==0:
            t = pd.DataFrame.from_dict(typ_state_R[typ][state])
            t['date'] = dd.values
            t['state'] = state
            t['type'] = typ
        else:
            temp = pd.DataFrame.from_dict(typ_state_R[typ][state])
            temp['date'] = dd.values
            temp['state'] = state
            temp['type'] = typ
            t = t.append(temp)
    #R_I
    i = pd.DataFrame(np.tile(samples['R_I'].values,(len(dd.values),100)))
    i['date'] = dd.values
    i['type'] = 'R_I'
    i['state'] = state

    t = t.append(i)

    d = d.append(t)

        #df_Rhats = df_Rhats.loc[(df_Rhats.state==state)&(df_Rhats.type=='R_L')].join( t)

d = d.set_index(['state','date','type'])
df_Rhats = df_Rhats.join(d)
df_Rhats = df_Rhats.reset_index()
df_Rhats.state = df_Rhats.state.astype(str)
df_Rhats.type = df_Rhats.type.astype(str)

fig, ax = plt.subplots(figsize=(12,9), nrows=4,ncols=2,sharex=True, sharey=True)

for i,state in enumerate(plot_states):

    row = i//2
    col = i%2

    plot_df = df_Rhats.loc[(df_Rhats.state==state) & (df_Rhats.type=='R_L')]

    ax[row,col].plot(plot_df.date, plot_df['mean'])

    ax[row,col].fill_between( plot_df.date, plot_df['lower'],plot_df['upper'],alpha=0.4,color='C0')
    ax[row,col].fill_between( plot_df.date, plot_df['bottom'],plot_df['top'],alpha=0.4,color='C0')

    ax[row,col].tick_params('x',rotation=90)
    ax[row,col].set_title(state)
    ax[row,col].set_yticks([1],minor=True,)
    ax[row,col].set_yticks([0,2,3],minor=False)
    ax[row,col].set_yticklabels([0,2,3],minor=False)
    ax[row,col].yaxis.grid(which='minor',linestyle='--',color='black',linewidth=2)
    ax[row,col].set_ylim((0,3))

    ax[row,col].set_xticks([plot_df.date.values[-n_forecast]],minor=True,)
    ax[row,col].xaxis.grid(which='minor', linestyle='-.',color='grey', linewidth=1)
#fig.autofmt_xdate()
fig.text(
    0.03,0.5,
    'Effective \nreproduction number',
    va='center',ha='center',
    rotation='vertical',
    fontsize=20
)
fig.text(
    0.525,0.02,
    'Date',
    va='center',ha='center',
    fontsize=20
)
plt.tight_layout(rect=[0.04,0.04,1,1])
os.makedirs("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date, exist_ok=True)
plt.savefig("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+scenario_date+"/soc_mob_R_L_hats"+data_date.strftime('%Y-%m-%d')+".png",dpi=102)

################## now we save the posterior stuff

df_Rhats = df_Rhats[['state','date','type','median',
'bottom','lower','upper','top']+[i for i in range(2000)] ]
#df_Rhats.columns = ['state','date','type','median',
#'bottom','lower','upper','top']  + [i for i in range(1000)]

df_hdf = df_Rhats.loc[df_Rhats.type=='R_L']
df_hdf = df_hdf.append(df_Rhats.loc[(df_Rhats.type=='R_I')&(df_Rhats.date=='2020-03-01')])
df_hdf = df_hdf.append(df_Rhats.loc[(df_Rhats.type=='R_L0')&(df_Rhats.date=='2020-03-01')])
#df_Rhats.to_csv('./soc_mob_R'+today+'.csv')
df_hdf.to_hdf('results/soc_mob_R'+data_date.strftime('%Y-%m-%d')+scenario+scenario_date+'.h5',key='Reff')
