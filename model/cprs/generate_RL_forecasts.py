import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

from scipy.stats import norm
from scipy.special import expit
from datetime import date, timedelta, datetime
from sys import argv

from Reff_constants import *
from Reff_functions import *

import matplotlib.dates as mdates
locator =mdates.MonthLocator()
df_google_all = read_in_google(Aus_only=True,moving=True,local=True)
states = ['NSW','QLD','SA','VIC','TAS','WA','ACT','NT']#,'AUS']
plot_states = states.copy()
#plot_states.remove('AUS')


# Scenario modelling
if len(argv) > 2:
    scenario = argv[2]
    scenario_change_point = 5
else:
    scenario = ''

## grab survey data

#surveys = pd.read_csv("data/md/Barometer wave 1 to 10.csv",parse_dates = ['date'])
#surveys = surveys.append(pd.read_csv("data/md/Barometer wave 11 complience.csv",parse_dates=['date'])) #they spelt compliance wrong??
surveys = pd.DataFrame()
##Improve this to read by glob.glob and get all of them


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




data_date = pd.to_datetime(argv[1])
print(data_date)
df_samples = read_in_posterior(date = data_date.strftime("%Y-%m-%d"))

one_month = data_date + timedelta(days= 42)

days_from_March = one_month.dayofyear -pd.to_datetime('2020-03-01').dayofyear
if days_from_March<0:
    days_from_March =days_from_March + 366

##filter out future info
prop = prop_all.loc[:data_date]
df_google = df_google_all.loc[df_google_all.date<=data_date]

#Simple interpolation for missing vlaues in Google data
df_google = df_google.interpolate(method='linear',axis=0)

#forecast time parameters
today = data_date.strftime('%Y-%m-%d')
if df_google.date.values[-1] < data_date:
    #check if google has dates up to now
    # df_google is sorted by date
    # if not add days to the forecast
    n_forecast = 42 + (data_date- df_google.date.values[-1]).days
else:
    n_forecast = 42

#cap = 0 #10?
training_start_date = datetime(2020,3,1,0,0)
print("Forecast ends at {} days after 1st March".format(
    (pd.to_datetime(today) - pd.to_datetime(training_start_date)).days + 42))
print("Final date is {}".format(pd.to_datetime(today) + timedelta(days=42)))
df_google = df_google.loc[df_google.date>= training_start_date]
outdata = {'date': [],
        'type': [],
        'state': [],
            'mean': [],
            'std' : [],
        }
predictors = mov_values.copy()
predictors.remove('residential_7days')

mob_samples = 1000

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
#fig.suptitle(var)
figs.append(fig)

n_training = 14 #  Period to examine trend
n_baseline = 91 # Period to create baseline

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
            p_force = max((n_forecast-i-5)/(n_forecast), 0)

            trend_force = np.random.multivariate_normal(mu, cov) # Generate a single forward realisation of trend
            regression_to_baseline_force = np.random.multivariate_normal(0.1*(R_baseline_mean - current), cov) # Generate a single forward realisation of baseline regression
                
            new_forcast_points = current+p_force*trend_force +(1-p_force)*regression_to_baseline_force # Find overall simulation step
            current = new_forcast_points

            # Apply minimum and maximum
            new_forcast_points = np.maximum(minRmed, new_forcast_points)
            new_forcast_points = np.minimum(maxRmed, new_forcast_points)

            # ## SIMULATION MODELLING
            # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
            if (state == "VIC") and len(argv)>2:
                cov_baseline = np.cov(Rmed[-21:-14,:], rowvar=False) # Make baseline cov for generating points
                mu_baseline = np.mean(Rmed[-21:-14,:], axis =0)
                mu_current = Rmed[-1,:]

                # Constant Lockdown
                if scenario == "no_reversion":
                    new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 

                # No Lockdown
                elif scenario == "full_reversion":  
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 
                    else:
                        # Revert to values the week before lockdown started
                        new_forcast_points = np.random.multivariate_normal(mu_baseline, cov_baseline) 

                # Temporary Lockdown
                elif scenario[:14] == "half_reversion":  # No Lockdown
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(mu_current, cov_baseline) 
                    else:
                        new_forcast_points = np.random.multivariate_normal((mu_current + mu_baseline)/2, cov_baseline) 

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
        regression_to_baseline_force = np.random.normal(0.01*(mu_overall - current), std_diffs)  # Generate realisations that draw closer to baseline
        current = current+p_force*trend_force +(1-p_force)*regression_to_baseline_force # Balance forces

        ## SIMULATION MODELLING
        # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
        if (state == "VIC") and len(argv)>2:
            std_baseline = np.std(prop[state].values[-21:-14]) # Make baseline cov for generating points
            mu_baseline = np.mean(prop[state].values[-21:-14], axis =0)
            mu_current =prop[state].values[-1]

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


        new_md_forecast.append(current)

    md_sims = np.vstack(new_md_forecast) # Put forecast days together
    md_sims = np.minimum(1, md_sims)
    md_sims = np.maximum(0, md_sims)
    #get dates
    dd_md = [prop[state].index[-1] + timedelta(days=x) for x in range(1,n_forecast+extra_days_md+1)]


    for j, var in enumerate(predictors+['md_prop']):
        #Record data
        axs=axes[j]
        if (state=='AUS') and (var=='md_prop'):
            continue

        if var != 'md_prop':
            outdata['type'].extend([var]*len(dd))
            outdata['state'].extend([state]*len(dd))
            outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd])
            outdata['mean'].extend(np.mean(sims[:,j,:],axis=1))
            outdata['std'].extend(np.std(sims[:,j,:],axis=1))
        else:
            outdata['type'].extend([var]*len(dd_md))
            outdata['state'].extend([state]*len(dd_md))
            outdata['date'].extend([d.strftime('%Y-%m-%d') for d in dd_md])
            outdata['mean'].extend(np.mean(md_sims,axis=1))

            outdata['std'].extend(np.std(md_sims,axis=1))

        if state in plot_states:
            if var != 'md_prop':
                axs[rownum,colnum].plot(dates,df_google[df_google['state'] == state][var].values)
                axs[rownum,colnum].fill_between(dates,
                                                np.percentile(Rmed_array[:,j,:], 25, axis =1),
                                                np.percentile(Rmed_array[:,j,:], 75, axis =1),
                                            alpha=0.5)

                axs[rownum,colnum].plot(dd,sims_med[:,j],'k')
                axs[rownum,colnum].fill_between(dd, sims_q25[:,j], sims_q75[:,j], color='k',alpha = 0.1)
            else:
                ##md plot
                axs[rownum,colnum].plot(prop[state].index,prop[state].values)
                #axs[rownum,colnum].fill_between(dates,
                #                                np.percentile(Rmed_array[:,j,:], 25, axis =1),
                #                                np.percentile(Rmed_array[:,j,:], 75, axis =1),
                #                               alpha=0.5)

                axs[rownum,colnum].plot(dd_md,np.median(md_sims,axis=1),'k')
                axs[rownum,colnum].fill_between(dd_md, np.quantile(md_sims,0.25, axis=1),
                                                np.quantile(md_sims,0.75,axis=1), color='k',alpha = 0.1)

            axs[rownum,colnum].set_title(state)
            axs[rownum,colnum].axhline(1,ls = '--', c = 'k')
            axs[rownum,colnum].set_title(state)
            axs[rownum,colnum].tick_params('x',rotation=45)
            axs[rownum,colnum].xaxis.set_major_locator(locator)
            #fig.autofmt_xdate()

    state_Rmed[state] = Rmed_array
    state_sims[state] = sims
os.makedirs("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario, exist_ok=True)
for i,fig in enumerate(figs):
    fig.text(0.5, 0.02, 
    'Date', 
    ha='center', va='center',
    fontsize=15)

    
    if i<len(predictors):

        fig.text(0.03, 0.5, 
        predictors[i].replace('_',' ')[:-5], 
        ha='center', va='center',
        rotation='vertical',
        fontsize=15)
        fig.tight_layout(rect=[0.02,0.04,1,1])
        fig.savefig(
            "figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+"/"+str(predictors[i])+scenario+".png",dpi=144)


    else:
        fig.text(0.03, 0.5, 
        'Proportion of respondents\n micro-distancing', 
        ha='center', va='center',
        rotation='vertical',
        fontsize=15)
        fig.tight_layout(rect=[0.02,0.04,1,1])
        fig.savefig(
            "figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+"/micro_dist.png",dpi=144)

df_out = pd.DataFrame.from_dict(outdata)

df_md = df_out.loc[df_out.type=='md_prop']
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
#prop_std = pd.DataFrame(np.random.beta(1+survey_counts, 1+survey_respond), columns = survey_counts.columns, index = prop.index)

#df_md_std = pd.concat([prop_std,df_forecast_md_std.set_index('date')])

expo_decay=True
theta_md = np.tile(df_samples['theta_md'].values, (df_md['NSW'].shape[0],1))


fig, ax = plt.subplots(figsize=(12,9), nrows=4,ncols=2,sharex=True, sharey=True)

plt.locator_params(axis='x',nbins=2)

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
    ax[row,col].xaxis.set_major_locator(locator)

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
fig.savefig("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+"/md_factor.png",dpi=144)


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
for typ in forecast_type:
    state_R={}
    for state in states:
    #sort df_R by date so that rows are dates

        #rows are dates, columns are predictors
        df_state = df_R.loc[df_R.state==state]
        dd = df_state.date
        post_values = samples[predictors].values.T
        prop_sim = df_md[state].values

                    #take right size of md to be N by N
        theta_md = np.tile(samples['theta_md'].values, (df_state.shape[0],mob_samples))
        if expo_decay:
            md = ((1+theta_md).T**(-1* prop_sim)).T
        #else:
        #    md = (2*expit(-1*theta_md*prop_sim[:,np.newaxis]))

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
            if state in ("ACT","NT"):
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

        R_L = 2* md *sim_R * expit( logodds )

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

plt.locator_params(axis='x',nbins=2)
for i,state in enumerate(plot_states):

    row = i//2
    col = i%2

    plot_df = df_Rhats.loc[(df_Rhats.state==state)& (df_Rhats.type=='R_L')]

    ax[row,col].plot(plot_df.date, plot_df['mean'])

    ax[row,col].fill_between( plot_df.date, plot_df['lower'],plot_df['upper'],alpha=0.4,color='C0')
    ax[row,col].fill_between( plot_df.date, plot_df['bottom'],plot_df['top'],alpha=0.4,color='C0')

    ax[row,col].tick_params('x',rotation=45)
    ax[row,col].xaxis.set_major_locator(locator)
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
os.makedirs("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario, exist_ok=True)
plt.savefig("figs/mobility_forecasts/"+data_date.strftime("%Y-%m-%d")+scenario+"/soc_mob_R_L_hats"+data_date.strftime('%Y-%m-%d')+".png",dpi=102)

df_Rhats = df_Rhats[['state','date','type','median',
'bottom','lower','upper','top']+[i for i in range(2000)] ]
#df_Rhats.columns = ['state','date','type','median',
#'bottom','lower','upper','top']  + [i for i in range(1000)]



df_hdf = df_Rhats.loc[df_Rhats.type=='R_L']
df_hdf = df_hdf.append(df_Rhats.loc[(df_Rhats.type=='R_I')&(df_Rhats.date=='2020-03-01')])
df_hdf = df_hdf.append(df_Rhats.loc[(df_Rhats.type=='R_L0')&(df_Rhats.date=='2020-03-01')])
#df_Rhats.to_csv('./soc_mob_R'+today+'.csv')
df_hdf.to_hdf('results/soc_mob_R'+data_date.strftime('%Y-%m-%d')+scenario+'.h5',key='Reff')
