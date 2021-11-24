######### imports #########
from datetime import time, timedelta
from math import trunc
import sys
from Reff_constants import *
from Reff_functions import *
import glob
import os
from sys import argv
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arviz.utils import _var_names
import matplotlib
from numpy.random import sample
matplotlib.use('Agg')
from params import apply_vacc_to_R_L_hats, truncation_days, download_google_automatically, \
    run_inference_only, third_start_date, on_phoenix, testing_inference, run_inference
# depending on whether we are on phoenix or not changes the version of stan
if on_phoenix:
    import pystan
else:
    import stan

######### start #########
print('Performing inference on state level Reff')
data_date = pd.to_datetime(argv[1])  # Define data date
print("Data date is {}".format(data_date.strftime('%d%b%Y')))
fit_date = pd.to_datetime(data_date-timedelta(days=truncation_days))
print("Last date in fitting {}".format(fit_date.strftime('%d%b%Y')))
# note: 2020-09-09 won't work (for some reason)

######### Read in microdistancing (md) surveys #########
surveys = pd.DataFrame()
path = "data/md/Barometer wave*.csv"
for file in glob.glob(path):
    surveys = surveys.append(pd.read_csv(file, parse_dates=['date']))

surveys = surveys.sort_values(by='date')
print("Latest Microdistancing survey is {}".format(surveys.date.values[-1]))

surveys.loc[surveys.state != 'ACT', 'state'] = surveys.loc[surveys.state != 'ACT', 'state'].map(states_initials).fillna(surveys.loc[surveys.state != 'ACT', 'state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

always = surveys.loc[surveys.response == 'Always'].set_index(["state", 'date'])
always = always.unstack(['state'])
# If you get an error here saying 'cannot create a new series when the index is not unique', then you have a duplicated md file.

idx = pd.date_range('2020-03-01', pd.to_datetime("today"))
always = always.reindex(idx, fill_value=np.nan)
always.index.name = 'date'

# fill back to earlier and between weeks.
# Assume survey on day x applies for all days up to x - 6
always = always.fillna(method='bfill')
# assume values continue forward if survey hasn't completed
always = always.fillna(method='ffill')
always = always.stack(['state'])

# Zero out before first survey 20th March
always = always.reset_index().set_index('date')
always.loc[:'2020-03-20', 'count'] = 0
always.loc[:'2020-03-20', 'respondents'] = 0
always.loc[:'2020-03-20', 'proportion'] = 0

always = always.reset_index().set_index(['state', 'date'])

survey_X = pd.pivot_table(data=always, index='date', columns='state', values='proportion')
survey_counts_base = pd.pivot_table(data=always, index='date', columns='state', values='count').drop(['Australia', 'Other'], axis=1).astype(int)

survey_respond_base = pd.pivot_table(data=always, index='date', columns='state', values='respondents').drop(['Australia', 'Other'], axis=1).astype(int)

######### Read in EpyReff results #########
df_Reff = pd.read_csv("results/EpyReff/Reff" + data_date.strftime("%Y-%m-%d")+"tau_4.csv", parse_dates=['INFECTION_DATES'])
# df_Reff = pd.read_csv("results/EpyReff/Reff" + data_date.strftime("%Y-%m-%d")+"tau_2.csv", parse_dates=['INFECTION_DATES'])
df_Reff['date'] = df_Reff.INFECTION_DATES
df_Reff['state'] = df_Reff.STATE

######### Read in NNDSS/linelist data #########
# If this errors it may be missing a leading zero on the date.
df_state = read_in_cases(case_file_date=data_date.strftime('%d%b%Y'), 
                         apply_delay_at_read=True, 
                         apply_inc_at_read=True)

df_Reff = df_Reff.merge(df_state, how='left', left_on=['state', 'date'], right_on=['STATE', 'date_inferred'])  # how = left to use Reff days, NNDSS missing dates
df_Reff['rho_moving'] = df_Reff.groupby(['state'])['rho'].transform(lambda x: x.rolling(7, 1).mean())  # minimum number of 1

# some days have no cases, so need to fillna
df_Reff['rho_moving'] = df_Reff.rho_moving.fillna(method='bfill')

# counts are already aligned with infection date by subtracting a random incubation period
df_Reff['local'] = df_Reff.local.fillna(0)
df_Reff['imported'] = df_Reff.imported.fillna(0)

######### Read in Google mobility results #########
sys.path.insert(0, '../')

df_google = read_in_google(local=not download_google_automatically, moving=True)
df = df_google.merge(df_Reff[['date', 'state', 'mean', 'lower', 'upper', 'top', 'bottom', 'std', 'rho', 'rho_moving', 'local', 'imported']], on=['date', 'state'], how='inner')

######### Create useable dataset #########
# ACT and NT not in original estimates, need to extrapolated sorting keeps consistent with sort in data_by_state
# * Note that as we now consider the third wave for ACT, we include it in the third wave fitting only! 
states_to_fit_all_waves = sorted(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'ACT'])

first_states = sorted(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS'])
fit_post_March = True
ban = '2020-03-20'
start_date = '2020-03-01'
end_date = '2020-03-31'

# Second wave inputs
sec_states = sorted(['NSW', 'VIC'])
sec_start_date = '2020-06-01'
sec_end_date = '2021-01-19'

# Third wave inputs
third_states = sorted(['NSW', 'VIC', 'ACT', 'QLD'])
# third_states = sorted(['NSW', 'VIC', 'ACT'])
# third_states = sorted(['VIC'])
# Subtract the truncation days to avoid right truncation as we consider infection dates 
# and not symptom onset dates 
third_end_date = data_date - pd.Timedelta(days=truncation_days)

fit_mask = df.state.isin(first_states)
if fit_post_March:
    fit_mask = (fit_mask) & (df.date >= start_date)

fit_mask = (fit_mask) & (df.date <= end_date)

second_wave_mask = df.state.isin(sec_states)
second_wave_mask = (second_wave_mask) & (df.date >= sec_start_date)
second_wave_mask = (second_wave_mask) & (df.date <= sec_end_date)

# Add third wave stuff here
third_wave_mask = df.state.isin(third_states)
third_wave_mask = (third_wave_mask) & (df.date >= third_start_date)
third_wave_mask = (third_wave_mask) & (df.date <= third_end_date)

predictors = mov_values.copy()
# predictors.extend(['driving_7days','transit_7days','walking_7days','pc'])

# remove residential to see if it improves fit
predictors.remove('residential_7days')

df['post_policy'] = (df.date >= ban).astype(int)

dfX = df.loc[fit_mask].sort_values('date')
df2X = df.loc[second_wave_mask].sort_values('date')
df3X = df.loc[third_wave_mask].sort_values('date')

# choose dates for the first wave — this is kinda redundant but ensures a common format of 
# data between waves. 
first_date_range = {"NSW": pd.date_range(start='2020-03-01', end=end_date).values, 
                    "QLD": pd.date_range(start='2020-03-01', end=end_date).values, 
                    "SA": pd.date_range(start='2020-03-01', end=end_date).values, 
                    "TAS": pd.date_range(start='2020-03-01', end=end_date).values, 
                    "VIC": pd.date_range(start='2020-03-01', end=end_date).values, 
                    "WA": pd.date_range(start='2020-03-01', end=end_date).values}

# choose dates for each state for sec wave
sec_date_range = {'NSW': pd.date_range(start=sec_start_date, end='2021-01-19').values,
                  'VIC': pd.date_range(start=sec_start_date, end='2020-10-20').values}

# choose dates for each state for third wave
# third_date_range = {'ACT': pd.date_range(start='2021-08-16', end=third_end_date).values,
#                     'NSW': pd.date_range(start=third_start_date, end=third_end_date).values,
#                     'QLD': pd.date_range(start=third_start_date, end='2021-10-10').values,
#                     'VIC': pd.date_range(start=third_start_date, end=third_end_date).values}
# third_date_range = {'ACT': pd.date_range(start='2021-08-16', end='2021-11-14').values,  # truncate further to deal with delay
#                     'NSW': pd.date_range(start='2021-06-23', end=third_end_date).values,
#                     'QLD': pd.date_range(start='2021-07-30', end='2021-10-10').values,
#                     'VIC': pd.date_range(start='2021-07-14', end=third_end_date).values}
third_date_range = {'ACT': pd.date_range(start='2021-08-17', end='2021-11-14').values,          # truncate further to deal with delay
                    'NSW': pd.date_range(start=third_start_date, end=third_end_date).values,
                    'QLD': pd.date_range(start=third_start_date, end='2021-10-10').values,
                    'VIC': pd.date_range(start=third_start_date, end=third_end_date).values}

dfX['is_first_wave'] = 0
for state in first_states:
    dfX.loc[dfX.state == state, 'is_first_wave'] = dfX.loc[dfX.state == state].date.isin(first_date_range[state]).astype(int).values
    
df2X['is_sec_wave'] = 0
for state in sec_states:
    df2X.loc[df2X.state == state, 'is_sec_wave'] = df2X.loc[df2X.state == state].date.isin(sec_date_range[state]).astype(int).values

df3X['is_third_wave'] = 0
for state in third_states:
    df3X.loc[df3X.state == state, 'is_third_wave'] = df3X.loc[df3X.state == state].date.isin(third_date_range[state]).astype(int).values

data_by_state = {}
sec_data_by_state = {}
third_data_by_state = {}

for value in ['mean', 'std', 'local', 'imported']:
    data_by_state[value] = pd.pivot(dfX[['state', value, 'date']],
                                    index='date', 
                                    columns='state', 
                                    values=value).sort_index(axis='columns')
    
    # account for dates pre pre second wave
    if df2X.loc[df2X.state == sec_states[0]].shape[0] == 0:
        print("making empty")
        sec_data_by_state[value] = pd.DataFrame(columns=sec_states).astype(float)
    else: 
        sec_data_by_state[value] = pd.pivot(df2X[['state', value, 'date']],
                                            index='date', 
                                            columns='state', 
                                            values=value).sort_index(axis='columns')
    # account for dates pre pre third wave
    if df3X.loc[df3X.state == third_states[0]].shape[0] == 0:
        print("making empty")
        third_data_by_state[value] = pd.DataFrame(columns=third_states).astype(float)
    else:
        third_data_by_state[value] = pd.pivot(df3X[['state', value, 'date']],
                                              index='date', 
                                              columns='state', 
                                              values=value).sort_index(axis='columns')
        

# FIRST PHASE
mobility_by_state = []
mobility_std_by_state = []
count_by_state = []
respond_by_state = []

# filtering survey responses to dates before this wave fitting
survey_respond = survey_respond_base.loc[:dfX.date.values[-1]]
survey_counts = survey_counts_base.loc[:dfX.date.values[-1]]
include_in_first_wave = []

def exponential_smoother(x, alpha=1.0):
    """
    Smooths an input vector x using the standard exponential moving average with parameter alpha = 1.
    """
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1-alpha) * s[i-1]
        
    return s

for state in first_states:
    mobility_by_state.append(exponential_smoother(dfX.loc[dfX.state == state, predictors].values/100))
    mobility_std_by_state.append(exponential_smoother(dfX.loc[dfX.state == state, [val+'_std' for val in predictors]].values/100))
    count_by_state.append(exponential_smoother(survey_counts.loc[start_date:end_date, state].values))
    respond_by_state.append(exponential_smoother(survey_respond.loc[start_date:end_date, state].values))
    include_in_first_wave.append(exponential_smoother(dfX.loc[dfX.state == state, 'is_first_wave'].values))

# SECOND PHASE
sec_mobility_by_state = []
sec_mobility_std_by_state = []
sec_count_by_state = []
sec_respond_by_state = []
include_in_sec_wave = []

# filtering survey responses to dates before this wave fitting
survey_respond = survey_respond_base.loc[:df2X.date.values[-1]]
survey_counts = survey_counts_base.loc[:df2X.date.values[-1]]

for state in sec_states:
    sec_mobility_by_state.append(exponential_smoother(df2X.loc[df2X.state == state, predictors].values/100))
    sec_mobility_std_by_state.append(exponential_smoother(df2X.loc[df2X.state == state, [val+'_std' for val in predictors]].values/100))
    sec_count_by_state.append(exponential_smoother(survey_counts.loc[sec_start_date:sec_end_date, state].values))
    sec_respond_by_state.append(exponential_smoother(survey_respond.loc[sec_start_date:sec_end_date, state].values))
    include_in_sec_wave.append(exponential_smoother(df2X.loc[df2X.state == state, 'is_sec_wave'].values))

# THIRD WAVE
third_mobility_by_state = []
third_mobility_std_by_state = []
third_count_by_state = []
third_respond_by_state = []
include_in_third_wave = []

# filtering survey responses to dates before this wave fitting
survey_respond = survey_respond_base.loc[:df3X.date.values[-1]]
survey_counts = survey_counts_base.loc[:df3X.date.values[-1]]

for state in third_states:
    third_mobility_by_state.append(exponential_smoother(df3X.loc[df3X.state == state, predictors].values/100))
    third_mobility_std_by_state.append(exponential_smoother(df3X.loc[df3X.state == state, [val+'_std' for val in predictors]].values/100))
    third_count_by_state.append(exponential_smoother(survey_counts.loc[third_start_date:third_end_date, state].values))
    third_respond_by_state.append(exponential_smoother(survey_respond.loc[third_start_date:third_end_date, state].values))
    include_in_third_wave.append(exponential_smoother(df3X.loc[df3X.state == state, 'is_third_wave'].values))

# policy boolean flag for after travel ban in each wave
policy = dfX.loc[dfX.state == first_states[0],'post_policy']     # this is the post ban policy
policy_sec_wave = [1]*df2X.loc[df2X.state == sec_states[0]].shape[0]
policy_third_wave = [1]*df3X.loc[df3X.state == third_states[0]].shape[0]

######### loading and cleaning vaccine data #########

# Load in vaccination data by state and date
vaccination_by_state = pd.read_csv('data/vaccine_effect_timeseries_'+data_date.strftime('%Y-%m-%d')+'.csv', 
                                   parse_dates=['date'])
# there are a couple NA's early on in the time series but is likely due to slightly different start dates
vaccination_by_state.fillna(1, inplace=True)
vaccination_by_state = vaccination_by_state[['state', 'date', 'effect']]

# display the latest available date in the NSW data (will be the same date between states)
print("Latest date in vaccine data is {}".format(vaccination_by_state[vaccination_by_state.state == 'NSW'].date.values[-1]))

vaccination_by_state = vaccination_by_state[(vaccination_by_state.date >= third_start_date) & 
                                            (vaccination_by_state.date <= third_end_date)]  # Get only the dates we need.
vaccination_by_state = vaccination_by_state[vaccination_by_state['state'].isin(third_states)]  # Isolate fitting states
vaccination_by_state = vaccination_by_state.pivot(index='state', columns='date', values='effect')  # Convert to matrix form

# If we are missing recent vaccination data, fill it in with the most recent available data.
latest_vacc_data = vaccination_by_state.columns[-1]
if latest_vacc_data < pd.to_datetime(third_end_date):
    vaccination_by_state = pd.concat([vaccination_by_state]+
                                     [pd.Series(vaccination_by_state[latest_vacc_data], name=day) 
                                      for day in pd.date_range(start=latest_vacc_data, end=third_end_date)], axis=1)
    
# Convert to simple array only useful to pass to stan
vaccination_by_state_array = vaccination_by_state.to_numpy()

# elementwise comparison of the third states with NSW and then convert to int which is easier than 
# keeping track of indices in the stan code
is_VIC = (np.array(third_states) == 'VIC').astype(int)
is_NSW = (np.array(third_states) == 'NSW').astype(int)

# calculate how many days the end of august is after the third start date
decay_start_date_third = (pd.to_datetime('2021-08-20') - pd.to_datetime(third_start_date)).days

# Make state by state arrays
state_index = {state: i+1 for i, state in enumerate(states_to_fit_all_waves)}

third_wave_dates = pd.date_range(start=third_start_date,end=third_end_date)
VIC_tough_period = np.array((third_wave_dates >= pd.to_datetime('2021-07-13')) * 
                            (third_wave_dates <= pd.to_datetime('2021-08-01'))).astype(int)

# input data block for stan model
input_data = {
    'j_total': len(states_to_fit_all_waves),
    
    'N': dfX.loc[dfX.state == first_states[0]].shape[0],
    'K': len(predictors),
    'j_first_wave': len(first_states),
    'Reff': data_by_state['mean'].values,
    'Mob': mobility_by_state,
    'Mob_std': mobility_std_by_state,
    'sigma2': data_by_state['std'].values**2,
    'policy': policy.values,
    'local': data_by_state['local'].values,
    'imported': data_by_state['imported'].values,

    'N_sec_wave': df2X.loc[df2X.state == sec_states[0]].shape[0],
    'j_sec_wave': len(sec_states),
    'Reff_sec_wave': sec_data_by_state['mean'].values,
    'Mob_sec_wave': sec_mobility_by_state,
    'Mob_sec_wave_std': sec_mobility_std_by_state,
    'sigma2_sec_wave': sec_data_by_state['std'].values**2,
    'policy_sec_wave': policy_sec_wave,
    'local_sec_wave': sec_data_by_state['local'].values,
    'imported_sec_wave': sec_data_by_state['imported'].values,

    'N_third_wave': df3X.loc[df3X.state == third_states[0]].shape[0],
    'j_third_wave': len(third_states),
    'Reff_third_wave': third_data_by_state['mean'].values,
    'Mob_third_wave': third_mobility_by_state,
    'Mob_third_wave_std': third_mobility_std_by_state,
    'sigma2_third_wave': third_data_by_state['std'].values**2,
    'policy_third_wave': policy_third_wave,
    'local_third_wave': third_data_by_state['local'].values,
    'imported_third_wave': third_data_by_state['imported'].values,

    'count_md': count_by_state,
    'respond_md': respond_by_state,
    'count_md_sec_wave': sec_count_by_state,
    'respond_md_sec_wave': sec_respond_by_state,
    'count_md_third_wave': third_count_by_state,
    'respond_md_third_wave': third_respond_by_state,

    'map_to_state_index_first': [state_index[state] for state in first_states],
    'map_to_state_index_sec': [state_index[state] for state in sec_states],
    'map_to_state_index_third': [state_index[state] for state in third_states],
    # needed to convert this to primitive int for pystan
    'total_N_p_sec': sum([sum(x) for x in include_in_sec_wave]).item(),
    # needed to convert this to primitive int for pystan
    'total_N_p_third': sum([sum(x) for x in include_in_third_wave]).item(),
    
    # The include_in_..._wave variables are used for appropriate indexing inside of stan
    'include_in_first_wave': include_in_first_wave,
    'include_in_sec_wave': include_in_sec_wave,
    'include_in_third_wave': include_in_third_wave,
    'pos_starts_sec': np.cumsum([sum(x) for x in include_in_sec_wave]),
    'pos_starts_third': np.cumsum([sum(x) for x in include_in_third_wave]),

    'is_VIC': is_VIC,   # indicator for whether we are looking at ACT
    'VIC_tough_period': VIC_tough_period,
    'is_NSW': is_NSW,   # indicator for whether we are looking at NSW
    # days into third wave that we start return to homogoeneity in vaccination
    'decay_start_date_third': decay_start_date_third,
    'vaccine_effect_data': vaccination_by_state_array,  # the vaccination data
}

# make results dir
results_dir = "figs/soc_mob_posterior/"
os.makedirs(results_dir, exist_ok=True)

######### running inference #########
if testing_inference:
    num_chains = 2
    num_samples = 1000
else:
    num_chains = 4
    num_samples = 4000
    
# to run the inference set run_inference to True in params
if run_inference or run_inference_only:

    # import the stan model as a string
    model_file = open("model/fitting_and_forecasting/rho_model_gamma.stan", "r")
    rho_model_gamma = model_file.read()
    model_file.close()

    # slightly different setup depending if we are running on phoenix or locally due to
    # different versions of pystan
    if on_phoenix:
        sm_pol_gamma = pystan.StanModel(model_code=rho_model_gamma, 
                                        model_name='gamma_pol_state')
        fit = sm_pol_gamma.sampling(data=input_data, 
                                    iter=num_samples, 
                                    chains=num_chains)

        filename = "stan_posterior_fit" + data_date.strftime("%Y-%m-%d") + ".txt"
        with open(results_dir+filename, 'w') as f:
            print(fit.stansummary(pars=['bet', 'R_I', 'R_L', 'R_Li', 'theta_md', 'sig',
                                        'voc_effect_sec_wave', 'voc_effect_third_wave', 
                                        'eta_NSW', 'eta_other', 'r_NSW', 'r_other']), file=f)

        samples_mov_gamma = fit.to_dataframe(pars=['bet', 'R_I', 'R_L', 'R_Li', 'sig', 
                                                   'brho', 'theta_md', 'brho_sec_wave', 'brho_third_wave',
                                                   'voc_effect_sec_wave', 'voc_effect_third_wave', 
                                                   'eta_NSW', 'eta_other', 'r_NSW', 'r_other', 'TP_local_adjustment_factor'])
    else:

        # compile the stan model
        posterior = stan.build(rho_model_gamma, data=input_data)
        fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)

        ######### Saving Output #########

        filename = "stan_posterior_fit" + data_date.strftime("%Y-%m-%d") + ".txt"
        with open(results_dir+filename, 'w') as f:
            print(az.summary(fit, var_names=['bet', 'R_I', 'R_L', 'R_Li', 'theta_md', 'sig',
                                             'voc_effect_sec_wave', 'voc_effect_third_wave', 
                                             'eta_NSW', 'eta_other', 'r_NSW', 'r_other']), file=f)

        ######### now a hacky fix to put the data in the same format as before -- might break stuff in the future #########
        # create extended summary of parameters to index the samples by
        summary_df = az.summary(fit, var_names=['bet', 'R_I', 'R_L', 'R_Li', 'sig', 
                                                'brho', 'theta_md', 'brho_sec_wave', 'brho_third_wave',
                                                'voc_effect_sec_wave', 'voc_effect_third_wave', 
                                                'eta_NSW', 'eta_other', 'r_NSW', 'r_other', 'TP_local_adjustment_factor'])

        match_list_names = summary_df.index.to_list()

        # extract the names of the constrained parameters which are the ones we actually sample
        names = fit.constrained_param_names
        df_fit = fit.to_frame()

        for name in names:
            dot_pos = name.find('.')
            if dot_pos != -1:
                var_name = name[:dot_pos]
                num_name = name[(dot_pos+1):]
                dot_pos2 = num_name.find('.')
                if dot_pos2 != -1:
                    num_name1 = int(num_name[:dot_pos2]) - 1
                    num_name2 = int(num_name[(dot_pos2+1):]) - 1
                    updated_name = var_name + '[' + str(num_name1) + ',' + str(num_name2) + ']'
                else:
                    num_name = int(num_name) - 1
                    updated_name = var_name + '[' + str(num_name) + ']'

            else:
                updated_name = name

            df_fit = df_fit.rename(columns={name: updated_name})

        # produces dataframe with variables matching those needed
        df_fit = df_fit.loc[:, match_list_names]

        names = df_fit.columns

        updated_names = []

        # now we need to rename one more time because the naming convention is so dumb
        for name in names:
            bracket1_pos = name.find('[')
            bracket2_pos = name.find(']')
            if bracket1_pos != -1:
                var_name = name[:bracket1_pos]
                # now we check whether the thing we are indexing is a matrix and if so we want to increase
                # the labels by 1. this is just because python's indexing starts at 0 but the labelling used
                # is 1, 2, ...
                comma_pos = name.find(',')
                if comma_pos != -1:
                    num_name1 = int(name[(bracket1_pos+1):comma_pos]) + 1
                    num_name2 = int(name[(comma_pos+1):(bracket2_pos)]) + 1
                    updated_name = var_name + '[' + str(num_name1) + ',' + str(num_name2) + ']'
                else:
                    num_name = int(name[(bracket1_pos+1):bracket2_pos]) + 1
                    updated_name = var_name + '[' + str(num_name) + ']'
            else:
                updated_name = name

            updated_names.append(updated_name)

        names = names.to_list()
        name_updates = {}

        for i in range(np.size(names)):
            name_updates.update({names[i]: updated_names[i]})

        df_fit_new = df_fit.rename(columns=name_updates)

        # we save the df to csv so we have it
        df_fit_new.to_csv("results/samples_mov_gamma.csv")
        # we read it right back in to fix formatting
        samples_mov_gamma = pd.read_csv("results/samples_mov_gamma.csv")

# decide what to do next based on whether we want to plot
if run_inference_only:
    sys.exit()
elif not on_phoenix:
    # we read it right back in to fix formatting
    samples_mov_gamma = pd.read_csv("results/samples_mov_gamma.csv")

######### plotting the results #########
######### ratio of imported to local cases #########

# First phase
# rho calculated at data entry
if isinstance(df_state.index, pd.MultiIndex):
    df_state = df_state.reset_index()

states = sorted(['NSW', 'QLD', 'VIC', 'TAS', 'SA', 'WA', 'ACT', 'NT'])
fig, ax = plt.subplots(figsize=(24, 9), ncols=len(states), sharey=True)

states_to_fitd = {state: i+1 for i, state in enumerate(first_states)}

for i, state in enumerate(states):
    if state in first_states:
        dates = df_Reff.loc[(df_Reff.date >= start_date) & (df_Reff.state == state) & (df_Reff.date <= end_date)].date
        rho_samples = samples_mov_gamma[['brho['+str(j+1)+','+str(states_to_fitd[state])+']' 
                                         for j in range(dfX.loc[dfX.state == first_states[0]].shape[0])]]
        ax[i].plot(dates, rho_samples.median(), label='fit', color='C0')
        ax[i].fill_between(dates, rho_samples.quantile(0.25), rho_samples.quantile(0.75), color='C0', alpha=0.4)

        ax[i].fill_between(dates, rho_samples.quantile(0.05), rho_samples.quantile(0.95), color='C0', alpha=0.4)
    else:
        sns.lineplot(x='date_inferred', y='rho',
                     data=df_state.loc[(df_state.date_inferred >= start_date) & 
                                       (df_state.STATE == state) & 
                                       (df_state.date_inferred <= end_date)], 
                     ax=ax[i], color='C1', label='data')
        
    sns.lineplot(x='date', y='rho',
                 data=df_Reff.loc[(df_Reff.date >= start_date) & 
                                  (df_Reff.state == state) & 
                                  (df_Reff.date <= end_date)], 
                 ax=ax[i], color='C1', label='data')
    sns.lineplot(x='date', y='rho_moving',
                 data=df_Reff.loc[(df_Reff.date >= start_date) & 
                                  (df_Reff.state == state) & 
                                  (df_Reff.date <= end_date)], 
                 ax=ax[i], color='C2', label='moving')

    dates = dfX.loc[dfX.state == first_states[0]].date

    ax[i].tick_params('x', rotation=90)
    ax[i].xaxis.set_major_locator(plt.MaxNLocator(4))
    ax[i].set_title(state)
    
ax[0].set_ylabel('Proportion of imported cases')
plt.legend()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d") + "rho_first_phase.png", dpi=144)

# Second phase
if df2X.shape[0] > 0:
    fig, ax = plt.subplots(figsize=(24, 9), ncols=len(sec_states), sharey=True, squeeze=False)
    states_to_fitd = {state: i+1 for i, state in enumerate(sec_states)}
    pos = 1
    for i, state in enumerate(sec_states):
        # Google mobility only up to a certain date, so take only up to that value
        dates = df2X.loc[(df2X.state == state) & (df2X.is_sec_wave == 1)].date.values
        rho_samples = samples_mov_gamma[['brho_sec_wave['+str(j)+']'
                                         for j in range(pos, pos+df2X.loc[df2X.state == state].is_sec_wave.sum())]]
        pos = pos + df2X.loc[df2X.state == state].is_sec_wave.sum()

        ax[0, i].plot(dates, rho_samples.median(), label='fit', color='C0')
        ax[0, i].fill_between(dates, rho_samples.quantile(0.25), rho_samples.quantile(0.75), color='C0', alpha=0.4)

        ax[0, i].fill_between(dates, rho_samples.quantile(0.05), rho_samples.quantile(0.95), color='C0', alpha=0.4)

        sns.lineplot(x='date_inferred', y='rho',
                     data=df_state.loc[(df_state.date_inferred >= sec_start_date) & 
                                       (df_state.STATE == state) & 
                                       (df_state.date_inferred <= sec_end_date)], 
                     ax=ax[0, i], color='C1', label='data')
        sns.lineplot(x='date', y='rho',
                     data=df_Reff.loc[(df_Reff.date >= sec_start_date) & 
                                      (df_Reff.state == state) & 
                                      (df_Reff.date <= sec_end_date)], 
                     ax=ax[0, i], color='C1', label='data')
        sns.lineplot(x='date', y='rho_moving',
                     data=df_Reff.loc[(df_Reff.date >= sec_start_date) & 
                                      (df_Reff.state == state) & 
                                      (df_Reff.date <= sec_end_date)], 
                     ax=ax[0, i], color='C2', label='moving')

        dates = dfX.loc[dfX.state == sec_states[0]].date

        ax[0, i].tick_params('x', rotation=90)
        ax[0, i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[0, i].set_title(state)
        
    ax[0, 0].set_ylabel('Proportion of imported cases')
    plt.legend()
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d") + "rho_sec_phase.png", dpi=144)

# Third  phase
if df3X.shape[0] > 0:
    fig, ax = plt.subplots(figsize=(24, 9), ncols=len(third_states), sharey=True, squeeze=False)
    states_to_fitd = {state: i+1 for i, state in enumerate(third_states)}
    pos = 1
    for i, state in enumerate(third_states):
        # Google mobility only up to a certain date, so take only up to that value
        dates = df3X.loc[(df3X.state == state) & (df3X.is_third_wave == 1)].date.values
        rho_samples = samples_mov_gamma[['brho_third_wave['+str(j)+']'
                                         for j in range(pos, pos+df3X.loc[df3X.state == state].is_third_wave.sum())]]
        pos = pos + df3X.loc[df3X.state == state].is_third_wave.sum()
        
        ax[0, i].plot(dates, rho_samples.median(), label='fit', color='C0')
        ax[0, i].fill_between(dates, rho_samples.quantile(0.25), rho_samples.quantile(0.75), color='C0', alpha=0.4)

        ax[0, i].fill_between(dates, rho_samples.quantile(0.05), rho_samples.quantile(0.95), color='C0', alpha=0.4)

        sns.lineplot(x='date_inferred', y='rho',
                     data=df_state.loc[(df_state.date_inferred >= third_start_date) & 
                                       (df_state.STATE == state) & 
                                       (df_state.date_inferred <= third_end_date)], 
                     ax=ax[0, i], color='C1', label='data')
        sns.lineplot(x='date', y='rho',
                     data=df_Reff.loc[(df_Reff.date >= third_start_date) & 
                                      (df_Reff.state == state) & 
                                      (df_Reff.date <= third_end_date)], 
                     ax=ax[0, i], color='C1', label='data')
        sns.lineplot(x='date', y='rho_moving',
                     data=df_Reff.loc[(df_Reff.date >= third_start_date) & 
                                      (df_Reff.state == state) & 
                                      (df_Reff.date <= third_end_date)], 
                     ax=ax[0, i], color='C2', label='moving')

        dates = dfX.loc[dfX.state == third_states[0]].date

        ax[0, i].tick_params('x', rotation=90)
        ax[0, i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[0, i].set_title(state)
        
    ax[0, 0].set_ylabel('Proportion of imported cases')
    plt.legend()
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d") + "rho_third_phase.png", dpi=144)

######### plotting marginal posterior distributions of R0's #########

fig, ax = plt.subplots(figsize=(12, 9))

# sample from the priors for RL and RI 
samples_mov_gamma['R_L_prior'] = np.random.gamma(1.8*1.8/0.05, 0.05/1.8, size=samples_mov_gamma.shape[0])
samples_mov_gamma['R_I_prior'] = np.random.gamma(0.5**2/0.2, .2/0.5, size=samples_mov_gamma.shape[0])

samples_mov_gamma['R_L_national'] = np.random.gamma(samples_mov_gamma.R_L.values ** 2 / samples_mov_gamma.sig.values,
                                                    samples_mov_gamma.sig.values / samples_mov_gamma.R_L.values)

df_R_values = pd.melt(samples_mov_gamma[[col for col in samples_mov_gamma if 'R' in col]])
print(df_R_values.variable.unique())

sns.violinplot(x='variable', y='value',
               data=pd.melt(samples_mov_gamma[[col for col in samples_mov_gamma if 'R' in col]]),
               ax=ax,
               cut=0)

ax.set_yticks([1], minor=True,)
ax.set_yticks([0, 2, 3], minor=False)
ax.set_yticklabels([0, 2, 3], minor=False)
ax.set_ylim((0, 3))
# state labels in alphabetical
ax.set_xticklabels(['R_I', 'R_L0 mean', 'R_L0 ACT', 'R_L0 NSW', 
                    'R_L0 QLD', 'R_L0 SA', 'R_L0 TAS', 'R_L0 VIC', 'R_L0 WA',
                    'R_L0 prior', 'R_I prior', 'R_L0 national'])
ax.set_xlabel('')
ax.set_ylabel('Effective reproduction number')
ax.tick_params('x', rotation=90)
ax.yaxis.grid(which='minor', linestyle='--', color='black', linewidth=2)
plt.tight_layout()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"R_priors.png", dpi=144)

# Making a new figure that doesn't include the priors
fig, ax = plt.subplots(figsize=(12, 9))

small_plot_cols = ['R_Li['+str(i)+']' for i in range(1,8)] + ['R_I']

sns.violinplot(x='variable', y='value',
               data=pd.melt(samples_mov_gamma[small_plot_cols]),
               ax=ax, cut=0)

ax.set_yticks([1], minor=True,)
ax.set_yticks([0, 2, 3], minor=False)
ax.set_yticklabels([0, 2, 3], minor=False)
ax.set_ylim((0, 3))
# state labels in alphabetical
ax.set_xticklabels(['$R_L0$ ACT', '$R_L0$ NSW', '$R_L0$ QLD', '$R_L0$ SA',
                   '$R_L0$ TAS', '$R_L0$ VIC', '$R_L0$ WA', '$R_I$'])
ax.tick_params('x', rotation=90)
ax.set_xlabel('')
ax.set_ylabel('Effective reproduction number')
ax.yaxis.grid(which='minor', linestyle='--', color='black', linewidth=2)
plt.tight_layout()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d") + "R_priors_(without_priors).png", dpi=288)

######### plotting figures for vaccine and voc effects #########

# Making a new figure that doesn't include the priors
fig, ax = plt.subplots(figsize=(12, 9))

small_plot_cols = ['voc_effect_third_wave', 'eta_NSW', 'eta_other', 'r_NSW', 'r_other']

sns.violinplot(x='variable', y='value',
               data=pd.melt(samples_mov_gamma[small_plot_cols]),
               ax=ax, cut=0)

ax.set_yticks([1], minor=True,)
ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3], minor=False)
ax.set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3], minor=False)
ax.set_ylim((0, 3))
# state labels in alphabetical
ax.set_xticklabels(['VoC 3rd wave', '$\eta$ (NSW)', '$\eta$ (not NSW)', '$r$ (NSW)', '$r$ (not NSW)'])
ax.tick_params('x', rotation=90)
ax.set_xlabel('')
ax.set_ylabel('value')
ax.yaxis.grid(which='minor', linestyle='--', color='black', linewidth=2)
plt.tight_layout()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d") + "voc_vaccine_effect_posteriors.png", dpi=288)

######### plotting mobility coefficients #########

posterior = samples_mov_gamma[['bet['+str(i)+']' for i in range(1, 1+len(predictors))]]

split = True
md = 'power'  # samples_mov_gamma.md.values

posterior.columns = [val for val in predictors]
long = pd.melt(posterior)

fig, ax2 = plt.subplots(figsize=(12, 9))

ax2 = sns.violinplot(x='variable', y='value',
                     data=long, ax=ax2, color='C0')

ax2.plot([0]*len(predictors), linestyle='dashed', alpha=0.6, color='grey')
ax2.tick_params(axis='x', rotation=90)

ax2.set_title('Coefficients of mobility indices')
ax2.set_xlabel('Social mobility index')
ax2.set_xticklabels([var[:-6] for var in predictors])
ax2.set_xticklabels(['Retail and Recreation', 'Grocery and Pharmacy', 'Parks', 'Transit Stations', 'Workplaces'])
ax2.tick_params('x', rotation=15)
plt.tight_layout()

plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+'mobility_posteriors.png', dpi=288)

######### generating and plotting TP plots #########
RL_by_state = {state: samples_mov_gamma['R_Li['+str(i)+']'].values for state, i in state_index.items()}
ax3 = predict_plot(samples_mov_gamma, df.loc[(df.date >= start_date) & 
                                             (df.date <= end_date)], 
                   gamma=True, moving=True, split=split, grocery=True, ban=ban,
                   R=RL_by_state, var=True, md_arg=md,
                   rho=first_states, R_I=samples_mov_gamma.R_I.values,
                   prop=survey_X.loc[start_date:end_date]) 
for ax in ax3:
    for a in ax:
        a.set_ylim((0, 3))
        a.set_xlim((pd.to_datetime(start_date), pd.to_datetime(end_date)))
        
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"total_Reff_allstates.png", dpi=144)

if df2X.shape[0] > 0:
    df['is_sec_wave'] = 0
    for state in sec_states:
        df.loc[df.state == state, 'is_sec_wave'] = df.loc[df.state == state].date.isin(sec_date_range[state]).astype(int).values
    # plot only if there is second phase data - have to have second_phase=True
    ax4 = predict_plot(samples_mov_gamma, df.loc[(df.date >= sec_start_date) & 
                                                 (df.date <= sec_end_date)], 
                       gamma=True, moving=True, split=split, grocery=True, ban=ban,
                       R=RL_by_state, var=True, md_arg=md,
                       rho=sec_states, second_phase=True,
                       R_I=samples_mov_gamma.R_I.values, prop=survey_X.loc[sec_start_date:sec_end_date])  
    for ax in ax4:
        for a in ax:
            a.set_ylim((0, 3))
            
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"Reff_sec_phase.png", dpi=144)

    # remove plots from memory
    fig.clear()
    plt.close(fig)

if apply_vacc_to_R_L_hats:
    # Load in vaccination data by state and date and this time do NOT isolate by fitting states
    vaccination_by_state = pd.read_csv('data/vaccine_effect_timeseries_'+data_date.strftime('%Y-%m-%d')+'.csv', 
                                       parse_dates=['date'])
    # there are a couple NA's early on in the time series but is likely due to slightly different start dates
    vaccination_by_state.fillna(1, inplace=True)
    vaccination_by_state = vaccination_by_state[['state', 'date', 'effect']]

    third_end_date = pd.to_datetime(data_date) - pd.Timedelta(days=truncation_days)
    vaccination_by_state = vaccination_by_state[(vaccination_by_state.date >= third_start_date) & 
                                                (vaccination_by_state.date <= third_end_date)]  # Get only the dates we need.
    vaccination_by_state = vaccination_by_state.pivot(index='state', columns='date', values='effect')  # Convert to matrix form

    # If we are missing recent vaccination data, fill it in with the most recent available data.
    latest_vacc_data = vaccination_by_state.columns[-1]
    if latest_vacc_data < pd.to_datetime(third_end_date):
        vaccination_by_state = pd.concat([vaccination_by_state]+
                                         [pd.Series(vaccination_by_state[latest_vacc_data], name=day) 
                                          for day in pd.date_range(start=latest_vacc_data, end=third_end_date)], 
                                         axis=1)

if df3X.shape[0] > 0:
    df['is_third_wave'] = 0
    for state in third_states:
        df.loc[df.state == state, 'is_third_wave'] = df.loc[df.state == state].date.isin(third_date_range[state]).astype(int).values    
    
    # plot only if there is third phase data - have to have third_phase=True
    ax4 = predict_plot(samples_mov_gamma, df.loc[(df.date >= third_start_date) & 
                                                 (df.date <= third_end_date)],
                       gamma=True, moving=True, split=split, grocery=True, ban=ban,
                       R=RL_by_state, var=True, md_arg=md, rho=third_states, third_phase=True,
                       R_I=samples_mov_gamma.R_I.values,
                       prop=survey_X.loc[third_start_date:third_end_date], vaccination=vaccination_by_state)  # by states....
    for ax in ax4:
        for a in ax:
            a.set_ylim((0, 3))
            # a.set_xlim((start_date,end_date))
            
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"Reff_third_phase.png", dpi=144)

    # remove plots from memory
    fig.clear()
    plt.close(fig)

######### plotting the inferred vaccine effect trajectory #########

dates = vaccination_by_state.columns

fig, ax = plt.subplots(figsize=(15, 12), ncols=2, nrows=4, sharey=True, sharex=True)


# find days after the third start date began that we want to apply the effect — currently this is fixed from the
# 20th of Aug and is not a problem with ACT as this is just a plot of the posterior vaccine effect
heterogeneity_delay_start_day = (pd.to_datetime('2021-08-20') - pd.to_datetime(third_start_date)).days
for i, state in enumerate(states):

    # apply different vaccine form depending on if NSW
    if state == 'NSW':
        eta = samples_mov_gamma.eta_NSW
        r = samples_mov_gamma.r_NSW
    else:
        eta = samples_mov_gamma.eta_other
        r = samples_mov_gamma.r_other

    # tile the states vaccination data from Curtin
    vacc_tmp = np.tile(vaccination_by_state.loc[state], (samples_mov_gamma.shape[0], 1)).T
    # create zero vector to fill in with vaccine effect
    vacc_eff = np.zeros_like(vacc_tmp)

    # loop ober days in third wave and apply the appropriate form (i.e. decay or not)
    # note that in here we apply the entire sample to the vaccination data to create a days by samples array
    for ii in range(vacc_tmp.shape[0]):
        if ii < heterogeneity_delay_start_day:
            vacc_eff[ii] = eta + (1-eta)*vacc_tmp[ii]
        else:
            # number of days after the heterogeneity should start to wane
            heterogeneity_delay_days = ii - heterogeneity_delay_start_day
            decay_factor = np.exp(-r*heterogeneity_delay_days)
            vacc_eff[ii] = eta*decay_factor + (1-eta*decay_factor)*vacc_tmp[ii]

    row = i % 4
    col = i//4

    ax[row, col].plot(dates, vaccination_by_state.loc[state].values, label='data', color='C1')
    ax[row, col].plot(dates, np.median(vacc_eff, axis=1), label='fit', color='C0')
    ax[row, col].fill_between(dates, 
                              np.quantile(vacc_eff, 0.25, axis=1), 
                              np.quantile(vacc_eff, 0.75, axis=1), color='C0', alpha=0.4)
    ax[row, col].fill_between(dates, 
                              np.quantile(vacc_eff, 0.05, axis=1), 
                              np.quantile(vacc_eff, 0.95, axis=1), color='C0', alpha=0.4)
    ax[row, col].set_title(state)
    ax[row, col].tick_params(axis='x', rotation=90)

# ax[0, 0].set_ylabel('reduction in TP from vaccination')
ax[1, 0].set_ylabel('reduction in TP from vaccination')

plt.savefig(results_dir+data_date.strftime("%Y-%m-%d") + "vaccine_reduction_in_TP.png", dpi=144)

######### saving the final processed posterior samples to h5 for generate_RL_forecasts.py #########

var_to_csv = predictors
samples_mov_gamma[predictors] = samples_mov_gamma[['bet['+str(i)+']' 
                                                   for i in range(1, 1+len(predictors))]]
var_to_csv = ['R_I', 'R_L', 'sig', 'theta_md', 'voc_effect_third_wave', 'eta_NSW', 'eta_other', 'r_NSW', 'r_other']
var_to_csv = var_to_csv + predictors + ['R_Li['+str(i+1)+']' 
                                        for i in range(len(states_to_fit_all_waves))]
var_to_csv = var_to_csv + ['TP_local_adjustment_factor['+str(j)+']' 
                           for j in range(1, 1+df.loc[df.state == 'VIC'].is_third_wave.sum())]

samples_mov_gamma[var_to_csv].to_hdf('results/soc_mob_posterior'+data_date.strftime("%Y-%m-%d")+'.h5', key='samples') 

