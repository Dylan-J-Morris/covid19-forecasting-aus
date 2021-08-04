print('Performing inference on state level Reff')

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
import pystan
import os, glob
from Reff_functions import *
from Reff_constants import *

## Define pystan model
rho_model_gamma = """
data {
    // data for the initial wave
    int N;                                                      //data length num days
    int K;                                                      //Number of mobility indices
    int j;                                                      //Number of states
    matrix[N,j] Reff;                                           //response
    matrix[N,K] Mob[j];                                         //Mobility indices
    matrix[N,K] Mob_std[j];                                     //std of mobility
    matrix[N,j] sigma2;                                         //Variances of R_eff from previous study
    vector[N] policy;                                           //Indicators for post policy or not
    matrix[N,j] local;                                          //local number of cases
    matrix[N,j] imported;                                       //imported number of cases

    // data for the secondary wave
    int N_sec_wave;                                             //length of VIC days
    int j_sec_wave;                                             //second wave states
    matrix[N_sec_wave,j_sec_wave] Reff_sec_wave;                //Reff for VIC in June
    matrix[N_sec_wave,K] Mob_sec_wave[j_sec_wave];              //Mob for VIC June
    matrix[N_sec_wave,K] Mob_sec_wave_std[j_sec_wave];          // std of mobility
    matrix[N_sec_wave,j_sec_wave] sigma2_sec_wave;              // variance of R_eff from previous study
    vector[N_sec_wave] policy_sec_wave;                         // micro distancing compliance
    matrix[N_sec_wave,j_sec_wave] local_sec_wave;               //local cases in VIC
    matrix[N_sec_wave,j_sec_wave] imported_sec_wave;            //imported cases in VIC

    // data for the third wave -- 
    int N_third_wave;                                           //length of VIC days
    int j_third_wave;                                           //thirdond wave states
    matrix[N_third_wave,j_third_wave] Reff_third_wave;          //Reff for VIC in June
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];        //Mob for VIC June
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];    // std of mobility
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;        // variance of R_eff from previous study
    vector[N_third_wave] policy_third_wave;                     // micro distancing compliance
    matrix[N_third_wave,j_third_wave] local_third_wave;         //local cases in VIC
    matrix[N_third_wave,j_third_wave] imported_third_wave;      //imported cases in VIC
    

    // data relating to mobility and microdistancing
    vector[N] count_md[j];                                      //count of always
    vector[N] respond_md[j];                                    // num respondants
    vector[N_sec_wave] count_md_sec_wave[j_sec_wave];           //count of always
    vector[N_sec_wave] respond_md_sec_wave[j_sec_wave];         // num respondants
    vector[N_third_wave] count_md_third_wave[j_third_wave];     //count of always
    vector[N_third_wave] respond_md_third_wave[j_third_wave];   // num respondants

    int map_to_state_index_sec[j_sec_wave];                     // indices of second wave to map to first
    int map_to_state_index_third[j_third_wave];                 // indices of second wave to map to first
    int total_N_p_sec;                                          //total number of data in sec wave, entire state first
    int total_N_p_third;                                        //total number of data in sec wave, entire state first
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];         // dates include in sec_wave 
    vector[N_third_wave] include_in_third_wave[j_third_wave];   // dates include in sec_wave 
    int pos_starts_sec[j_sec_wave];                             //starting positions for each state in the second wave
    int pos_starts_third[j_third_wave];                         //starting positions for each state in the third wave 
}
parameters {
    vector[K] bet;                                              //coefficients
    real<lower=0> R_I;                                          //base level imports,
    real<lower=0> R_L;                                          //base level local
    vector<lower=0>[j] R_Li;                                    //state level estimates
    real<lower=0> sig;                                          //state level variance
    real<lower=0> theta_md;                                     // md weighting
    matrix<lower=0,upper=1>[N,j] prop_md;                       // proportion who are md'ing
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    matrix<lower=0,upper=1>[N,j] brho;                          //estimate of proportion of imported cases
    matrix<lower=0,upper=1>[N,K] noise[j];
    //real<lower=0> R_temp;

    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;       //estimate of proportion of imported cases
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;   //estimate of proportion of imported cases
}
transformed parameters {
    matrix<lower=0>[N,j] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    matrix<lower=0>[N,j] md;                                    //micro distancing
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;

    for (i in 1:j) {
        for (n in 1:N){
            md[n,i] = pow(1+theta_md , -1*prop_md[n,i]);
            //mean estimate
            mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*2*R_Li[i]*(
            (1-policy[n]) + md[n,i]*policy[n] )*inv_logit(
            Mob[i][n,:]*(bet)); 
        }
    }
    for (i in 1:j_sec_wave){
        int pos;
        if (i==1){
            pos=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos =pos_starts_sec[i-1]+1;
            }
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n]==1){
                md_sec_wave[pos] = pow(1+theta_md ,-1*prop_md_sec_wave[pos]);
                if (map_to_state_index_sec[i] == 5) {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*(2*R_Li[
                    map_to_state_index_sec[i]
                    ])*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet)
                    ); //mean estimate
                }
                else {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*2*R_Li[
                    map_to_state_index_sec[i]
                    ]*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet)); //mean estimate
                }
                pos += 1;
            }
        }
    }
    for (i in 1:j_third_wave){
        int pos;
        if (i==1){
            pos=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos =pos_starts_third[i-1]+1;
            }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                md_third_wave[pos] = pow(1+theta_md ,-1*prop_md_third_wave[pos]);
                if (map_to_state_index_third[i] == 5) {
                    mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*(2*R_Li[
                    map_to_state_index_third[i]
                    ])*(
                    (1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n] )*inv_logit(
                    Mob_third_wave[i][n,:]*(bet)
                    ); //mean estimate
                }
                else {
                    mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*2*R_Li[
                    map_to_state_index_third[i]
                    ]*(
                    (1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n] )*inv_logit(
                    Mob_third_wave[i][n,:]*(bet)); //mean estimate
                }
                pos += 1;
            }
        }
    }

}
model {
    int pos2;
    bet ~ normal(0,1);
    theta_md ~ lognormal(0,0.5);
    //md ~ beta(7,3);

    R_L ~ gamma(1.8*1.8/0.05,1.8/0.05); //hyper-prior
    R_I ~ gamma(0.5*0.5/.2,0.5/.2);
    sig ~ exponential(20); //mean is 1/50=0.02
    R_Li ~ gamma( R_L*R_L/sig, R_L/sig); //partial pooling of state level estimates
    for (i in 1:j) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1+ respond_md[i][n] - count_md[i][n]);
            brho[n,i] ~ beta( 1+ imported[n,i], 1+ local[n,i]); //ratio imported/ (imported + local)
            //noise[i][n,:] ~ normal( Mob[i][n,:] , Mob_std[i][n,:]);
            mu_hat[n,i] ~ gamma( Reff[n,i]*Reff[n,i]/(sigma2[n,i]), Reff[n,i]/sigma2[n,i]); //Stan uses shape/inverse scale
        }
    }
    
    for (i in 1:j_sec_wave){
        if (i==1){
            pos2=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos2 =pos_starts_sec[i-1]+1; 
            }
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n]==1){
                prop_md_sec_wave[pos2] ~ beta(1 + count_md_sec_wave[i][n], 1+ respond_md_sec_wave[i][n] - count_md_sec_wave[i][n]);
                brho_sec_wave[pos2] ~ beta( 1+ imported_sec_wave[n,i], 1+ local_sec_wave[n,i]); //ratio imported/ (imported + local)
                //noise_sec_wave[i][n,:] ~ normal( Mob_sec_wave[i][n,:] , Mob_sec_wave_std[i][n,:]);
                mu_hat_sec_wave[pos2] ~ gamma( Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
                pos2+=1;
            }
        }
    }

    for (i in 1:j_third_wave){
        if (i==1){
            pos2=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos2 =pos_starts_third[i-1]+1; 
            }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                prop_md_third_wave[pos2] ~ beta(1 + count_md_third_wave[i][n], 1+ respond_md_third_wave[i][n] - count_md_third_wave[i][n]);
                brho_third_wave[pos2] ~ beta( 1+ imported_third_wave[n,i], 1+ local_third_wave[n,i]); //ratio imported/ (imported + local)
                //noise_third_wave[i][n,:] ~ normal( Mob_third_wave[i][n,:] , Mob_third_wave_std[i][n,:]);
                mu_hat_third_wave[pos2] ~ gamma( Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
                pos2+=1;
            }
        }
    }
}
"""

sm_pol_gamma = pystan.StanModel(
    model_code = rho_model_gamma,
    model_name ='gamma_pol_state'
)

data_date = pd.to_datetime(argv[1]) # Define data date
print(data_date.strftime('%d%b%Y'))
# note: 2020-09-09 won't work (for some reason)


######### Read in microdistancing (md) surveys #########
surveys = pd.DataFrame()
path = "data/md/Barometer wave*.csv"
for file in glob.glob(path):
    surveys = surveys.append(pd.read_csv(file,parse_dates=['date']))

surveys = surveys.sort_values(by='date')
print("Latest Microdistancing survey is {}".format(surveys.date.values[-1]))

surveys.loc[surveys.state!='ACT','state'] = surveys.loc[surveys.state!='ACT','state'].map(states_initials).fillna(surveys.loc[surveys.state!='ACT','state'])
surveys['proportion'] = surveys['count']/surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

always =surveys.loc[surveys.response=='Always'].set_index(["state",'date'])
always = always.unstack(['state'])
# If you get an error here saying 'cannot create a new series when the index is not unique', then you have a duplicated md file.

idx = pd.date_range('2020-03-01',pd.to_datetime("today"))
always = always.reindex(idx, fill_value=np.nan)
always.index.name = 'date'

# fill back to earlier and between weeks.
# Assume survey on day x applies for all days up to x - 6
always =always.fillna(method='bfill')
#assume values continue forward if survey hasn't completed
always = always.fillna(method='ffill')
always = always.stack(['state'])

#Zero out before first survey 20th March
always = always.reset_index().set_index('date')
always.loc[:'2020-03-20','count'] =0
always.loc[:'2020-03-20','respondents'] =0
always.loc[:'2020-03-20','proportion'] =0

always = always.reset_index().set_index(['state','date'])

survey_X = pd.pivot_table(data=always,
                          index='date',columns='state',values='proportion')
survey_counts_base =pd.pivot_table(data=always,
                          index='date',columns='state',values='count').drop(['Australia','Other'],axis=1).astype(int)

survey_respond_base = pd.pivot_table(data=always,
                          index='date',columns='state',values='respondents').drop(['Australia','Other'],axis=1).astype(int)


######### Read in EpyReff results #########
df_Reff = pd.read_csv("results/EpyReff/Reff"+
            data_date.strftime("%Y-%m-%d")+"tau_4.csv",parse_dates=['INFECTION_DATES'])
df_Reff['date'] = df_Reff.INFECTION_DATES
df_Reff['state'] = df_Reff.STATE


######### Read in NNDSS/linelist data #########
df_state = read_in_cases(case_file_date=data_date.strftime('%d%b%Y')) # If this errors it may be missing a leading zero on the date.

df_Reff = df_Reff.merge(df_state,how='left',left_on=['state','date'], right_on=['STATE','date_inferred']) #how = left to use Reff days, NNDSS missing dates
df_Reff['rho_moving'] = df_Reff.groupby(['state'])['rho'].transform(lambda x: x.rolling(7,1).mean()) #minimum number of 1

#some days have no cases, so need to fillna
df_Reff['rho_moving'] = df_Reff.rho_moving.fillna(method='bfill')


#shift counts to align with infection date not symptom date
# dates should be complete at this point, no days skipped
# will be some end days with NaN, but that should be fine since
# we don't use the most recent 10 days
df_Reff['local'] = df_Reff.local.shift(periods=-5)
df_Reff['imported'] = df_Reff.imported.shift(periods=-5)
df_Reff['rho_moving'] = df_Reff.rho_moving.shift(periods=-5)
df_Reff['rho'] = df_Reff.rho.shift(periods=-5)
df_Reff['local'] = df_Reff.local.fillna(0)
df_Reff['imported'] = df_Reff.imported.fillna(0)


######### Read in Google mobility results #########
import sys; sys.path.insert(0, '../'); from params import download_google_automatically  
df_google = read_in_google(local=not download_google_automatically,moving=True)

df= df_google.merge(df_Reff[['date','state','mean','lower','upper',
                            'top','bottom','std','rho','rho_moving','local','imported']], on=['date','state'],how='inner')

##### Create useable dataset
## ACT and NT not in original estimates, need to extrapolated
states_to_fit = sorted(['NSW','VIC','QLD','SA','WA','TAS']) #sorting keeps consistent with sort in data_by_state
fit_post_March = True
ban = '2020-03-20'
start_date = '2020-03-01'
end_date = '2020-03-31'

##Second wave inputs
sec_states=sorted(['NSW'])
sec_start_date = '2020-06-01'
sec_end_date = '2021-01-19'

# TEMPLATE: Third wave inputs
# third_states=sorted(['VIC','NSW','WA','TAS'])
third_states=sorted(['NSW'])
third_start_date = '2021-06-27'
third_end_date = data_date - pd.Timedelta(days=10) # Subtract 10 days to avoid right truncation

fit_mask = df.state.isin(states_to_fit)
if fit_post_March:
    fit_mask = (fit_mask) & (df.date >= start_date)

fit_mask = (fit_mask) & (df.date <= end_date )

second_wave_mask = df.state.isin(sec_states)
second_wave_mask = (second_wave_mask) & (df.date >= sec_start_date)
second_wave_mask = (second_wave_mask) & (df.date <= sec_end_date)

# Add third wave stuff here

third_wave_mask = df.state.isin(third_states)
third_wave_mask = (third_wave_mask) & (df.date >= third_start_date)
third_wave_mask = (third_wave_mask) & (df.date <= third_end_date)

predictors = mov_values.copy()
#predictors.extend(['driving_7days','transit_7days','walking_7days','pc'])

#remove residential to see if it improves fit
predictors.remove('residential_7days')

df['post_policy'] = (df.date >= ban).astype(int)

dfX = df.loc[fit_mask].sort_values('date')
df2X = df.loc[second_wave_mask].sort_values('date')
df3X = df.loc[third_wave_mask].sort_values('date')

#filter out the surveys we don't have
# if df3X.shape[0]>0:
#     survey_respond = survey_respond_base.loc[:df3X.date.values[-1]]
#     survey_counts = survey_counts_base.loc[:df3X.date.values[-1]]
# elif df2X.shape[0]>0:
#     survey_respond = survey_respond_base.loc[:df2X.date.values[-1]]
#     survey_counts = survey_counts_base.loc[:df2X.date.values[-1]]
# else:
#     survey_respond = survey_respond_base.loc[:dfX.date.values[-1]]
#     survey_counts = survey_counts_base.loc[:dfX.date.values[-1]]    

# #original code for filtering out the surveys we dont have
# if df2X.shape[0]>0:
#     survey_respond = survey_respond_base.loc[:df2X.date.values[-1]]
#     survey_counts = survey_counts_base.loc[:df2X.date.values[-1]]
# else:
#     survey_respond = survey_respond_base.loc[:dfX.date.values[-1]]
#     survey_counts = survey_counts_base.loc[:dfX.date.values[-1]]


#choose dates for each state for sec wave
sec_date_range = {
    'NSW':pd.date_range(start=sec_start_date,end=sec_end_date).values,
    # 'VIC':pd.date_range(start=sec_start_date,end='2020-10-28').values
}

#choose dates for each state for third wave
third_date_range = {
    'NSW':pd.date_range(start=third_start_date,end=third_end_date).values,
    # 'VIC':pd.date_range(start=third_start_date,end='2020-10-28').values
}

df2X['is_sec_wave'] =0
for state in sec_states:
    df2X.loc[df2X.state==state,'is_sec_wave'] = df2X.loc[df2X.state==state].date.isin(
        sec_date_range[state]
        ).astype(int).values

df3X['is_third_wave'] =0
for state in third_states:
    df3X.loc[df3X.state==state,'is_third_wave'] = df3X.loc[df3X.state==state].date.isin(
        third_date_range[state]
        ).astype(int).values
        
data_by_state= {}
sec_data_by_state={}
third_data_by_state={}

for value in ['mean','std','local','imported']:
    data_by_state[value] = pd.pivot(dfX[['state',value,'date']],
                        index='date',columns='state',values=value).sort_index(
        axis='columns')
    sec_data_by_state[value] = pd.pivot(df2X[['state',value,'date']],
                        index='date',columns='state',values=value).sort_index(
        axis='columns')
    third_data_by_state[value] = pd.pivot(df3X[['state',value,'date']],
                        index='date',columns='state',values=value).sort_index(
        axis='columns')
    #account for dates pre pre second wave
    if df2X.loc[df2X.state==sec_states[0]].shape[0]==0:
        print("making empty")
        sec_data_by_state[value] = pd.DataFrame(columns=sec_states).astype(float)
    #account for dates pre pre third wave
    if df3X.loc[df3X.state==third_states[0]].shape[0]==0:
        print("making empty")
        third_data_by_state[value] = pd.DataFrame(columns=third_states).astype(float)

#FIRST PHASE
mobility_by_state =[]
mobility_std_by_state=[]
count_by_state =[]
respond_by_state=[]

# filtering survey responses to dates before this wave fitting
survey_respond = survey_respond_base.loc[:dfX.date.values[-1]]
survey_counts = survey_counts_base.loc[:dfX.date.values[-1]]  

for state in states_to_fit:

    mobility_by_state.append(dfX.loc[dfX.state==state, predictors].values/100)
    mobility_std_by_state.append(
        dfX.loc[dfX.state==state,[val+'_std' for val in predictors]].values/100
    )
    count_by_state.append(survey_counts.loc[start_date:end_date,state].values)
    respond_by_state.append(survey_respond.loc[start_date:end_date,state].values)

#SECOND PHASE
sec_mobility_by_state =[]
sec_mobility_std_by_state=[]
sec_count_by_state=[]
sec_respond_by_state=[]
include_in_sec_wave=[]

# filtering survey responses to dates before this wave fitting
survey_respond = survey_respond_base.loc[:df2X.date.values[-1]]
survey_counts = survey_counts_base.loc[:df2X.date.values[-1]]

for state in sec_states:

    sec_mobility_by_state.append(df2X.loc[df2X.state==state, predictors].values/100)
    sec_mobility_std_by_state.append(
        df2X.loc[df2X.state==state,[val+'_std' for val in predictors]].values/100
    )
    sec_count_by_state.append(survey_counts.loc[sec_start_date:sec_end_date,state].values)
    sec_respond_by_state.append(survey_respond.loc[sec_start_date:sec_end_date,state].values)
    include_in_sec_wave.append(df2X.loc[df2X.state==state,'is_sec_wave'].values)

#THIRD WAVE
third_mobility_by_state =[]
third_mobility_std_by_state=[]
third_count_by_state=[]
third_respond_by_state=[]
include_in_third_wave=[]

# filtering survey responses to dates before this wave fitting
survey_respond = survey_respond_base.loc[:df3X.date.values[-1]]
survey_counts = survey_counts_base.loc[:df3X.date.values[-1]]

for state in third_states:

    third_mobility_by_state.append(df3X.loc[df3X.state==state, predictors].values/100)
    third_mobility_std_by_state.append(
        df3X.loc[df3X.state==state,[val+'_std' for val in predictors]].values/100
    )
    third_count_by_state.append(survey_counts.loc[third_start_date:third_end_date,state].values)
    third_respond_by_state.append(survey_respond.loc[third_start_date:third_end_date,state].values)
    include_in_third_wave.append(df3X.loc[df3X.state==state,'is_third_wave'].values)
    
#policy boolean flag for after travel ban in each wave 
policy = dfX.loc[dfX.state==states_to_fit[0],'post_policy']     # this is the post ban policy 
policy_sec_wave = [1]*df2X.loc[df2X.state==sec_states[0]].shape[0]
policy_third_wave = [1]*df3X.loc[df3X.state==third_states[0]].shape[0]

state_index = { state : i+1  for i, state in enumerate(states_to_fit)}
##Make state by state arrays
input_data ={
    'N': dfX.loc[dfX.state==states_to_fit[0]].shape[0],
    'K': len(predictors),
    'j':len(states_to_fit),
    'Reff': data_by_state['mean'].values,
    'Mob': mobility_by_state,
    'Mob_std':mobility_std_by_state,
    'sigma2': data_by_state['std'].values**2,
    'policy': policy.values,
    'local':data_by_state['local'].values,
    'imported':data_by_state['imported'].values,

    'N_sec_wave': df2X.loc[df2X.state==sec_states[0]].shape[0],
    'j_sec_wave': len(sec_states),
    'Reff_sec_wave': sec_data_by_state['mean'].values,
    'Mob_sec_wave': sec_mobility_by_state,
    'Mob_sec_wave_std':sec_mobility_std_by_state,
    'sigma2_sec_wave': sec_data_by_state['std'].values**2,
    'policy_sec_wave': policy_sec_wave,
    'local_sec_wave':sec_data_by_state['local'].values,
    'imported_sec_wave':sec_data_by_state['imported'].values,

    'N_third_wave': df3X.loc[df3X.state==third_states[0]].shape[0],
    'j_third_wave': len(third_states),
    'Reff_third_wave': third_data_by_state['mean'].values,
    'Mob_third_wave': third_mobility_by_state,
    'Mob_third_wave_std':third_mobility_std_by_state,
    'sigma2_third_wave': third_data_by_state['std'].values**2,
    'policy_third_wave': policy_third_wave,
    'local_third_wave':third_data_by_state['local'].values,
    'imported_third_wave':third_data_by_state['imported'].values,

    'count_md':count_by_state,
    'respond_md':respond_by_state,
    'count_md_sec_wave':sec_count_by_state,
    'respond_md_sec_wave':sec_respond_by_state,
    'count_md_third_wave':third_count_by_state,
    'respond_md_third_wave':third_respond_by_state,

    'map_to_state_index_sec': [state_index[state] for state in sec_states],
    'map_to_state_index_third': [state_index[state] for state in third_states],
    'total_N_p_sec':sum( [sum(x) for x in include_in_sec_wave]),
    'total_N_p_third':sum( [sum(x) for x in include_in_third_wave]),
    'include_in_sec_wave': include_in_sec_wave,
    'include_in_third_wave': include_in_third_wave,
    'pos_starts_sec': np.cumsum([sum(x) for x in include_in_sec_wave]),
    'pos_starts_third': np.cumsum([sum(x) for x in include_in_third_wave])
}

fit = sm_pol_gamma.sampling(
    data=input_data,
    iter=5000,
    chains=2,
    #control={'max_treedepth':15}
)


######## Plotting & Saving Output #########

#make results dir
results_dir ="figs/soc_mob_posterior/"
os.makedirs(results_dir,exist_ok=True)

filename = "stan_posterior_fit" + data_date.strftime("%Y-%m-%d") + ".txt"
with open(results_dir+filename, 'w') as f:
    print(fit.stansummary(pars=['bet','R_I','R_L','R_Li','theta_md','sig']), file=f)
samples_mov_gamma = fit.to_dataframe(pars=['bet','R_I','R_L','R_Li','sig','brho','theta_md','brho_sec_wave','brho_third_wave'])

# Plot ratio of imported to total cases
# First phase
#rho calculated at data entry
if isinstance(df_state.index, pd.MultiIndex):
    df_state = df_state.reset_index()


states=sorted(['NSW','QLD','VIC','TAS','SA','WA','ACT','NT'])
fig,ax = plt.subplots(figsize=(24,9), ncols=len(states),sharey=True)
states_to_fitd = {state: i+1 for i,state in enumerate(states_to_fit)      }

for i, state in enumerate(states):
    if state in states_to_fit:
        dates = df_Reff.loc[(df_Reff.date>=start_date) &
                            (df_Reff.state==state)&(df_Reff.date<=end_date)].date
        rho_samples = samples_mov_gamma[['brho['+str(j+1)+','+str(states_to_fitd[state])+']' for j in range(dfX.loc[dfX.state==states_to_fit[0]].shape[0])]]
        ax[i].plot(dates, rho_samples.median(),label='fit',color='C0')
        ax[i].fill_between(dates, rho_samples.quantile(0.25),rho_samples.quantile(0.75),color='C0',alpha=0.4)

        ax[i].fill_between(dates, rho_samples.quantile(0.05),rho_samples.quantile(0.95),color='C0',alpha=0.4)
    else:
        sns.lineplot(x='date_inferred',y='rho',
            data=df_state.loc[(df_state.date_inferred>=start_date) & (df_state.STATE==state)&(df_state.date_inferred<=end_date)], ax=ax[i],color='C1',label='data')
    sns.lineplot(x='date',y='rho',
            data=df_Reff.loc[(df_Reff.date>=start_date) & (df_Reff.state==state)&(df_Reff.date<=end_date)], ax=ax[i],color='C1',label='data')
    sns.lineplot(x='date',y='rho_moving',
            data=df_Reff.loc[(df_Reff.date>=start_date) & (df_Reff.state==state)&(df_Reff.date<=end_date)], ax=ax[i],color='C2',label='moving')

    dates = dfX.loc[dfX.state==states_to_fit[0]].date

    ax[i].tick_params('x',rotation=90)
    ax[i].xaxis.set_major_locator(plt.MaxNLocator(4))
    ax[i].set_title(state)
ax[0].set_ylabel('Proportion of imported cases')
plt.legend()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"rho_first_phase.png",dpi = 144)

# Second phase
if df2X.shape[0]>0:
    fig,ax = plt.subplots(figsize=(24,9), ncols=len(sec_states),sharey=True, squeeze=False)
    states_to_fitd = {state: i+1 for i,state in enumerate(sec_states)      }
    pos = 1
    for i, state in enumerate(sec_states):
        #Google mobility only up to a certain date, so take only up to that value
        dates = df2X.loc[(df2X.state==state)&(
            df2X.is_sec_wave==1
        )].date.values
        #df_Reff.loc[(df_Reff.date>=sec_start_date) &
        #                    (df_Reff.state==state)&(df_Reff.date<=sec_end_date)].date
        rho_samples = samples_mov_gamma[
            ['brho_sec_wave['+str(j)+']'
            for j in range(pos, pos+df2X.loc[df2X.state==state].is_sec_wave.sum() ) ]
            ]
        pos = pos + df2X.loc[df2X.state==state].is_sec_wave.sum()

        ax[0,i].plot(dates, rho_samples.median(),label='fit',color='C0')
        ax[0,i].fill_between(dates, rho_samples.quantile(0.25),rho_samples.quantile(0.75),color='C0',alpha=0.4)

        ax[0,i].fill_between(dates, rho_samples.quantile(0.05),rho_samples.quantile(0.95),color='C0',alpha=0.4)

        sns.lineplot(x='date_inferred',y='rho',
            data=df_state.loc[(df_state.date_inferred>=sec_start_date) & (df_state.STATE==state)&(df_state.date_inferred<=sec_end_date)], ax=ax[0,i],color='C1',label='data')
        sns.lineplot(x='date',y='rho',
                data=df_Reff.loc[(df_Reff.date>=sec_start_date) & (df_Reff.state==state)&(df_Reff.date<=sec_end_date)], ax=ax[0,i],color='C1',label='data')
        sns.lineplot(x='date',y='rho_moving',
                data=df_Reff.loc[(df_Reff.date>=sec_start_date) & (df_Reff.state==state)&(df_Reff.date<=sec_end_date)], ax=ax[0,i],color='C2',label='moving')

        dates = dfX.loc[dfX.state==sec_states[0]].date

        ax[0,i].tick_params('x',rotation=90)
        ax[0,i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[0,i].set_title(state)
    ax[0,0].set_ylabel('Proportion of imported cases')
    plt.legend()
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"rho_sec_phase.png",dpi = 144)

# Third  phase
if df3X.shape[0]>0:
    fig,ax = plt.subplots(figsize=(24,9), ncols=len(third_states),sharey=True, squeeze=False)
    states_to_fitd = {state: i+1 for i,state in enumerate(third_states)      }
    pos = 1
    for i, state in enumerate(third_states):
        #Google mobility only up to a certain date, so take only up to that value
        dates = df3X.loc[(df3X.state==state)&(
            df3X.is_third_wave==1
        )].date.values
        #df_Reff.loc[(df_Reff.date>=sec_start_date) &
        #                    (df_Reff.state==state)&(df_Reff.date<=sec_end_date)].date
        rho_samples = samples_mov_gamma[
            ['brho_third_wave['+str(j)+']'
            for j in range(pos, pos+df3X.loc[df3X.state==state].is_third_wave.sum() ) ]
            ]
        pos = pos + df3X.loc[df3X.state==state].is_third_wave.sum()

        ax[0,i].plot(dates, rho_samples.median(),label='fit',color='C0')
        ax[0,i].fill_between(dates, rho_samples.quantile(0.25),rho_samples.quantile(0.75),color='C0',alpha=0.4)

        ax[0,i].fill_between(dates, rho_samples.quantile(0.05),rho_samples.quantile(0.95),color='C0',alpha=0.4)

        sns.lineplot(x='date_inferred',y='rho',
            data=df_state.loc[(df_state.date_inferred>=third_start_date) & (df_state.STATE==state)&(df_state.date_inferred<=third_end_date)], ax=ax[0,i],color='C1',label='data')
        sns.lineplot(x='date',y='rho',
                data=df_Reff.loc[(df_Reff.date>=third_start_date) & (df_Reff.state==state)&(df_Reff.date<=third_end_date)], ax=ax[0,i],color='C1',label='data')
        sns.lineplot(x='date',y='rho_moving',
                data=df_Reff.loc[(df_Reff.date>=third_start_date) & (df_Reff.state==state)&(df_Reff.date<=third_end_date)], ax=ax[0,i],color='C2',label='moving')

        dates = dfX.loc[dfX.state==third_states[0]].date

        ax[0,i].tick_params('x',rotation=90)
        ax[0,i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[0,i].set_title(state)
    ax[0,0].set_ylabel('Proportion of imported cases')
    plt.legend()
    plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"rho_third_phase.png",dpi = 144)

#plot marginal distributions
fig,ax = plt.subplots(figsize=(12,9))
samples_mov_gamma['R_L_prior'] = np.random.gamma(
1.8*1.8/0.05,0.05/1.8, size=samples_mov_gamma.shape[0])

samples_mov_gamma['R_I_prior'] = np.random.gamma(
0.5**2/0.2, .2/0.5, size=samples_mov_gamma.shape[0])

samples_mov_gamma['R_L_national'] = np.random.gamma(
    samples_mov_gamma.R_L.values **2/ samples_mov_gamma.sig.values,
    samples_mov_gamma.sig.values / samples_mov_gamma.R_L.values
)
df_R_values = pd.melt(samples_mov_gamma[[col for col in samples_mov_gamma if 'R' in col]])
print(df_R_values.variable.unique())
sns.violinplot(x='variable',y='value',
            data=pd.melt(samples_mov_gamma[[col for col in samples_mov_gamma if 'R' in col]]),
            ax=ax,
            cut=0)

ax.set_yticks([1],minor=True,)
ax.set_yticks([0,2,3],minor=False)
ax.set_yticklabels([0,2,3],minor=False)
ax.set_ylim((0,3))
#state labels in alphabetical
ax.set_xticklabels(['R_I','R_L0 mean',
'R_L0 NSW','R_L0 QLD','R_L0 SA','R_L0 TAS','R_L0 VIC','R_L0 WA',#'R_temp',
'R_L0 prior','R_I prior','R_L0 national'])
ax.set_xlabel('')
ax.set_ylabel('Effective reproduction number')
ax.tick_params('x',rotation=90)
ax.yaxis.grid(which='minor',linestyle='--',color='black',linewidth=2)
plt.tight_layout()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"R_priors.png",dpi = 144)

# Making a new figure that doesn't include the priors
fig,ax = plt.subplots(figsize=(12,9))

small_plot_cols =['R_Li[1]', 'R_Li[2]', 'R_Li[3]', 'R_Li[4]', 'R_Li[5]', 'R_Li[6]', 'R_I']

sns.violinplot(x='variable',y='value',
            data=pd.melt(samples_mov_gamma[small_plot_cols]),
            ax=ax,
            cut=0)

ax.set_yticks([1],minor=True,)
ax.set_yticks([0,2,3],minor=False)
ax.set_yticklabels([0,2,3],minor=False)
ax.set_ylim((0,3))
#state labels in alphabetical
ax.set_xticklabels(['$R_L0$ NSW','$R_L0$ QLD','$R_L0$ SA','$R_L0$ TAS','$R_L0$ VIC','$R_L0$ WA', '$R_I$',])
ax.tick_params('x',rotation=90)
ax.set_xlabel('')
ax.set_ylabel('Effective reproduction number')
ax.yaxis.grid(which='minor',linestyle='--',color='black',linewidth=2)
plt.tight_layout()
plt.savefig(results_dir+data_date.strftime("%Y-%m-%d")+"R_priors_(without_priors).png",dpi = 288)


posterior = samples_mov_gamma[['bet['+str(i)+']' for i in range(1,1+len(predictors))]
                            ]

split=True
md = 'power'#samples_mov_gamma.md.values

posterior.columns = [val for val in predictors]
long = pd.melt(posterior)

fig,ax2 =plt.subplots(figsize=(12,9))

ax2 = sns.violinplot(x='variable',y='value',#hue='policy',
                    data=long,
                    ax=ax2,
                    color='C0'
                )


ax2.plot([0]*len(predictors), linestyle='dashed',alpha=0.6, color = 'grey')
ax2.tick_params(axis='x',rotation=90)

#ax =plot_posterior_violin(posterior)

ax2.set_title('Coefficients of mobility indices')
ax2.set_xlabel('Social mobility index')
ax2.set_xticklabels([var[:-6] for var in predictors])
ax2.set_xticklabels(['Retail and Recreation','Grocery and Pharmacy', 'Parks', 'Transit Stations', 'Workplaces'])
ax2.tick_params('x',rotation=15)
plt.tight_layout()

plt.savefig(
    results_dir+data_date.strftime("%Y-%m-%d")+'mobility_posteriors.png', dpi =288)


RL_by_state = { state: samples_mov_gamma[
    'R_Li['+str(i)+']'].values for state, i in state_index.items()
}
ax3 =predict_plot(samples_mov_gamma,df.loc[(df.date>=start_date)&(df.date<=end_date)],gamma=True,
                moving=True,split=split,grocery=True,ban = ban,
                R=RL_by_state, var= True, md_arg=md,
                rho=states_to_fit, R_I =samples_mov_gamma.R_I.values,prop=survey_X.loc[start_date:end_date])#by states....
for ax in ax3:
    for a in ax:
        a.set_ylim((0,3))
        #a.set_xlim((start_date,end_date))
plt.savefig(
    results_dir+data_date.strftime("%Y-%m-%d")+"total_Reff_allstates.png", dpi=144)

if df2X.shape[0]>0:
    df['is_sec_wave'] =0
    for state in sec_states:
        df.loc[df.state==state,'is_sec_wave'] = df.loc[df.state==state].date.isin(
            sec_date_range[state]
            ).astype(int).values
    #plot only if there is second phase data - have to have second_phase=True
    ax4 =predict_plot(samples_mov_gamma,df.loc[(df.date>=sec_start_date)&(df.date<=sec_end_date)],gamma=True, moving=True,split=split,grocery=True,ban = ban,
                    R=RL_by_state, var= True, md_arg=md,
                    rho=sec_states, second_phase=True,
                    R_I =samples_mov_gamma.R_I.values,prop=survey_X.loc[sec_start_date:sec_end_date])#by states....
    for ax in ax4:
        for a in ax:
            a.set_ylim((0,3))
            #a.set_xlim((start_date,end_date))
    plt.savefig(
        results_dir+data_date.strftime("%Y-%m-%d")+"Reff_sec_phase.png", dpi=144)

    #remove plots from memory
    fig.clear()
    plt.close(fig)


if df3X.shape[0]>0:
    df['is_third_wave'] =0
    for state in third_states:
        df.loc[df.state==state,'is_third_wave'] = df.loc[df.state==state].date.isin(
            third_date_range[state]
            ).astype(int).values
    #plot only if there is third phase data - have to have third_phase=True
    ax4 =predict_plot(samples_mov_gamma,df.loc[(df.date>=third_start_date)&(df.date<=third_end_date)],gamma=True, moving=True,split=split,grocery=True,ban = ban,
                    R=RL_by_state, var= True, md_arg=md,
                    rho=third_states, third_phase=True,
                    R_I =samples_mov_gamma.R_I.values,prop=survey_X.loc[third_start_date:third_end_date])#by states....
    for ax in ax4:
        for a in ax:
            a.set_ylim((0,3))
            #a.set_xlim((start_date,end_date))
    plt.savefig(
        results_dir+data_date.strftime("%Y-%m-%d")+"Reff_third_phase.png", dpi=144)

    #remove plots from memory
    fig.clear()
    plt.close(fig)

var_to_csv = predictors
samples_mov_gamma[predictors] = samples_mov_gamma[['bet['+str(i)+']' for i in range(1,1+len(predictors))]]
var_to_csv = ['R_I']+['R_L','sig']+['theta_md']+predictors + [
    'R_Li['+str(i+1)+']' for i in range(len(states_to_fit))
    ]


samples_mov_gamma[var_to_csv].to_hdf('results/soc_mob_posterior'+data_date.strftime("%Y-%m-%d")+'.h5',key='samples')
