
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Reff_constants import *
plt.style.use('seaborn-poster')

def read_in_posterior(date='2020-07-30'):
    """
    read in samples from posterior from inference
    """
    df = pd.read_hdf("data/soc_mob_posterior"+date+".h5", key='samples')
    
    return df

def read_in_google(Aus_only=True,local=False,moving=False):
    """
    Read in the Google data set
    """
    if local:
        if type(local)==str:
            df = pd.read_csv(local,parse_dates=['date'])
        elif type(local)==bool:
            local = 'data/Global_Mobility_Report.csv'
            df = pd.read_csv(local,parse_dates=['date'])
    else:
        df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',parse_dates=['date'])
    
    
    if Aus_only:
        df = df.loc[df.country_region_code=='AU']
        #Change state column to state initials
        df['state'] = df.sub_region_1.map(lambda x: states_initials[x] if not pd.isna(x) else 'AUS' )
    df = df.loc[df.sub_region_2.isna()]
    if moving:
        # generate moving average columns in reverse
        df = df.sort_values(by='date')
        mov_values = []
        for val in value_vars:
            mov_values.append(val[:-29]+'_7days')
            df[mov_values[-1]]=df.groupby(['state'])[val].transform(
                lambda x: x[::-1].rolling(7,1).mean()[::-1]) #minimumnumber of 1
            
            #minimum of 7 days for std, forward fill the rest
            df[mov_values[-1]+'_std'] = df.groupby(['state'])[val].transform(
                lambda x: x[::-1].rolling(7,7).std()[::-1])
            #fill final values as std doesn't work with single value
            df[mov_values[-1]+'_std'] = df.groupby('state')[mov_values[-1]+'_std'].fillna(method='ffill')
    #show latest date
    print("Latest date in Google indices " + str(df.date.values[-1]))
    return df
    
    
def read_in_apple(Aus_only=True,local=False,moving=False):
    """
    Read in Apple mobility dataset
    """
    if local:
        if type(local)==str:
            df = pd.read_csv(local)
        elif type(local)==bool:
            local = '../data/applemobilitytrends-2020-05-02.csv'
            df = pd.read_csv(local)
    else:
        df = pd.read_csv('https://covid19-static.cdn-apple.com/covid19-mobility-data/2007HotfixDev46/v2/en-us/applemobilitytrends-2020-05-02.csv')
    
    if Aus_only:
    #filter out non-Australian locations. NOTE only Mel, Syd, BNE and PER in apple dataset
        cities = ['Melbourne','Sydney','Brisbane','Adelaide','Canberra','Perth','Hobart','Australia']
        df = df.loc[df.region.isin(cities)]
    dates = [col for col in df.columns if '2020' in col]
    df = pd.melt(df,value_vars=dates,id_vars=['region','transportation_type'],var_name='date')
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df = df.pivot_table(index=['region','date'],columns=['transportation_type'],values=['value'])
    
    apple_cols = ['driving','transit','walking']
    df.columns = apple_cols
    
    df = df.reset_index()
    city_state = {
    'Sydney':'NSW',
    'Melbourne':'VIC',
    'Perth':'WA',
    'Brisbane':'QLD',
    'Australia':'AUS',#hack
    }
    df['state'] = df.region.apply(lambda x: city_state[x])
    df[apple_cols] = df[apple_cols] -100
    
    if moving:
    # generate moving average columns in reverse
        df = df.sort_values(by='date')
        for val in apple_cols:
            df[val+'_7days']=df.groupby(['state'])[val].transform(lambda x: x[::-1].rolling(7,1).mean()[::-1]) #minimum number of 1
    
    return df

def read_in_FB():
    """
    Read in Lewis' FB data
    """
    
    df = pd.read_csv('../data/av_contacts_state_pc.csv', parse_dates=['date'], index_col=[0])
    df['state'] = df.state.apply(lambda x: states_initials[x])
    
    return df

def read_in_Reff(path='../data/Dennis_2020_04_23/'):
    """
    Read in Reff csv from Price et al 2020. Originals are in RDS, are converted to csv in R script
    """
    import pandas as pd
    Reff = pd.read_csv(path+'R_eff_2020_04_23.csv', parse_dates=['date'])
    return Reff
    
def read_AddInsight():
    """
    Read in the geogson files from local directory and collate into 
    """
    import os
    from datetime import datetime as dt
    import geopandas as gpd
    years = (2018,2019,2020)
    df_json = pd.DataFrame()
    for year in years:
        start ="{}-01-01".format(year)
        end = "{}-04-26".format(year)
        dates = [ dt.strftime(date, '%Y_%m_%dT00_00') 
                 for date in 
                 pd.date_range(start=start, end=end).to_pydatetime()]



        jsons_list = []

        for date in dates:
        #    try:
            df_json1 = gpd.read_file(
                    os.getcwd()+'/../jcoxwrapper/geojson/od_trips_by_destination_allday{}_geo.json'.format(date))
            df_json1['date'] = date
            df_json = df_json.append(df_json1, ignore_index=True)
            jsons_list.append(df_json1)
    #            except:
            #    print(date,"data for this day is not downloaded")
               # continue
    df_json['date'] = df_json.date.apply(lambda x: pd.to_datetime(x.replace('_','').replace('T','+'), format='%Y%m%d%z'))
    df_json['day'] = df_json.date.dt.dayofweek # Monday =0, Sunday = 6
    df_json['week'] = df_json.date.dt.weekofyear
    df_json['year'] = df_json.date.dt.year
    
    return df_json
def predict_plot(samples, df, split=True,gamma=False,moving=True,grocery=True, 
                 delta=1.0,R=2.2,sigma=1, md_arg=None,
                 ban='2020-03-16',single=False,var=None,
                rho=None, R_I =None, winter=False, prop=None,second_phase=False):
    """
    Produce posterior predictive plots for all states
    """
    from scipy.special import expit

    value_vars=['retail_and_recreation_percent_change_from_baseline',
                            'grocery_and_pharmacy_percent_change_from_baseline',
                            'parks_percent_change_from_baseline',
                            'transit_stations_percent_change_from_baseline',
                            'workplaces_percent_change_from_baseline',
                            'residential_percent_change_from_baseline']
    value_vars.remove('residential_percent_change_from_baseline')
    if not grocery:
        value_vars.remove('grocery_and_pharmacy_percent_change_from_baseline')
    if moving:
        value_vars = [ val[:-29]+'_7days' for val in value_vars]
    
    if single:
        #Single state
        fig, ax = plt.subplots(figsize=(12,9))
        df_state = df
        post_values = samples[['beta['+str(i)+']' for i in range(1,1+len(value_vars))]].sample(df_state.shape[0]).values.T      
        if split:
            #split model with parameters pre and post policy
        
            df1 =df_state.loc[df_state.date<=ban]
            df2 = df_state.loc[df_state.date>ban]
            X1 = df1[value_vars]/100 #N by K
            X2 = df2[value_vars]/100
            logodds = X1 @ post_values # N by K times (Nsamples by K )^T = N by N
            
            if md is None:
                post_alphas = samples[['alpha['+str(i)+']' for i in range(1,1+len(value_vars))]].sample(df_state.shape[0]).values.T
                logodds = np.append(logodds, X2 @ (post_values + post_alphas),axis=0)
            else:
                #take right size of md
                md = np.random.choice(md, size=df_state.shape[0])
                
                #set initial pre ban values of md to 1
                md[:logodds.shape[0]] = np.ones(size=logodds.shape[0])
                
                #make logodds by appending post ban values
                logodds = np.append(logodds, X2@ post_values, axis=0)
            
        else:
            X1 = df_state[value_vars]/100
            logodds = X1 @ post_values # N by K times (Nsamples by K )^T = N by N
        if gamma:
            if type(md)==np.ndarray:
                mu_hat = 2* expit(logodds) *policy*md
            else:
                mu_hat = 2 * expit(logodds)
            
            if type(delta)==np.ndarray:
                delta = np.random.choice(delta,size=df_state.shape[0])
            R = np.random.choice(R,size =df_state.shape[0] )
            R_eff_hat = np.random.gamma(shape=R * mu_hat*delta, scale=1.0/delta)
        else:
            #Use normal distribution
            mu_hat = R * 2 * expit(logodds)
            if type(sigma)== pd.Series:    
                sigma_i = sigma.sample(df_state.shape[0]).values
            else:
                sigma_i = sigma
            R_eff_hat = np.random.normal(mu_hat, sigma_i) # N by N, where rows = datum, column = sample from posterior

        df_hat = pd.DataFrame(R_eff_hat.T)

        #plot actual R_eff
        ax.plot(df_state.date, df_state['mean'], label='R_eff from Price et al')
        ax.fill_between(df_state.date, df_state['bottom'], df_state['top'],color='C0', alpha=0.3)

        ax.plot(df_state.date,df_hat.quantile(0.5,axis=0), label='R_eff_hat',color='C1')
        ax.fill_between(df_state.date, df_hat.quantile(0.25,axis=0), df_hat.quantile(0.75,axis=0),color='C1',alpha=0.3)
        ax.fill_between(df_state.date, df_hat.quantile(0.05,axis=0), df_hat.quantile(0.95,axis=0),color='C1',alpha=0.3)
        
        #grid line at R_eff =1
        ax.set_yticks([1],minor=True,)
        ax.yaxis.grid(b=True,which='minor',linestyle='dashed',color='grey')
        ax.tick_params(axis='x',rotation=90)
        
    else:
        #all states
        fig, ax = plt.subplots(figsize=(15,12), ncols=3,nrows=2, sharex=True, sharey=True)

        states = sorted(list(states_initials.keys()))
        states.remove('Northern Territory')
        states.remove('Australian Capital Territory')
         #no R_eff modelled for these states, skip
        #counter for brho_v
        pos=1
        for i,state in enumerate(states):

            df_state = df.loc[df.sub_region_1==state]
            if second_phase:
                df_state = df_state.loc[df_state.is_sec_wave==1]
            samples_sim = samples.sample(1000)
            post_values = samples_sim[['bet['+str(i)+']' for i in range(1,1+len(value_vars))]].values.T    
            prop_sim = prop[states_initials[state]].values[:df_state.shape[0]]
            if split:

                #split model with parameters pre and post policy
        
                df1 =df_state.loc[df_state.date<=ban]
                df2 = df_state.loc[df_state.date>ban]
                X1 = df1[value_vars]/100 #N by K
                X2 = df2[value_vars]/100
                logodds = X1 @ post_values # N by K times (Nsamples by K )^T = N by N

                if md_arg is None:
                    post_alphas = samples_sim[['alpha['+str(i)+']' for i in range(1,1+len(value_vars))]].values.T
                    logodds = np.append(logodds, X2 @ (post_values + post_alphas),axis=0)
                    md=1
                elif md_arg=='power':
                    theta_md = samples_sim.theta_md.values #1 by samples shape
                    theta_md = np.tile(theta_md, (df_state.shape[0],1)) #each row is a date, column a new sample
                    md = ((1+theta_md).T**(-1* prop_sim)).T
                    md[:logodds.shape[0]] = 1
                    #make logodds by appending post ban values
                    logodds = np.append(logodds, X2@ post_values, axis=0)
                elif md_arg=='logistic':
                    theta_md = samples_sim.theta_md.values #1 by samples shape
                    theta_md = np.tile(theta_md, (df_state.shape[0],1)) #each row is a date, column a new sample
                    md = 2*expit(-1*theta_md* prop_sim)
                    md[:logodds.shape[0]] = 1
                    #make logodds by appending post ban values
                    logodds = np.append(logodds, X2@ post_values, axis=0)
                    
                else:
                    #take right size of md to be N by N
                    md = np.tile(samples_sim['md'].values, (df_state.shape[0],1))

                    #set initial pre ban values of md to 1
                    md[:logodds.shape[0],:] = 1

                    #make logodds by appending post ban values
                    logodds = np.append(logodds, X2@ post_values, axis=0)
            
            if gamma:
                if type(R)==str: #'state'
                    try:
                        sim_R = samples_sim['R_'+states_initials[state]]
                    except KeyError:
                        #this state not fitted, use gamma prior on initial value
                        print("using initial value for state" +state)
                        sim_R = np.random.gamma(
                        shape=df.loc[df.date=='2020-03-01','mean'].mean()**2/0.2,
                      scale=0.2/df.loc[df.date=='2020-03-01','mean'].mean(),
                                size=df_state.shape[0]
                            )
                if type(R)==dict:
                    if states_initials[state] != ['ACT','NT']:
                        #if state, use inferred
                        sim_R = np.tile(R[states_initials[state]][:samples_sim.shape[0]], (df_state.shape[0],1))
                    else:
                        #if territory, use generic R_L
                        sim_R = np.tile(samples_sim.R_L.values, (df_state.shape[0],1))
                else:
                    sim_R = np.tile(samples_sim.R_L.values, (df_state.shape[0],1))
                mu_hat = 2 *md*sim_R* expit(logodds)
                if winter:
                    mu_hat = (1+samples_sim['winter'].values)*mu_hat
                if rho:
                    if rho=='data':
                        rho_data = np.tile(df_state.rho_moving.values[np.newaxis].T,
                                           (1,samples_sim.shape[0]))
                    else:
                        states_to_fitd = {s: i+1 for i,s in enumerate(rho)      }
                        if states_initials[state] in states_to_fitd.keys():
                            #transpose as columns are days, need rows to be days
                            if second_phase:
                                #use brho_v

                                rho_data = samples_sim[
                                    ['brho_v['+str(j)+']' 
                                        for j in range(pos, pos+df.loc[df.state==states_initials[state]].is_sec_wave.sum() ) ]
                                ].values.T

                                pos = pos + df.loc[df.state==states_initials[state]].is_sec_wave.sum()
                            else:
                                # first phase
                                rho_data = samples_sim[
                                    ['brho['+str(j+1)+','+
                                    str(states_to_fitd[states_initials[state]])+']' 
                                        for j in range(df_state.shape[0])]].values.T 
                        else:
                            print("Using data as inference not done on {}".format(state))
                            rho_data = np.tile(df_state.rho_moving.values[np.newaxis].T,
                                               (1,samples_sim.shape[0]))
                    R_I_sim = np.tile(samples_sim.R_I.values, (df_state.shape[0],1))
                                           
                    mu_hat = rho_data * R_I_sim + (1- rho_data) *mu_hat
                    
                if var is not None:
                        ##Place the data derived delta here
                        delta = (np.sqrt(mu_hat).T/df_state['std'].values).T #double tranpose to ensure variance is  divided for each datum, not each sample  #size = mu_hat N by N / std 1 byN
                else:
                    if type(delta)==np.ndarray:
                            delta = np.random.choice(delta,size=df_state.shape[0])


                R_eff_hat = mu_hat #np.random.gamma(shape= mu_hat*delta, scale=1.0/delta)

            else:
                #Use normal distribution
                mu_hat = R * 2 * expit(logodds)
                if type(sigma)== pd.Series:    
                    sigma_i = sigma.sample(df_state.shape[0]).values
                else:
                    sigma_i = sigma
                    
                R_eff_hat = np.random.normal(mu_hat, sigma_i) # N by N, where rows = datum, column = sample from posterior

            df_hat = pd.DataFrame(R_eff_hat.T)

            if states_initials[state] not in rho:
                if i//3==1:
                    ax[i//3,i%3].tick_params(axis='x',rotation=90)
                continue
            #plot actual R_eff
            ax[i//3,i%3].plot(df_state.date, df_state['mean'], label='$R_{eff}$',color='C1')
            ax[i//3,i%3].fill_between(df_state.date, df_state['bottom'], df_state['top'],color='C1', alpha=0.3)
            ax[i//3,i%3].fill_between(df_state.date, df_state['lower'], df_state['upper'],color='C1', alpha=0.3)

            ax[i//3,i%3].plot(df_state.date,df_hat.quantile(0.5,axis=0), label='$\hat{\mu}$',color='C0')
            ax[i//3,i%3].fill_between(df_state.date, df_hat.quantile(0.25,axis=0), df_hat.quantile(0.75,axis=0),color='C0',alpha=0.3)
            ax[i//3,i%3].fill_between(df_state.date, df_hat.quantile(0.05,axis=0), df_hat.quantile(0.95,axis=0),color='C0',alpha=0.3)
            ax[i//3,i%3].set_title(state)
            
            #grid line at R_eff =1
            ax[i//3,i%3].set_yticks([1],minor=True,)
            ax[i//3,i%3].set_yticks([0,2,3],minor=False)
            ax[i//3,i%3].set_yticklabels([0,2,3],minor=False)
            ax[i//3,i%3].yaxis.grid(which='minor',linestyle='--',color='black',linewidth=2)
            ax[i//3,i%3].set_ylim((0,4))
            if i//3==1:
                ax[i//3,i%3].tick_params(axis='x',rotation=90)

    plt.legend()
    return ax

def read_in_cases(case_file_date='29Jun'):
    """
    Read in NNDSS data
    """
    #from data, find rho
    from datetime import timedelta
    import glob
    
    path = "data/COVID-19 UoM "+case_file_date+"*.xlsx"
    for file in glob.glob(path):
        df_NNDSS = pd.read_excel(file,
                       parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'],
                       dtype= {'PLACE_OF_ACQUISITION':str})
    if glob.glob(path) is None:
        print("No file found for ")
        print(path)
    df_NNDSS.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) #Fill blanks with simply unknown

    df_NNDSS['date_inferred'] = df_NNDSS.TRUE_ONSET_DATE
    df_NNDSS.loc[df_NNDSS.TRUE_ONSET_DATE.isna(),'date_inferred'] = df_NNDSS.loc[df_NNDSS.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
    df_NNDSS.loc[df_NNDSS.date_inferred.isna(),'date_inferred'] = df_NNDSS.loc[df_NNDSS.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)

    df_NNDSS['imported'] = df_NNDSS.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[:4]!='1101' else 0)
    df_NNDSS['local'] = 1 - df_NNDSS.imported


    df_state = df_NNDSS[['date_inferred','STATE','imported','local']].groupby(['STATE','date_inferred']).sum()

    df_state['rho'] = [ 0 if (i+l == 0) else i/(i+l) for l,i in zip(df_state.local,df_state.imported)  ]
    return df_state


def read_in_LSHTM():
    """
    Read in new LSHTM Reff csv from David Price et al
    """
    import pandas as pd
    path = "../data/LSHTM_Reff_estimates/Reff_LSHTM.csv"
    return pd.read_csv(path,parse_dates=['date','date_of_analysis'])