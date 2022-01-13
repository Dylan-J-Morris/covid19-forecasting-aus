import os
import sys
from numpy.core.numeric import zeros_like
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')
# I hate this too but it allows everything to use the same helper functions.
sys.path.insert(0, 'model')
from helper_functions import read_in_NNDSS
from Reff_constants import *
from params import alpha_start_date, delta_start_date, omicron_start_date, vaccination_start_date, apply_voc_to_R_L_hats, apply_vacc_to_R_L_hats

def read_in_posterior(date):
    """
    read in samples from posterior from inference
    """
    df = pd.read_hdf("results/soc_mob_posterior"+date+".h5", key='samples')

    return df


def read_in_google(Aus_only=True, local=False, moving=False):
    """
    Read in the Google data set
    """
    if local:
        if type(local) == str:
            df = pd.read_csv(local, parse_dates=['date'])
        elif type(local) == bool:
            local = 'data/Global_Mobility_Report.csv'
            df = pd.read_csv(local, parse_dates=['date'])
    else:
        # Download straight from the web
        df = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', parse_dates=['date'])
        # Make it save automatically.
        df.to_csv('data/Global_Mobility_Report.csv', index=False)

    if Aus_only:
        df = df.loc[df.country_region_code == 'AU']
        # Change state column to state initials
        df['state'] = df.sub_region_1.map(lambda x: states_initials[x] if not pd.isna(x) else 'AUS')
        
    df = df.loc[df.sub_region_2.isna()]
    if moving:
        # generate moving average columns in reverse
        df = df.sort_values(by='date')
        mov_values = []
        for val in value_vars:
            mov_values.append(val[:-29]+'_7days')
            df[mov_values[-1]] = df.groupby(['state'])[val].transform(lambda x: x[::-1].rolling(7, 1).mean()[::-1])  # minimumnumber of 1

            # minimum of 7 days for std, forward fill the rest
            df[mov_values[-1]+'_std'] = df.groupby(['state'])[val].transform(lambda x: x[::-1].rolling(7, 7).std()[::-1])
            # fill final values as std doesn't work with single value
            df[mov_values[-1]+'_std'] = df.groupby('state')[mov_values[-1]+'_std'].fillna(method='ffill')
            
    # show latest date
    print("Latest date in Google indices " + str(df.date.values[-1]))
    return df


def predict_plot(samples, df, third_date_range=None, split=True, gamma=False, moving=True, grocery=True,
                 delta=1.0, R=2.2, sigma=1, md_arg=None,
                 ban='2020-03-16', single=False, var=None,
                 rho=None, R_I=None, winter=False, prop=None, masks_prop=None,
                 second_phase=False, third_phase=False, 
                 vaccination=None, third_states=None, prop_omicron_to_delta=None):
    """
    Produce posterior predictive plots for all states
    """
    from scipy.special import expit
    from params import third_start_date

    os.makedirs("results/third_wave_fit/", exist_ok=True)

    value_vars = ['retail_and_recreation_percent_change_from_baseline',
                  'grocery_and_pharmacy_percent_change_from_baseline',
                  'parks_percent_change_from_baseline',
                  'transit_stations_percent_change_from_baseline',
                  'workplaces_percent_change_from_baseline',
                  'residential_percent_change_from_baseline']
    value_vars.remove('residential_percent_change_from_baseline')
    
    if not grocery:
        value_vars.remove('grocery_and_pharmacy_percent_change_from_baseline')
    if moving:
        value_vars = [val[:-29]+'_7days' for val in value_vars]

    if single:
        # Single state
        fig, ax = plt.subplots(figsize=(12, 9))
        df_state = df
        post_values = samples[[
            'beta['+str(i)+']' for i in range(1, 1+len(value_vars))]].sample(df_state.shape[0]).values.T
        if split:
            # split model with parameters pre and post policy

            df1 = df_state.loc[df_state.date <= ban]
            df2 = df_state.loc[df_state.date > ban]
            X1 = df1[value_vars]/100  # N by K
            X2 = df2[value_vars]/100
            # N by K times (Nsamples by K )^T = N by N
            logodds = X1 @ post_values

            if md is None:
                post_alphas = samples[['alpha['+str(i)+']' 
                                       for i in range(1, 1+len(value_vars))]].sample(df_state.shape[0]).values.T
                logodds = np.append(logodds, X2 @ (post_values + post_alphas), axis=0)
            else:
                # take right size of md
                md = np.random.choice(md, size=df_state.shape[0])

                # set initial pre ban values of md to 1
                md[:logodds.shape[0]] = np.ones(size=logodds.shape[0])

                # make logodds by appending post ban values
                logodds = np.append(logodds, X2 @ post_values, axis=0)

        else:
            X1 = df_state[value_vars]/100
            # N by K times (Nsamples by K )^T = N by N
            logodds = X1 @ post_values
        if gamma:
            if type(md) == np.ndarray:
                mu_hat = 2 * expit(logodds) * policy * md
            else:
                mu_hat = 2 * expit(logodds)

            if type(delta) == np.ndarray:
                delta = np.random.choice(delta, size=df_state.shape[0])
                
            R = np.random.choice(R, size=df_state.shape[0])
            R_eff_hat = np.random.gamma(shape=R * mu_hat*delta, scale=1.0/delta)
        else:
            # Use normal distribution
            mu_hat = R * 2 * expit(logodds)
            if type(sigma) == pd.Series:
                sigma_i = sigma.sample(df_state.shape[0]).values
            else:
                sigma_i = sigma
                
            # N by N, where rows = datum, column = sample from posterior
            R_eff_hat = np.random.normal(mu_hat, sigma_i)

        df_hat = pd.DataFrame(R_eff_hat.T)

        # plot actual R_eff
        ax.plot(df_state.date, df_state['mean'], label='R_eff from Price et al')
        ax.fill_between(df_state.date, df_state['bottom'], df_state['top'], color='C0', alpha=0.3)

        ax.plot(df_state.date, df_hat.quantile(0.5, axis=0), label='R_eff_hat', color='C1')
        ax.fill_between(df_state.date, df_hat.quantile(0.25, axis=0), df_hat.quantile(0.75, axis=0), color='C1', alpha=0.3)
        ax.fill_between(df_state.date, df_hat.quantile(0.05, axis=0), df_hat.quantile(0.95, axis=0), color='C1', alpha=0.3)

        # grid line at R_eff =1
        ax.set_yticks([1], minor=True,)
        ax.yaxis.grid(b=True, which='minor', linestyle='dashed', color='grey')
        ax.tick_params(axis='x', rotation=90)

    else:
        # all states
        fig, ax = plt.subplots(figsize=(15, 12), ncols=4, nrows=2, sharex=True, sharey=True)

        states = sorted(list(states_initials.keys()))
        
        if not third_phase:
            states.remove('Northern Territory')
            states.remove('Australian Capital Territory')
            
        # no R_eff modelled for these states, skip
        # counter for brho_v
        pos = 1
        for i, state in enumerate(states):

            df_state = df.loc[df.sub_region_1 == state]
            if second_phase:
                df_state = df_state.loc[df_state.is_sec_wave == 1]
            elif third_phase:
                df_state = df_state.loc[df_state.is_third_wave == 1] 
                
            if third_phase:    
                masks_prop_sim = masks_prop[states_initials[state]].values[:df_state.shape[0]]
                
            samples_sim = samples.sample(1000)
            post_values = samples_sim[['bet['+str(i)+']' for i in range(1, 1+len(value_vars))]].values.T
            prop_sim = prop[states_initials[state]].values[:df_state.shape[0]]
            
            if split:

                # split model with parameters pre and post policy
                df1 = df_state.loc[df_state.date <= ban]
                df2 = df_state.loc[df_state.date > ban]
                X1 = df1[value_vars]/100  # N by K
                X2 = df2[value_vars]/100
                # N by K times (Nsamples by K )^T = N by N
                logodds = X1 @ post_values

                if md_arg is None:
                    post_alphas = samples_sim[['alpha['+str(i)+']' for i in range(1, 1+len(value_vars))]].values.T
                    logodds = np.append(logodds, X2 @ (post_values + post_alphas), axis=0)
                    md = 1
                elif md_arg == 'power':
                    theta_md = samples_sim.theta_md.values  # 1 by samples shape
                    # each row is a date, column a new sample
                    theta_md = np.tile(theta_md, (df_state.shape[0], 1))
                    md = ((1+theta_md).T**(-1 * prop_sim)).T
                    # set preban md values to 1
                    md[:logodds.shape[0]] = 1
                    if third_phase:
                        theta_masks = samples_sim.theta_masks.values  # 1 by samples shape
                        # each row is a date, column a new sample
                        theta_masks = np.tile(theta_masks, (df_state.shape[0], 1))
                        masks = ((1+theta_masks).T**(-1 * masks_prop_sim)).T
                        # set preban mask values to 1
                        masks[:logodds.shape[0]] = 1
                    # make logodds by appending post ban values
                    logodds = np.append(logodds, X2 @ post_values, axis=0)
                elif md_arg == 'logistic':
                    theta_md = samples_sim.theta_md.values  # 1 by samples shape
                    # each row is a date, column a new sample
                    theta_md = np.tile(theta_md, (df_state.shape[0], 1))
                    md = 2*expit(-1*theta_md * prop_sim)
                    md[:logodds.shape[0]] = 1
                    # make logodds by appending post ban values
                    logodds = np.append(logodds, X2 @ post_values, axis=0)
                else:
                    # take right size of md to be N by N
                    md = np.tile(samples_sim['md'].values,(df_state.shape[0], 1))
                    # set initial pre ban values of md to 1
                    md[:logodds.shape[0], :] = 1
                    # make logodds by appending post ban values
                    logodds = np.append(logodds, X2 @ post_values, axis=0)

                # grab posterior sampled vaccination effects here and multiply by the daily efficacy
                if vaccination is not None and states_initials[state] in third_states:
                    # transposing the vaccination sampled values so that it can be multiplied by the data
                    # the str(i+1) is required because the state indexing starts at 0
                    
                    # print(vaccination.loc[states_initials[state]])
                    idx = (
                        (vaccination.columns >= third_date_range[states_initials[state]][0]) & 
                        (vaccination.columns <= third_date_range[states_initials[state]][-1]) 
                    )
                    vacc_ts_data = vaccination.loc[states_initials[state]][idx]
                    
                    third_states_indices = {state: index+1 for (index, state) in enumerate(third_states)}

                    third_days = {k: v.shape[0] for (k, v) in third_date_range.items()}
                    third_days_cumulative = np.append([0], np.cumsum([v for v in third_days.values()]))
                    vax_idx_ranges = {k: range(third_days_cumulative[i], third_days_cumulative[i+1]) for (i, k) in enumerate(third_days.keys())}
                    third_days_tot = sum(v for v in third_days.values())
                    # get the sampled vaccination effect (this will be incomplete as it's only over the fitting period)
                    sampled_vax_effects_all = samples_sim[["vacc_effect[" + str(j)  + "]" for j in range(1, third_days_tot+1)]].T
                    vacc_tmp = sampled_vax_effects_all.iloc[vax_idx_ranges[states_initials[state]],:]
                    # now we layer in the posterior vaccine multiplier effect which ill be a (T,mob_samples) array
                    if states_initials[state] in third_states:
                        eta = samples_sim['eta[' + str(third_states_indices[states_initials[state]]) + ']']
                        r = samples_sim['eta[' + str(third_states_indices[states_initials[state]]) + ']']
                    else:
                        eta = samples_sim['eta[1]']
                        r = samples_sim['r[1]']
                        
                    # get before and after fitting and tile them
                    vacc_ts_data_before = pd.concat(
                        [vacc_ts_data.loc[vacc_ts_data.index < third_date_range[states_initials[state]][0]]] * eta.shape[0], 
                        axis=1
                    )
                    vacc_ts_data_after = pd.concat(
                        [vacc_ts_data.loc[vacc_ts_data.index > third_date_range[states_initials[state]][-1]]] * eta.shape[0], 
                        axis=1
                    )
                    # rename columns for easy merging
                    vacc_ts_data_before.columns = vacc_tmp.columns
                    vacc_ts_data_after.columns = vacc_tmp.columns
                    # merge in order
                    vacc_ts = pd.concat(
                        [vacc_ts_data_before, vacc_tmp, vacc_ts_data_after], axis=0, ignore_index=True         
                    )
                    # reset the index to be the dates for easier information handling
                    vacc_ts.set_index(vacc_ts_data.index, inplace=True)
                    
                    vacc_ts = vacc_tmp
                    
                    third_states_indices = {state: index+1 for (index, state) in enumerate(third_states)}

                    # From conversations with James and Nic we think the heterogeneity / assortativity was more prominent before late 
                    # August (hence the fixed date) 
                    # in order for this to be correctly applied in the plot, we need to get the start dates after the beginning of 
                    # the third wave data which we determine based off the third_date_range 
                    heterogeneity_delay_start_day = (pd.to_datetime('2021-08-20') - third_date_range[states_initials[state]][0]).days
                    omicron_start_day = (pd.to_datetime(omicron_start_date) - third_date_range[states_initials[state]][0]).days
                    omicron_start_day = 0 if omicron_start_day < 0 else omicron_start_day
                    
                    # this will hold the posterior VE, with adjustement factors
                    vacc_post = np.zeros_like(vacc_ts)

                    # loop ober days in third wave and apply the appropriate form (i.e. decay or not)
                    # note that in here we apply the entire sample to the vaccination data to create a days by samples array
                    # set the full vaccination data as the mean 
                    # vacc_sig = 0.001
                    # vacc_mu = vacc_sim
                    # # calculate shape and scale 
                    # a_vacc = vacc_mu*(vacc_mu*(1-vacc_mu)/vacc_sig - 1)
                    # b_vacc = (1-vacc_mu)*(vacc_mu*(1-vacc_mu)/vacc_sig - 1)
                    # # sample a noisier version of the vax effect
                    # vacc_sim_adj = np.random.beta(a_vacc, b_vacc)
                    # create zero array to fill in with the full vaccine effect model
                    vacc_post = np.zeros_like(vacc_ts)
                    
                    days_into_omicron = np.cumsum(np.append([0], [(v >= pd.to_datetime(omicron_start_date)).sum() for v in third_date_range.values()]))
                    idx = {}
                    kk = 0
                    for k in third_date_range.keys():
                        idx[k] = range(days_into_omicron[kk], days_into_omicron[kk+1])
                        kk += 1
                    
                    m = prop_omicron_to_delta.iloc[:, idx[states_initials[state]]].to_numpy()
                    m = m[:vacc_post.shape[1]].T
                    
                    reduction_vacc_effect_omicron = samples_sim['reduction_vacc_effect_omicron'].to_numpy()
                    
                    # note that in here we apply the entire sample to the vaccination data to create a days by samples array
                    for ii in range(vacc_post.shape[0]):
                        if ii < heterogeneity_delay_start_day:
                            vacc_post[ii] = eta + (1-eta)*vacc_ts.iloc[ii, :]
                        elif ii < omicron_start_day:
                            # number of days after the heterogeneity should start to wane
                            heterogeneity_delay_days = ii - heterogeneity_delay_start_day
                            decay_factor = np.exp(-r*heterogeneity_delay_days)
                            vacc_post[ii] = eta*decay_factor + (1-eta*decay_factor)*vacc_ts.iloc[ii, :]
                        else:
                            # number of days after the heterogeneity should start to wane
                            heterogeneity_delay_days = ii - heterogeneity_delay_start_day
                            jj = ii - omicron_start_day
                            decay_factor = np.exp(-r*heterogeneity_delay_days)
                            # calculate the raw vax effect
                            vacc_tmp = eta*decay_factor + (1-eta*decay_factor)*vacc_ts.iloc[ii, :]
                            # calculate the full vaccination effect
                            vacc_post[ii] = 1+(m[jj] - m[jj]*reduction_vacc_effect_omicron - 1) * (1-vacc_tmp)

                    for ii in range(vacc_post.shape[0]):
                        if ii < df_state.loc[df_state.date < vaccination_start_date].shape[0]:
                            vacc_post[ii] = 1.0

            if gamma:
                if type(R) == str:  # 'state'
                    try:
                        sim_R = samples_sim['R_'+states_initials[state]]
                    except KeyError:
                        # this state not fitted, use gamma prior on initial value
                        print("using initial value for state" + state)
                        sim_R = np.random.gamma(
                            shape=df.loc[df.date == '2020-03-01','mean'].mean()**2/0.2,
                            scale=0.2/df.loc[df.date == '2020-03-01', 'mean'].mean(),
                            size=df_state.shape[0]
                        )
                if type(R) == dict:
                    if states_initials[state] != ['NT']:
                        # if state, use inferred
                        sim_R = np.tile(R[states_initials[state]][:samples_sim.shape[0]], (df_state.shape[0], 1))
                    else:
                        # if territory, use generic R_L
                        sim_R = np.tile(samples_sim.R_L.values,(df_state.shape[0], 1))
                else:
                    sim_R = np.tile(samples_sim.R_L.values,(df_state.shape[0], 1))

                if vaccination is not None and states_initials[state] in third_states:
                    mu_hat = 2 * md * masks * sim_R * expit(logodds) * vacc_post
                else:
                    mu_hat = 2 * md*sim_R * expit(logodds)

                if winter:
                    mu_hat = (1+samples_sim['winter'].values)*mu_hat
                if rho:
                    if rho == 'data':
                        rho_data = np.tile(df_state.rho_moving.values[np.newaxis].T, (1, samples_sim.shape[0]))
                    else:
                        states_to_fitd = {s: i+1 for i, s in enumerate(rho)}
                        if states_initials[state] in states_to_fitd.keys():
                            # transpose as columns are days, need rows to be days
                            if second_phase:
                                # use brho_v

                                rho_data = samples_sim[['brho_sec_wave['+str(j)+']' for j in range(pos, pos+df.loc[df.state == states_initials[state]].is_sec_wave.sum())]].values.T

                                pos = pos + df.loc[df.state == states_initials[state]].is_sec_wave.sum()
                            elif third_phase:
                                # use brho_v

                                rho_data = samples_sim[['brho_third_wave['+str(j)+']' 
                                                        for j in range(pos, pos+df.loc[df.state == states_initials[state]].is_third_wave.sum())]].values.T

                                voc_multiplier_alpha = samples_sim[['voc_effect_alpha']].values.T
                                voc_multiplier_delta = np.tile(samples_sim[['voc_effect_delta']].values.T, (mu_hat.shape[0], 1))
                                voc_multiplier_omicron = np.tile(samples_sim[['voc_effect_omicron']].values.T, (mu_hat.shape[0], 1))
                                # now we just modify the values before the introduction of the voc to be 1.0
                                voc_multiplier = np.zeros_like(voc_multiplier_delta)
                                
                                for ii in range(voc_multiplier.shape[0]):
                                    if ii < df_state.loc[df_state.date < alpha_start_date].shape[0]:
                                        voc_multiplier[ii] = 1.0
                                    elif ii < df_state.loc[df_state.date < delta_start_date].shape[0]:
                                        voc_multiplier[ii] = voc_multiplier_alpha[ii]
                                    elif ii < df_state.loc[df_state.date < omicron_start_date].shape[0]:
                                        voc_multiplier[ii] = voc_multiplier_delta[ii]
                                    else:
                                        jj = ii - df_state.loc[df_state.date < omicron_start_date].shape[0]
                                        voc_multiplier[ii] = m[jj]*voc_multiplier_omicron[ii] + (1-m[jj])*voc_multiplier_delta[ii]
                                            
                                # now modify the mu_hat
                                mu_hat *= voc_multiplier

                                pos = pos + df.loc[df.state == states_initials[state]].is_third_wave.sum()

                            else:
                                # first phase
                                rho_data = samples_sim[['brho['+str(j+1)+',' + str(states_to_fitd[states_initials[state]])+']' for j in range(df_state.shape[0])]].values.T
                        else:
                            print("Using data as inference not done on {}".format(state))
                            rho_data = np.tile(df_state.rho_moving.values[np.newaxis].T, (1, samples_sim.shape[0]))
                            

                    R_I_sim = np.tile(samples_sim.R_I.values, (df_state.shape[0], 1))
                    
                    mu_hat = rho_data * R_I_sim + (1 - rho_data) * mu_hat

                if var is not None:
                    # Place the data derived delta here
                    # double tranpose to ensure variance is  divided for each datum, not each sample  #size = mu_hat N by N / std 1 byN
                    delta = (np.sqrt(mu_hat).T/df_state['std'].values).T
                else:
                    if type(delta) == np.ndarray:
                        delta = np.random.choice(delta, size=df_state.shape[0])

                R_eff_hat = mu_hat

            else:
                # Use normal distribution
                mu_hat = R * 2 * expit(logodds)
                if type(sigma) == pd.Series:
                    sigma_i = sigma.sample(df_state.shape[0]).values
                else:
                    sigma_i = sigma

                # N by N, where rows = datum, column = sample from posterior
                R_eff_hat = np.random.normal(mu_hat, sigma_i)

            df_hat = pd.DataFrame(R_eff_hat.T)

            if states_initials[state] not in rho:
                if i//4 == 1:
                    ax[i//4, i % 4].tick_params(axis='x', rotation=90)
                continue
            # plot actual R_eff
            ax[i//4, i % 4].plot(df_state.date, df_state['mean'], label='$R_{eff}$', color='C1')
            ax[i//4, i % 4].fill_between(df_state.date, df_state['bottom'], df_state['top'], color='C1', alpha=0.3)
            ax[i//4, i % 4].fill_between(df_state.date, df_state['lower'], df_state['upper'], color='C1', alpha=0.3)
            ax[i//4, i % 4].plot(df_state.date, df_hat.quantile(0.5, axis=0), label='$\hat{\mu}$', color='C0')
            ax[i//4, i % 4].fill_between(df_state.date, df_hat.quantile(0.25, axis=0), df_hat.quantile(0.75, axis=0), color='C0', alpha=0.3)
            ax[i//4, i % 4].fill_between(df_state.date, df_hat.quantile(0.05, axis=0), df_hat.quantile(0.95, axis=0), color='C0', alpha=0.3)
            ax[i//4, i % 4].set_title(state)
            # grid line at R_eff =1
            ax[i//4, i % 4].set_yticks([1], minor=True,)
            ax[i//4, i % 4].set_yticks([0, 2, 3], minor=False)
            ax[i//4, i % 4].set_yticklabels([0, 2, 3], minor=False)
            ax[i//4, i % 4].yaxis.grid(which='minor', linestyle='--', color='black', linewidth=2)
            ax[i//4, i % 4].set_ylim((0, 4))
            if i//4 == 1:
                ax[i//4, i % 4].tick_params(axis='x', rotation=90)

    plt.legend()
    return ax


def read_in_cases(case_file_date, apply_delay_at_read=False, apply_inc_at_read=False):
    """
    Read in NNDSS data and from data, find rho
    """
    from datetime import timedelta
    import glob

    df_NNDSS = read_in_NNDSS(case_file_date, apply_delay_at_read=apply_delay_at_read, apply_inc_at_read=apply_inc_at_read)
    
    df_state = df_NNDSS[['date_inferred', 'STATE', 'imported', 'local']].groupby(['STATE', 'date_inferred']).sum()

    df_state['rho'] = [0 if (i+l == 0) else i/(i+l) for l, i in zip(df_state.local, df_state.imported)]
    
    return df_state
