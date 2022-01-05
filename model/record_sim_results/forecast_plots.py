import sys

sys.path.insert(0,'model')
sys.path.insert(0,'model/record_sim_results')
from params import start_date, num_forecast_days, use_TP_adjustment
from collate_states import states
from helper_functions import read_in_NNDSS, read_in_Reff_file
from scipy.stats import beta
import os
from sys import argv
import json
from datetime import timedelta
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

plt.style.use("seaborn-poster")

def plot_results(df, int_vars: list, ax_arg=None, total=False, log=False, Reff=None,
                 plotpath=False, legend=False, summary=False, forecast_days=35):
    if ax_arg is None:
        if Reff is None:
            fig, ax = plt.subplots(figsize=(12, 9))
        else:
            #fig, (ax,ax2) = plt.subplots(figsize=(12,9),nrows=2,gridspec_kw={'height_ratios': [3, 1.5]}, sharex=True)
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(3, 1)
            ax = fig.add_subplot(gs[:2, 0])
            ax2 = fig.add_subplot(gs[2, 0], sharex=ax)
    elif Reff is not None:
        ax2 = ax_arg[1]
        ax = ax_arg[0]
    else:
        ax = ax_arg
    if summary:
        # Using summary files

        for var in int_vars:
            df.columns = df.columns.astype('datetime64[ns]')
            ax.fill_between(df.columns, df.loc[(var, 'lower')], df.loc[(var, 'upper')], alpha=0.4, color='C0')
            ax.fill_between(df.columns, df.loc[(var, 'bottom')], df.loc[(var, 'top')], alpha=0.2, color='C0')
            ax.fill_between(df.columns, df.loc[(var, 'lower10')], df.loc[(var, 'upper10')], alpha=0.2, color='C0')
            ax.fill_between(df.columns, df.loc[(var, 'lower15')], df.loc[(var, 'upper15')], alpha=0.2, color='C0')
            ax.fill_between(df.columns, df.loc[(var, 'lower20')], df.loc[(var, 'upper20')], alpha=0.2, color='C0')
            if plotpath:
                print("Cannot plot path using summary files")
                raise KeyError
            else:
                ax.plot(df.columns, df.loc[(var, 'median')], label=var)

            ax.set_xticks([df.columns.values[-1*forecast_days]], minor=True)
            # ax.xaxis.grid(which='minor', linestyle='--', alpha=0.6, color='black')

    else:
        # using the raw simulation files
        if total:
            for n in range(df.loc['symp_obs'].shape[0]):
                df.loc[('total_inci_obs', n), :] = df.loc[(
                    int_vars[0], n)] + df.loc[(int_vars[1], n)]
            int_vars = ['total_inci_obs']
        for var in int_vars:
            df.columns = df.columns.astype('datetime64[ns]')
            #ax.fill_between(df.columns, df.transpose()[var].quantile(0.05,axis=1), df.transpose()[var].quantile(0.95,axis=1), alpha=0.2,color='C0')
            ax.fill_between(df.columns, df.transpose()[var].quantile(0.25, axis=1), df.transpose()[var].quantile(0.75, axis=1), alpha=0.4, color='C0')

            if plotpath:
                n = 0
                good_sims = df.loc[~df.isna().any(axis=1)].index.get_level_values("sim")
                while True:
                    ax.plot(df.columns, df.loc[(var, good_sims[n])], label=var, alpha=0.8, color='C0', linewidth=0.5)
                    n += 1
                    if n >= 2000 or n >= good_sims.shape[0]:
                        break
            else:
                ax.plot(df.columns, df.transpose()[var].quantile(0.5, axis=1), label=var)

            ax.set_xticks([df.columns.values[-1*forecast_days]], minor=True)
            ax.xaxis.grid(b=True, which='minor', linestyle='--', alpha=0.6, color='black')

    if len(int_vars) > 1:
        ax.legend()
    ax.set_ylim(bottom=0)
    # ax.set_ylabel("Cases")
    if log:
        ax.set_yscale("log")
    if legend:
        fig.legend()

    if Reff is not None:

        ax2.plot(df.columns, Reff.loc[df.columns].mean(axis=1))
        ax2.fill_between(df.columns, Reff.loc[df.columns].quantile(0.25, axis=1), Reff.loc[df.columns].quantile(0.75, axis=1), alpha=0.4, color='C0')
        ax2.fill_between(df.columns, Reff.loc[df.columns].quantile(0.05, axis=1), Reff.loc[df.columns].quantile(0.95, axis=1), alpha=0.4, color='C0')
        ax2.set_yticks([1], minor=True,)
        ax2.set_yticks([0, 2, 4], minor=False)
        ax2.set_yticklabels([0, 2, 4], minor=False)
        ax2.yaxis.grid(which='minor', linestyle='--', color='black', linewidth=2)
        # ax2.set_ylabel("Reff")
        ax2.tick_params('x', rotation=90)
        plt.setp(ax.get_xticklabels(), visible=False)
        # ax2.set_xlabel("Date")

        ax2.set_xticks([df.columns.values[-1*forecast_days]], minor=True)
        # ax2.xaxis.grid(which='minor', linestyle='--', alpha=0.6, color='black')

        # ax2.set_ylim((0,3))
    else:
        # ax.set_xlabel("Date")
        ax.tick_params('x', rotation=90)
    if ax_arg is None:
        if Reff is None:
            return fig, ax
        else:
            return fig, ax, ax2
    elif Reff is not None:
        return ax, ax2
    else:
        return ax


def read_in_Reff(file_date, forecast_R=None, adjust_TP_forecast=False):
    """
    Read in Reff csv from Price et al 2020. Originals are in RDS, are converted to csv in R script
    """
    import pandas as pd

    df_forecast = read_in_Reff_file(file_date, adjust_TP_forecast=adjust_TP_forecast)
    df_forecast = df_forecast.loc[df_forecast.type == forecast_R]
    df_forecast.set_index(['state', 'date'], inplace=True)
    return df_forecast


def read_in_cases(cases_file_date, apply_delay_at_read=True):
    """
    Read in NNDSS case file data
    """
    import pandas as pd
    from datetime import timedelta
    import glob

    # read in the cases by applying the appropriate delays to the confirmation dates in the data, this should
    # mean that the plots align more appropriately
    df_NNDSS = read_in_NNDSS(cases_file_date, apply_delay_at_read=apply_delay_at_read)

    df_cases_state_time = df_NNDSS.groupby(['STATE', 'date_inferred'])[['imported', 'local']].sum()
    df_cases_state_time.reset_index(inplace=True)

    df_cases_state_time['cum_imported'] = df_cases_state_time.groupby('STATE').imported.transform(pd.Series.cumsum)
    df_cases_state_time['cum_local'] = df_cases_state_time.groupby('STATE').local.transform(pd.Series.cumsum)

    return df_cases_state_time


# use arguments to make sure we have the right files
n_sims = int(argv[1])
data_date = argv[2]
adjust_TP_forecast = use_TP_adjustment

forecast_type = 'R_L'
df_cases_state_time = read_in_cases(data_date)
Reff = read_in_Reff(forecast_R=forecast_type, file_date=data_date, adjust_TP_forecast=adjust_TP_forecast)
# states = ['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA', 'ACT', 'NT']

data_date = pd.to_datetime(data_date, format="%Y-%m-%d")
end_date = data_date + pd.Timedelta(days=num_forecast_days)
days = (end_date - pd.to_datetime(start_date, format="%Y-%m-%d")).days

# check if any dates are incorrect
try:
    num_bad_dates = df_cases_state_time.loc[(df_cases_state_time.date_inferred <= '2020-01-01')].shape[0]
    assert num_bad_dates == 0, "Data contains {} bad dates".format(
        num_bad_dates)
except AssertionError:
    print("Bad dates include:")
    print(df_cases_state_time.loc[(df_cases_state_time.date_inferred <= '2020-01-01')])


end_date = pd.to_datetime(
    start_date, format='%Y-%m-%d') + timedelta(days=days-1)

print("forecast up to: {}".format(end_date))


df_results = pd.read_parquet("results/quantiles"+forecast_type+start_date+"sim_"+str(n_sims)+"days_"+str(days)+".parquet")


df_cases_state_time = df_cases_state_time[df_cases_state_time.date_inferred != 'None']
df_cases_state_time.date_inferred = pd.to_datetime(df_cases_state_time.date_inferred)

df_results = pd.melt(df_results, id_vars=['state', 'date', 'type'],
                     value_vars=['bottom', 'lower', 'median', 'upper', 'top',
                     'lower10', 'upper10', 'lower15', 'upper15',
                                 'lower20', 'upper20',
                                 ],
                     )

df_results = pd.pivot_table(df_results,
                            index=['state', 'type', 'variable'],
                            columns='date',
                            values='value')

with open("results/good_sims"+str(n_sims)+"days_"+str(days)+".json", 'r') as file:
    good_sims = json.load(file)


# Local cases plot
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(4, 2)
axes = []
for i, state in enumerate(states):

    print("Number of sims not rejected for state " + state + " is %i" % len(good_sims[state]))
    Reff_used = [r % 2000 for r in good_sims[state]]
    print("Number of unique Reff paths not rejected is %i " %len(set(Reff_used)))

    # Plots
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
    ax = fig.add_subplot(gs0[:2, 0])
    ax2 = fig.add_subplot(gs0[2, 0], sharex=ax)
    axes.append(ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= start_date)
        & (df_cases_state_time.date_inferred <= end_date)]

    ax.bar(dfplot.date_inferred, dfplot.local, label='Actual', color='grey', alpha=0.6)
    R_plot = [r % 2000 for r in good_sims[state]]

    ax, ax2 = plot_results(df_results.loc[state], ['total_inci_obs'], ax_arg=(
        ax, ax2), summary=True, Reff=Reff.loc[state, R_plot])

    if i % 2 == 0:
        ax.set_ylabel("Observed \n local cases")
        ax2.set_ylabel("TP")
    ax.set_title(state)
    
    # if state == 'NSW':
    #     ax.axhline(200, linestyle='--', color='black')
    
    if i < len(states)-2:
        ax.set_xticklabels([])
        ax.set_xlabel('')
plt.tight_layout()
plt.savefig("figs/"+str(data_date.date())+forecast_type+"local_inci_" +
            str(n_sims)+"days_"+str(days)+'.png', dpi=300)

# Also produce a plot that shows the median more clearly.
# try:
for i, state in enumerate(states):
    ax = axes[i]
    print('max median', max(df_results.loc[state].loc[('total_inci_obs', 'median')])*1.5+10)
    ax.set_ylim((0, max(df_results.loc[state].loc[('total_inci_obs', 'median')])*1.5+10))

plt.savefig("figs/"+str(data_date.date())+forecast_type+"local_inci_median_" + str(n_sims)+"days_"+str(days)+'.png', dpi=300)

# Make a single plot for each state
os.makedirs("figs/single_state_plots/", exist_ok=True)
for i, state in enumerate(states):
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(3, 1)
    ax = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= start_date)
        & (df_cases_state_time.date_inferred <= end_date)]

    ax.bar(dfplot.date_inferred, dfplot.local, label='Actual', color='grey', alpha=0.6)
    R_plot = [r % 2000 for r in good_sims[state]]
    ax, ax2 = plot_results(df_results.loc[state], ['total_inci_obs'], ax_arg=(ax, ax2), summary=True, Reff=Reff.loc[state, R_plot])
    
    # ax2.set_xlim(left=pd.to_datetime(), right=pd.to_datetime('2021-12-21')+pd.Timedelta(days=7)) 
    # ax2.set_xlim(left=pd.to_datetime('2021-11-23'), right=pd.to_datetime('2021-12-21')+pd.Timedelta(days=7)) 
    # ax.axvline(pd.to_datetime('2021-12-21'), ls='--', color='black', alpha=0.2, linewidth=2)
    # ax2.axvline(pd.to_datetime('2021-12-21'), ls='--', color='black', alpha=0.2, linewidth=2)
    ax.set_ylim(0, 600)

    ax.set_ylabel("Observed \n local cases")
    ax2.set_ylabel("TP")
    ax.set_title(state)
    plt.tight_layout()
    plt.savefig("figs/single_state_plots/"+state+"local_inci_" + str(n_sims)+"days_"+str(days)+'.png', dpi=300)


# Total cases
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(4, 2)
for i, state in enumerate(states):
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
    ax = fig.add_subplot(gs0[:2, 0])
    ax2 = fig.add_subplot(gs0[2, 0], sharex=ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= start_date)
        & (df_cases_state_time.date_inferred <= end_date)]

    ax.bar(dfplot.date_inferred, dfplot.local, label='Actual', color='grey', alpha=0.6)
    if len(set(good_sims[state])) == 0:
        # no accepted sim, skip
        continue
    ax, ax2 = plot_results(df_results.loc[state], ['total_inci'], ax_arg=(ax, ax2), summary=True, Reff=Reff.loc[state, R_plot])
    # if state=='NSW':
    #    ax.set_ylim((0,100))
    # elif state=='VIC':
    #    ax.set_ylim((0,600))
    # if (state=='VIC') or (state=='NSW'):
    #    ax.set_ylim((0,100))
    if i % 2 == 0:
        ax.set_ylabel("Total \nlocal cases")
        ax2.set_ylabel("TP")
    ax.set_title(state)
    if i < len(states)-2:
        ax.set_xticklabels([])
        ax.set_xlabel('')
plt.tight_layout()
plt.savefig("figs/"+str(data_date.date())+forecast_type+"local_total_" + str(n_sims)+"days_"+str(days)+'.png', dpi=300)


# asymp cases
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(4, 2)
for i, state in enumerate(states):
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
    ax = fig.add_subplot(gs0[:2, 0])
    ax2 = fig.add_subplot(gs0[2, 0], sharex=ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= start_date)
        & (df_cases_state_time.date_inferred <= end_date)]

    ax.bar(dfplot.date_inferred, dfplot.local,
           label='Actual', color='grey', alpha=0.6)
    if len(set(good_sims[state])) == 0:
        # no accepted sim, skip
        continue
    ax, ax2 = plot_results(df_results.loc[state], ['asymp_inci'], ax_arg=(ax, ax2), summary=True, Reff=Reff.loc[state])

    # ax.set_ylim(top=70)
    # if (state=='VIC') or (state=='NSW'):
    #    ax.set_ylim((0,100))
    if i % 2 == 0:
        ax.set_ylabel("Asymp \ntotal cases")
        ax2.set_ylabel("TP")
    ax.set_title(state)
    if i < len(states)-2:
        ax.set_xticklabels([])
        ax.set_xlabel('')
plt.tight_layout()
plt.savefig("figs/"+str(data_date.date())+forecast_type+"asymp_inci_"+str(n_sims) + "days_"+str(days)+'.png', dpi=144)
# Imported cases
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(4, 2)
for i, state in enumerate(states):
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
    ax = fig.add_subplot(gs0[:2, 0])
    ax2 = fig.add_subplot(gs0[2, 0], sharex=ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= start_date)
        & (df_cases_state_time.date_inferred <= end_date)]

    ax.bar(dfplot.date_inferred, dfplot.imported, label='Actual', color='grey', alpha=0.6)
    if len(set(good_sims[state])) == 0:
        # no accepted sim, skip
        continue
    ax, ax2 = plot_results(df_results.loc[state], ['imports_inci_obs'], ax_arg=(ax, ax2), summary=True, Reff=Reff.loc[state])

    # ax.set_ylim(top=70)
    # if (state=='VIC') or (state=='NSW'):
    #    ax.set_ylim((0,100))
    if i % 2 == 0:
        ax.set_ylabel("Observed \nimported cases")
        ax2.set_ylabel("TP")
    ax.set_title(state)
    if i < len(states)-2:
        ax.set_xticklabels([])
        ax.set_xlabel('')

plt.tight_layout()
plt.savefig("figs/"+str(data_date.date())+forecast_type+"imported_inci_" + str(n_sims)+"days_"+str(days)+'.png', dpi=300)

# unobserved Imported cases
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(4, 2)
for i, state in enumerate(states):
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
    ax = fig.add_subplot(gs0[:2, 0])
    ax2 = fig.add_subplot(gs0[2, 0], sharex=ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= start_date)
        & (df_cases_state_time.date_inferred <= end_date)]

    ax.bar(dfplot.date_inferred, dfplot.imported, label='Actual', color='grey', alpha=0.6)
    if len(set(good_sims[state])) == 0:
        # no accepted sim, skip
        continue
    ax, ax2 = plot_results(df_results.loc[state], ['imports_inci'], ax_arg=(ax, ax2), summary=True, Reff=Reff.loc[state])

    # ax.set_ylim(top=70)
    if i % 2 == 0:
        ax.set_ylabel("Total Imported cases")
        ax2.set_ylabel("TP")
    ax.set_title(state)
    if i < len(states)-2:
        ax.set_xticklabels([])
        ax.set_xlabel('')

plt.tight_layout()
plt.savefig("figs/"+str(data_date.date())+forecast_type+"imported_unobs_"+str(n_sims) +
            "days_"+str(days)+'.png', dpi=144)

# Local cases, spaghetti plot
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(4, 2)

plot_start = pd.to_datetime(data_date) - pd.to_timedelta(60, unit="D")
dates_plot = pd.date_range(start=plot_start, periods=89)
for i, state in enumerate(states):

    try: 
        df_raw = pd.read_parquet("results/"+state+start_date+"sim_"+forecast_type+str(
            n_sims)+"_seed_days_"+str(days)+".parquet",
            columns=[d.strftime("%Y-%m-%d") for d in dates_plot])
    except:
        df_raw = pd.read_parquet("results/"+state+start_date+"sim_"+forecast_type+str(
            n_sims)+"days_"+str(days)+".parquet",
            columns=[d.strftime("%Y-%m-%d") for d in dates_plot])

    Reff_used = [r % 2000 for r in good_sims[state]]

    # plots
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
    ax = fig.add_subplot(gs0[:, 0])
    #ax2 = fig.add_subplot(gs0[2,0], sharex=ax)

    dfplot = df_cases_state_time.loc[
        (df_cases_state_time.STATE == state)
        & (df_cases_state_time.date_inferred >= dates_plot[0])
        & (df_cases_state_time.date_inferred <= dates_plot[-1])]

    R_plot = [r % 2000 for r in good_sims[state]]
    ax.bar(dfplot.date_inferred, dfplot.local,
           label='Actual', color='grey', alpha=0.6)
    ylims = ax.get_ylim()
    if len(set(good_sims[state])) == 0:
        # no accepted sim, skip
        continue
    ax = plot_results(df_raw, ['total_inci_obs'], ax_arg=ax, summary=False, plotpath=True)
    spag_ylim = ax.get_ylim()

    # if (state == 'VIC') or (state == 'NSW'):
    #     ax.set_ylim((0, 100))
    # elif spag_ylim[1] > ylims[1]:
    #     ax.set_ylim((ylims[0], 5*ylims[1]))

    if i % 2 == 0:
        ax.set_ylabel("Observed \n local cases")
    ax.set_title(state)
    if i < len(states)-2:
        ax.set_xticklabels([])
        ax.set_xlabel('')

    ax.set_xticks([df_raw.columns.values[-1*31]], minor=True)
    ax.xaxis.grid(which='minor', linestyle='--', alpha=0.6, color='black')
plt.tight_layout()
plt.savefig("figs/"+str(data_date.date())+forecast_type+"spagh"+str(n_sims) +
            "days_"+str(days)+'.png', dpi=300)
