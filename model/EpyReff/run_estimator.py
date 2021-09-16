###
# Run EpyReff on NNDSS data
###

from scipy.stats import gamma
from sys import argv
from epyreff import *
# this is not used in the estimation routine, it just lets the plot know what we ignore
from params import truncation_days
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs.tslibs.timedeltas import Timedelta
import matplotlib
print('Running EpyReff on NNDSS data')

matplotlib.use('Agg')

# parameters
tau = 4
prior_a = 1
prior_b = 2
trunc_days = 21

shape_inc = 5.807  # taken from Lauer et al. 2020 #1.62/0.418
scale_inc = 0.948  # 0.418
offset_inc = 0

shape_rd = 1.82
scale_rd = 2.88
offset_rd = 0

shape_gen = 3.64/3.07
scale_gen = 3.07
offset = 0
shift = 0

date = argv[1]
dt_date = pd.to_datetime(date, format="%Y-%m-%d")
file_date = dt_date.strftime("%Y-%m-%d")
try:
    plot_time = argv[2]
except:
    plot_time = False

# Read in the data

# read in case file data
print(dt_date.strftime("%d%b%Y"))
df_interim = read_cases_lambda(dt_date.strftime("%d%b%Y"))

# generate dataframe with id_vars date and state, variable SOURCE and number of cases
df_linel = tidy_cases_lambda(df_interim)

# infer extra cases in last 10 days by the reporting delay distribution


# generate possible infection dates from the notification data
df_inf = draw_inf_dates(df_linel, nreplicates=1000,
                        shape_rd=shape_rd, scale_rd=scale_rd, offset_rd=offset_rd,
                        shape_inc=shape_inc, scale_inc=scale_inc, offset_inc=offset_inc,
                        )

# reindex dataframe to include all dates,
# return df with index (STATE, INFECTION_DATE, SOURCE), columns are samples
df_inc_zeros = index_by_infection_date(df_inf)


# get all lambdas
lambda_dict = lambda_all_states(df_inc_zeros,
                                shape_gen=shape_gen, scale_gen=scale_gen, offset=offset,
                                trunc_days=trunc_days)

states = [*df_inc_zeros.index.get_level_values('STATE').unique()]
R_summary_states = {}
dates = {}
df = pd.DataFrame()
for state in states:
    lambda_state = lambda_dict[state]
    df_state_I = df_inc_zeros.xs((state, 'local'), level=('STATE', 'SOURCE'))
    # get Reproduciton numbers
    a, b, R = Reff_from_case(
        df_state_I.values, lambda_state, prior_a=prior_a, prior_b=prior_b, tau=tau)

    # summarise for plots and file printing
    R_summary_states[state] = generate_summary(R)
    dates[state] = df_state_I.index.values[trunc_days-1+tau:]

    temp = pd.DataFrame.from_dict(R_summary_states[state])
    temp['INFECTION_DATES'] = dates[state]
    temp['STATE'] = state
    #temp.index = pd.MultiIndex.from_product(([state], dates[state]))
    df = df.append(temp, ignore_index=True)

# make folder to record files
os.makedirs("results/EpyReff/", exist_ok=True)


if plot_time:
    # plot assumed distributions
    inc_period = offset_inc+np.random.gamma(shape_inc, scale_inc, size=1000)
    rep_delay = offset_rd+np.random.gamma(shape_rd, scale_rd, size=1000)

    # generation interval discretised
    # Find midpoints for discretisation
    xmids = [x+shift for x in range(trunc_days+1)]
    # double check parameterisation of scipy
    gamma_vals = gamma.pdf(xmids, a=shape_gen, scale=scale_gen)
    # renormalise the pdf
    disc_gamma = gamma_vals/sum(gamma_vals)
    ws = disc_gamma[:trunc_days]
    # offset
    ws[offset:] = disc_gamma[:trunc_days-offset]
    ws[:offset] = 0

    fig, ax = plt.subplots(figsize=(12, 18), nrows=3, sharex=True)

    ax[0].hist(rep_delay, bins=50, density=True)
    ax[0].set_title("Reporting Delay")
    ax[1].hist(inc_period, bins=50, density=True)
    ax[1].set_title("Incubation Period")
    ax[2].bar(xmids[:-1], height=ws, width=1)
    ax[2].set_title("Generation Interval")

    plt.savefig('figs/Time_distributions'+file_date +
                "tau_"+str(tau)+".png", dpi=400)


df.to_csv('results/EpyReff/Reff'+file_date+"tau_"+str(tau)+".csv", index=False)

# plot all the estimates
file_date_updated = dt_date
file_date_updated = file_date_updated.strftime("%Y-%m-%d")

fig, ax = plot_all_states(R_summary_states, df_interim, dates,
                          start='2020-03-01', end=file_date_updated, save=True,
                          tau=tau, date=date, nowcast_truncation=-truncation_days
                          )
plt.close()
