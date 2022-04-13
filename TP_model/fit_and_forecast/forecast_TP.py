import glob
import os
import sys

# these imports and usings need to be in the same order
sys.path.insert(0, "../")
sys.path.insert(0, "TP_model")
sys.path.insert(0, "TP_model/fit_and_forecast")
from Reff_functions import *
from Reff_constants import *
from params import (
    num_forecast_days,
    alpha_start_date,
    delta_start_date,
    omicron_start_date,
    truncation_days,
    third_start_date,
    start_date,
    sim_start_date,
    use_TP_adjustment,
    n_days_nowcast_TP_adjustment,
    mob_samples,
    p_detect_omicron,   # only used for naming 
)
from scenarios import scenarios, scenario_dates
from sys import argv
from datetime import timedelta, datetime
from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

try:
    # the file date 
    data_date = pd.to_datetime(argv[1])
except ValueError:
    print("Need to pass more inputs.")

# Define inputs
sim_start_date = pd.to_datetime(sim_start_date)

# Add 3 days buffer to mobility forecast
num_forecast_days = num_forecast_days + 3
# data_date = pd.to_datetime('2022-01-25')

print("============")
print("Generating TP forecasts using data from", data_date)
print("============")

# convert third start date to the correct format
third_start_date = pd.to_datetime(third_start_date)
third_end_date = data_date - timedelta(truncation_days)

# a different end date to deal with issues in fitting
third_end_date_diff = data_date - timedelta(18 + 7 + 7)

third_states = sorted(["NSW", "VIC", "ACT", "QLD", "SA", "TAS", "NT", "WA"])
# third_states = sorted(['NSW', 'VIC', 'ACT', 'QLD', 'SA', 'NT'])
# choose dates for each state for third wave
# NOTE: These need to be in date sorted order
third_date_range = {
    "ACT": pd.date_range(start="2021-08-15", end=third_end_date).values,
    "NSW": pd.date_range(start=third_start_date, end=third_end_date).values,
    "NT": pd.date_range(start="2021-12-01", end=third_end_date).values,
    "QLD": pd.date_range(start="2021-07-30", end=third_end_date).values,
    "SA": pd.date_range(start="2021-11-25", end=third_end_date).values,
    "TAS": pd.date_range(start="2021-12-20", end=third_end_date).values,
    "VIC": pd.date_range(start="2021-08-01", end=third_end_date).values,
    "WA": pd.date_range(start="2022-01-01", end=third_end_date).values,
}

# Get Google Data - Don't use the smoothed data?
df_google_all = read_in_google(Aus_only=True, moving=True, local=True)
third_end_date = pd.to_datetime(data_date) - pd.Timedelta(days=truncation_days)

results_dir = (
    "results/"
    + data_date.strftime("%Y-%m-%d")
    + "/"
)
# Load in vaccination data by state and date which should have the same date as the 
# NNDSS/linelist data use the inferred VE
vaccination_by_state_delta = pd.read_csv(
    results_dir + "adjusted_vaccine_ts_delta" + data_date.strftime("%Y-%m-%d") + ".csv",
    parse_dates=["date"],
)
vaccination_by_state_delta = vaccination_by_state_delta[["state", "date", "effect"]]
vaccination_by_state_delta = vaccination_by_state_delta.pivot(
    index="state", columns="date", values="effect"
)  # Convert to matrix form
# Convert to simple array for indexing
vaccination_by_state_delta_array = vaccination_by_state_delta.to_numpy()

vaccination_by_state_omicron = pd.read_csv(
    results_dir + "adjusted_vaccine_ts_omicron" + data_date.strftime("%Y-%m-%d") + ".csv",
    parse_dates=["date"],
)
vaccination_by_state_omicron = vaccination_by_state_omicron[["state", "date", "effect"]]
vaccination_by_state_omicron = vaccination_by_state_omicron.pivot(
    index="state", columns="date", values="effect"
)  # Convert to matrix form
# Convert to simple array for indexing
vaccination_by_state_omicron_array = vaccination_by_state_omicron.to_numpy()

# Get survey data
surveys = pd.DataFrame()
path = "data/md/Barometer wave*.csv"
for file in glob.glob(path):
    surveys = surveys.append(pd.read_csv(file, parse_dates=["date"]))

surveys = surveys.sort_values(by="date")
print("Latest microdistancing survey is {}".format(surveys.date.values[-1]))

surveys.loc[surveys.state != "ACT", "state"] = (
    surveys.loc[surveys.state != "ACT", "state"]
    .map(states_initials)
    .fillna(surveys.loc[surveys.state != "ACT", "state"])
)
surveys["proportion"] = surveys["count"] / surveys.respondents
surveys.date = pd.to_datetime(surveys.date)

always = surveys.loc[surveys.response == "Always"].set_index(["state", "date"])
always = always.unstack(["state"])

# fill in date range
idx = pd.date_range("2020-03-01", pd.to_datetime("today"))
always = always.reindex(idx, fill_value=np.nan)
always.index.name = "date"
always = always.fillna(method="bfill")
always = always.stack(["state"])

# Zero out before first survey 20th March
always = always.reset_index().set_index("date")
always.loc[:"2020-03-20", "count"] = 0
always.loc[:"2020-03-20", "respondents"] = 0
always.loc[:"2020-03-20", "proportion"] = 0

always = always.reset_index().set_index(["state", "date"])

survey_X = pd.pivot_table(
    data=always, index="date", columns="state", values="proportion"
)
prop_all = survey_X

## read in and process mask wearing data
mask_wearing = pd.DataFrame()
path = "data/face_coverings/face_covering_*_.csv"
for file in glob.glob(path):
    mask_wearing = mask_wearing.append(pd.read_csv(file, parse_dates=["date"]))

mask_wearing = mask_wearing.sort_values(by="date")
print("Latest mask wearing survey is {}".format(mask_wearing.date.values[-1]))

# mask_wearing['state'] = mask_wearing['state'].map(states_initials).fillna(mask_wearing['state'])
mask_wearing.loc[mask_wearing.state != "ACT", "state"] = (
    mask_wearing.loc[mask_wearing.state != "ACT", "state"]
    .map(states_initials)
    .fillna(mask_wearing.loc[mask_wearing.state != "ACT", "state"])
)
mask_wearing["proportion"] = mask_wearing["count"] / mask_wearing.respondents
mask_wearing.date = pd.to_datetime(mask_wearing.date)

mask_wearing_always = mask_wearing.loc[
    mask_wearing.face_covering == "Always"
].set_index(["state", "date"])
mask_wearing_always = mask_wearing_always.unstack(["state"])

idx = pd.date_range("2020-03-01", pd.to_datetime("today"))
mask_wearing_always = mask_wearing_always.reindex(idx, fill_value=np.nan)
mask_wearing_always.index.name = "date"
# fill back to earlier and between weeks.
# Assume survey on day x applies for all days up to x - 6
mask_wearing_always = mask_wearing_always.fillna(method="bfill")
mask_wearing_always = mask_wearing_always.stack(["state"])

# Zero out before first survey 20th March
mask_wearing_always = mask_wearing_always.reset_index().set_index("date")
mask_wearing_always.loc[:"2020-03-20", "count"] = 0
mask_wearing_always.loc[:"2020-03-20", "respondents"] = 0
mask_wearing_always.loc[:"2020-03-20", "proportion"] = 0

mask_wearing_X = pd.pivot_table(
    data=mask_wearing_always, index="date", columns="state", values="proportion"
)
mask_wearing_all = mask_wearing_X

# Get posterior
df_samples = read_in_posterior(
    date=data_date.strftime("%Y-%m-%d"),
)

states = sorted(["NSW", "QLD", "SA", "VIC", "TAS", "WA", "ACT", "NT"])
plot_states = states.copy()

one_month = data_date + timedelta(days=num_forecast_days)
days_from_March = (one_month - pd.to_datetime(start_date)).days

# filter out future info
prop = prop_all.loc[:data_date]
masks = mask_wearing_all.loc[:data_date]
df_google = df_google_all.loc[df_google_all.date <= data_date]

# use this trick of saving the google data and then reloading it to kill
# the date time values
df_google.to_csv("results/test_google_data.csv")
df_google = pd.read_csv("results/test_google_data.csv")
# remove the temporary file
# os.remove("results/test_google_data.csv")

# Simple interpolation for missing vlaues in Google data
df_google = df_google.interpolate(method="linear", axis=0)
df_google.date = pd.to_datetime(df_google.date)

# forecast time parameters
today = data_date.strftime("%Y-%m-%d")

# add days to forecast if we are missing data
if df_google.date.values[-1] < data_date:
    n_forecast = num_forecast_days + (data_date - df_google.date.values[-1]).days
else:
    n_forecast = num_forecast_days

training_start_date = datetime(2020, 3, 1, 0, 0)
print(
    "Forecast ends at {} days after 1st March".format(
        (pd.to_datetime(today) - pd.to_datetime(training_start_date)).days
        + num_forecast_days
    )
)
print(
    "Final date is {}".format(pd.to_datetime(today) + timedelta(days=num_forecast_days))
)
df_google = df_google.loc[df_google.date >= training_start_date]
outdata = {"date": [], "type": [], "state": [], "mean": [], "std": []}
predictors = mov_values.copy()
predictors.remove("residential_7days")

# Setup Figures
axes = []
figs = []
for var in predictors:
    fig, ax_states = plt.subplots(figsize=(7, 8), nrows=4, ncols=2, sharex=True)
    axes.append(ax_states)
    # fig.suptitle(var)
    figs.append(fig)

# extra fig for microdistancing
var = "Proportion people always microdistancing"
fig, ax_states = plt.subplots(figsize=(7, 8), nrows=4, ncols=2, sharex=True)
axes.append(ax_states)
figs.append(fig)

# # extra fig for mask wearing
var = "Proportion people always wearing masks"
fig, ax_states = plt.subplots(figsize=(7, 8), nrows=4, ncols=2, sharex=True)
axes.append(ax_states)
figs.append(fig)

var = "Reduction in Reff due to vaccination"
fig, ax_states = plt.subplots(figsize=(7, 8), nrows=4, ncols=2, sharex=True)
axes.append(ax_states)
figs.append(fig)

var = "Reduction in Reff due to vaccination"
fig, ax_states = plt.subplots(figsize=(7, 8), nrows=4, ncols=2, sharex=True)
axes.append(ax_states)
figs.append(fig)

# Forecasting Params
n_training = 21  # Period to examine trend
n_baseline = 150  # Period to create baseline
n_training_vaccination = 30  # period to create trend for vaccination

# since this can be useful, predictor ordering is:
# [
#     'retail_and_recreation_7days', 
#     'grocery_and_pharmacy_7days', 
#     'parks_7days', 
#     'transit_stations_7days', 
#     'workplaces_7days'
# ]

# Loop through states and run forecasting.
print("============")
print("Forecasting macro, micro and vaccination")
print("============")

state_Rmed = {}
state_sims = {}
for i, state in enumerate(states):

    rownum = int(i / 2)
    colnum = np.mod(i, 2)

    rows = df_google.loc[df_google.state == state].shape[0]
    # Rmed currently a list, needs to be a matrix
    Rmed_array = np.zeros(shape=(rows, len(predictors), mob_samples))
    Rmed_array_inflated = np.zeros(shape=(rows, len(predictors), mob_samples))
    for j, var in enumerate(predictors):
        for n in range(mob_samples):
            # historically we want a little more noise. In the actual forecasting of trends
            # we don't want this to be quite that prominent.
            Rmed_array[:, j, n] = df_google[df_google["state"] == state][
                var
            ].values.T + np.random.normal(
                loc=0, scale=df_google[df_google["state"] == state][var + "_std"]
            )

    dates = df_google[df_google["state"] == state]["date"]

    # cap min and max at historical or (-50,0)
    # 1 by predictors by mob_samples size
    minRmed_array = np.minimum(-50, np.amin(Rmed_array, axis=0))
    maxRmed_array = np.maximum(10, np.amax(Rmed_array, axis=0))

    # days by predictors by samples
    sims = np.zeros(shape=(n_forecast, len(predictors), mob_samples))
    for n in range(mob_samples):  # Loop through simulations
        Rmed = Rmed_array[:, :, n]
        minRmed = minRmed_array[:, n]
        maxRmed = maxRmed_array[:, n]
        if maxRmed[1] < 20:
            maxRmed[1] = 50

        R_baseline_mean = np.mean(Rmed[-n_baseline:, :], axis=0)
        if state not in {"WA"}:
            R_baseline_mean[-1] = 0
        R_diffs = np.diff(Rmed[-n_training:, :], axis=0)
        mu = np.mean(R_diffs, axis=0)
        cov = np.cov(R_diffs, rowvar=False)  # columns are vars, rows are obs

        # Forecast mobility forward sequentially by day.
        current = np.mean(Rmed[-9:-2, :], axis=0)  # Start from last valid days
        for i in range(n_forecast):
            # ## SCENARIO MODELLING
            # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
            if scenarios[state] == "":
                # Proportion of trend_force to regression_to_baseline_force
                p_force = (n_forecast - i) / (n_forecast)

                # Generate a single forward realisation of trend
                trend_force = np.random.multivariate_normal(mu, cov)
                # Generate a single forward realisation of baseline regression
                # regression to baseline force stronger in standard forecasting
                regression_to_baseline_force = np.random.multivariate_normal(
                    0.05 * (R_baseline_mean - current), cov
                )

                new_forcast_points = (
                    current + p_force * trend_force + (1 - p_force) * regression_to_baseline_force
                )  # Find overall simulation step
                # Apply minimum and maximum
                new_forcast_points = np.maximum(minRmed, new_forcast_points)
                new_forcast_points = np.minimum(maxRmed, new_forcast_points)

                current = new_forcast_points

            elif scenarios[state] != "":
                # Make baseline cov for generating points
                cov_baseline = np.cov(Rmed[-42:-28, :], rowvar=False)
                mu_current = Rmed[-1, :]
                mu_victoria = np.array(
                    [
                        -55.35057887,
                        -22.80891056,
                        -46.59531636,
                        -75.99942378,
                        -44.71119293,
                    ]
                )

                mu_baseline = np.mean(Rmed[-42:-28, :], axis=0)
                # mu_baseline = 0*np.mean(Rmed[-42:-28, :], axis=0)

                if scenario_dates[state] != "":
                    scenario_change_point = (
                        pd.to_datetime(scenario_dates[state]) - data_date
                    ).days + (n_forecast - 42)

                # Constant Lockdown
                if (
                    scenarios[state] == "no_reversion"
                    or scenarios[state] == "school_opening_2022"
                ):
                    # take a continuous median to account for noise in recent observations (such as sunny days)
                    # mu_current = np.mean(Rmed[-7:, :], axis=0)
                    # cov_baseline = np.cov(Rmed[-28:, :], rowvar=False)
                    new_forcast_points = np.random.multivariate_normal(
                        mu_current, cov_baseline
                    )

                elif scenarios[state] == "no_reversion_continuous_lockdown":
                    # add the new scenario here
                    new_forcast_points = np.random.multivariate_normal(
                        mu_current, cov_baseline
                    )

                # No Lockdown
                elif scenarios[state] == "full_reversion":
                    # a full reversion scenario changes the social mobility and microdistancing
                    # behaviours at the scenario change date and then applies a return to baseline force
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(
                            mu_current, cov_baseline
                        )
                    else:
                        # baseline is within lockdown period so take a new baseline of 0's and trend towards this
                        R_baseline_0 = np.zeros_like(R_baseline_mean)
                        R_baseline_0 = mu_baseline
                        # set adjusted baselines by eyeline for now, need to get this automated
                        # R_baseline_0[1] = 10    # baseline of +10% for Grocery based on other jurisdictions

                        # # apply specific baselines to the jurisdictions progressing towards normal restrictions
                        # if state == 'NSW':
                        #     R_baseline_0[3] = -25   # baseline of -25% for Transit based on 2021-April to 2021-July (pre-third-wave lockdowns)
                        # elif state == 'ACT':
                        #     R_baseline_0[1] = 20    # baseline of +20% for Grocery based on other jurisdictions
                        #     R_baseline_0[3] = -25   # baseline of -25% for Transit based on 2021-April to 2021-July (pre-third-wave lockdowns)
                        # elif state == 'VIC':
                        #     R_baseline_0[0] = -15   # baseline of -15% for R&R based on 2021-April to 2021-July (pre-third-wave lockdowns)
                        #     R_baseline_0[3] = -30   # baseline of -30% for Transit based on 2021-April to 2021-July (pre-third-wave lockdowns)
                        #     R_baseline_0[4] = -15   # baseline of -15% for workplaces based on 2021-April to 2021-July (pre-third-wave lockdowns)

                        # the force we trend towards the baseline above with
                        p_force = (n_forecast - i) / (n_forecast)
                        trend_force = np.random.multivariate_normal(
                            mu, cov
                        )  # Generate a single forward realisation of trend
                        # baseline scalar is smaller for this as we want slow returns
                        adjusted_baseline_drift_mean = R_baseline_0 - current
                        # we purposely scale the transit measure so that we increase a little more quickly
                        # tmp = 0.05 * adjusted_baseline_drift_mean[3]
                        adjusted_baseline_drift_mean *= 0.005
                        # adjusted_baseline_drift_mean[3] = tmp
                        regression_to_baseline_force = np.random.multivariate_normal(
                            adjusted_baseline_drift_mean, cov
                        )  # Generate a single forward realisation of baseline regression
                        new_forcast_points = (
                            current 
                            + p_force * trend_force 
                            + (1 - p_force) * regression_to_baseline_force
                        )  # Find overall simulation step
                        # new_forcast_points = current + regression_to_baseline_force # Find overall simulation step
                        # Apply minimum and maximum
                        new_forcast_points = np.maximum(minRmed, new_forcast_points)
                        new_forcast_points = np.minimum(maxRmed, new_forcast_points)
                        current = new_forcast_points

                elif scenarios[state] == "immediately_baseline":
                    # this scenario is used to return instantly to the baseline levels
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(
                            mu_current, cov_baseline
                        )
                    else:
                        # baseline is within lockdown period so take a new baseline of 0's and trend towards this
                        R_baseline_0 = np.zeros_like(R_baseline_mean)
                        # jump immediately to baseline
                        new_forcast_points = np.random.multivariate_normal(
                            R_baseline_0, cov_baseline
                        )

                # Temporary Lockdown
                elif scenarios[state] == "half_reversion":
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(
                            mu_current, cov_baseline
                        )
                    else:
                        new_forcast_points = np.random.multivariate_normal(
                            (mu_current + mu_baseline) / 2, cov_baseline
                        )

                # Stage 4
                elif scenarios[state] == "stage4":
                    if i < scenario_change_point:
                        new_forcast_points = np.random.multivariate_normal(
                            mu_current, cov_baseline
                        )
                    else:
                        new_forcast_points = np.random.multivariate_normal(
                            mu_victoria, cov_baseline
                        )

            # Set this day in this simulation to the forecast realisation
            sims[i, :, n] = new_forcast_points

    dd = [dates.tolist()[-1] + timedelta(days=x) for x in range(1, n_forecast + 1)]

    sims_med = np.median(sims, axis=2)  # N by predictors
    sims_q25 = np.percentile(sims, 25, axis=2)
    sims_q75 = np.percentile(sims, 75, axis=2)

    # forecast mircodistancing
    # Get a baseline value of microdistancing
    mu_overall = np.mean(prop[state].values[-n_baseline:])
    md_diffs = np.diff(prop[state].values[-n_training:])
    mu_diffs = np.mean(md_diffs) 
    std_diffs = np.std(md_diffs)

    extra_days_md = (
        pd.to_datetime(df_google.date.values[-1])
        - pd.to_datetime(prop[state].index.values[-1])
    ).days

    # Set all values to current value.
    current = [prop[state].values[-1]] * mob_samples
    new_md_forecast = []
    # Forecast mobility forward sequentially by day.
    for i in range(n_forecast + extra_days_md):

        # SCENARIO MODELLING
        # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
        if scenarios[state] == "":
            # Proportion of trend_force to regression_to_baseline_force
            p_force = (n_forecast + extra_days_md - i) / (n_forecast + extra_days_md)
            # Generate step realisations in training trend direction
            trend_force = np.random.normal(mu_diffs, std_diffs, size=mob_samples)
            # Generate realisations that draw closer to baseline
            regression_to_baseline_force = np.random.normal(
                0.05 * (mu_overall - current), std_diffs
            )
            current = (
                current
                + p_force * trend_force
                + (1 - p_force) * regression_to_baseline_force
            )  # Balance forces
            # current = current+p_force*trend_force  # Balance forces

        elif scenarios[state] != "":
            current = np.array(current)

            # Make baseline cov for generating points
            std_baseline = np.std(prop[state].values[-42:-28])
            mu_baseline = np.mean(prop[state].values[-42:-28], axis=0)
            mu_current = prop[state].values[-1]

            if scenario_dates[state] != "":
                scenario_change_point = (
                    pd.to_datetime(scenario_dates[state]) - data_date
                ).days + extra_days_md

            # Constant Lockdown
            if (
                scenarios[state] == "no_reversion"
                or scenarios[state] == "school_opening_2022"
            ):
                # use only more recent data to forecast under a no-reversion scenario
                # std_lockdown = np.std(prop[state].values[-24:-4])
                # current = np.random.normal(mu_current, std_lockdown)
                current = np.random.normal(mu_current, std_baseline)

            # No Lockdown
            elif scenarios[state] == "full_reversion":
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline)
                else:
                    mu_baseline_0 = 0.2
                    # Proportion of trend_force to regression_to_baseline_force
                    p_force = (n_forecast + extra_days_md - i) / (
                        n_forecast + extra_days_md
                    )
                    # take a mean of the differences over the last 2 weeks
                    mu_diffs = np.mean(np.diff(prop[state].values[-14:]))
                    # Generate step realisations in training trend direction
                    trend_force = np.random.normal(mu_diffs, std_baseline)
                    # Generate realisations that draw closer to baseline
                    regression_to_baseline_force = np.random.normal(
                        0.005 * (mu_baseline_0 - current), std_baseline
                    )
                    current = current + regression_to_baseline_force  # Balance forces

            elif scenarios[state] == "immediately_baseline":
                # this scenario is an immediate return to baseline values
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline)
                else:
                    mu_baseline_0 = 0.2
                    # jump immediately to baseline
                    current = np.random.normal(mu_baseline_0, std_baseline)

            # Temporary Lockdown
            elif scenarios[state] == "half_reversion":  # No Lockdown
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline)
                else:
                    # Revert to values halfway between the before and after
                    current = np.random.normal(
                        (mu_current + mu_baseline) / 2, std_baseline
                    )

        new_md_forecast.append(current)

    md_sims = np.vstack(new_md_forecast)  # Put forecast days together
    md_sims = np.minimum(1, md_sims)
    md_sims = np.maximum(0, md_sims)

    dd_md = [
        prop[state].index[-1] + timedelta(days=x)
        for x in range(1, n_forecast + extra_days_md + 1)
    ]

    ## currently not forecasting masks — may return in the future but will need to assess.

    # forecast mask wearing compliance
    # Get a baseline value of microdistancing
    mu_overall = np.mean(masks[state].values[-n_baseline:])
    md_diffs = np.diff(masks[state].values[-n_training:])
    mu_diffs = np.mean(md_diffs)
    std_diffs = np.std(md_diffs)

    extra_days_masks = (
        pd.to_datetime(df_google.date.values[-1])
        - pd.to_datetime(masks[state].index.values[-1])
    ).days

    # Set all values to current value.
    current = [masks[state].values[-1]] * mob_samples
    new_masks_forecast = []
    # Forecast mobility forward sequentially by day.
    for i in range(n_forecast + extra_days_masks):

        # SCENARIO MODELLING
        # This code chunk will allow you manually set the distancing params for a state to allow for modelling.
        if scenarios[state] == "":
            # masksortion of trend_force to regression_to_baseline_force
            p_force = (n_forecast + extra_days_masks - i) / (
                n_forecast + extra_days_masks
            )
            # Generate step realisations in training trend direction
            trend_force = np.random.normal(mu_diffs, std_diffs, size=mob_samples)
            # Generate realisations that draw closer to baseline
            # regression_to_baseline_force = np.random.normal(0.05*(mu_overall - current), std_diffs)
            # current = current + p_force*trend_force + (1-p_force)*regression_to_baseline_force  # Balance forces
            current = current + trend_force

        elif scenarios[state] != "":
            current = np.array(current)

            # Make baseline cov for generating points
            std_baseline = np.std(masks[state].values[-42:-28])
            mu_baseline = np.mean(masks[state].values[-42:-28], axis=0)
            mu_current = masks[state].values[-1]

            if scenario_dates[state] != "":
                scenario_change_point = (
                    pd.to_datetime(scenario_dates[state]) - data_date
                ).days + extra_days_masks

            # Constant Lockdown
            if (
                scenarios[state] == "no_reversion"
                or scenarios[state] == "school_opening_2022"
            ):
                # use only more recent data to forecast under a no-reversion scenario
                # std_lockdown = np.std(masks[state].values[-24:-4])
                # current = np.random.normal(mu_current, std_lockdown)
                current = np.random.normal(mu_current, std_baseline)

            # No Lockdown
            elif scenarios[state] == "full_reversion":
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline)
                else:
                    mu_baseline_0 = 0.2
                    # masksortion of trend_force to regression_to_baseline_force
                    p_force = (n_forecast + extra_days_masks - i) / (
                        n_forecast + extra_days_masks
                    )
                    # take a mean of the differences over the last 2 weeks
                    mu_diffs = np.mean(np.diff(masks[state].values[-14:]))
                    # Generate step realisations in training trend direction
                    trend_force = np.random.normal(mu_diffs, std_baseline)
                    # Generate realisations that draw closer to baseline
                    regression_to_baseline_force = np.random.normal(
                        0.005 * (mu_baseline_0 - current), std_baseline
                    )
                    current = current + regression_to_baseline_force  # Balance forces

            elif scenarios[state] == "immediately_baseline":
                # this scenario is an immediate return to baseline values
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline)
                else:
                    mu_baseline_0 = 0.2
                    # jump immediately to baseline
                    current = np.random.normal(mu_baseline_0, std_baseline)

            # Temporary Lockdown
            elif scenarios[state] == "half_reversion":  # No Lockdown
                if i < scenario_change_point:
                    current = np.random.normal(mu_current, std_baseline)
                else:
                    # Revert to values halfway between the before and after
                    current = np.random.normal(
                        (mu_current + mu_baseline) / 2, std_baseline
                    )

        new_masks_forecast.append(current)

    masks_sims = np.vstack(new_masks_forecast)  # Put forecast days together
    masks_sims = np.minimum(1, masks_sims)
    masks_sims = np.maximum(0, masks_sims)

    dd_masks = [
        masks[state].index[-1] + timedelta(days=x)
        for x in range(1, n_forecast + extra_days_masks + 1)
    ]

    # Forecasting vaccine effect
    # if state == "WA":
    #     last_fit_date = pd.to_datetime(third_end_date)
    # else:
    last_fit_date = pd.to_datetime(third_date_range[state][-1])

    extra_days_vacc = (pd.to_datetime(df_google.date.values[-1]) - last_fit_date).days
    total_forecasting_days = n_forecast + extra_days_vacc
    # get the VE on the last day
    mean_delta = vaccination_by_state_delta.loc[state][last_fit_date + timedelta(1)]
    mean_omicron = vaccination_by_state_omicron.loc[state][last_fit_date + timedelta(1)]

    current = np.zeros_like(mob_samples)

    new_delta = []
    new_omicron = []
    # variance on the vaccine forecasts is equivalent to what we use in the fitting 
    var_vax = 0.00005
    a_vax = np.zeros_like(mob_samples)
    b_vax = np.zeros_like(mob_samples)

    for d in pd.date_range(
        last_fit_date + timedelta(1),
        pd.to_datetime(today) + timedelta(days=num_forecast_days),
    ):

        mean_delta = vaccination_by_state_delta.loc[state][d]
        a_vax = mean_delta * (mean_delta * (1 - mean_delta) / var_vax - 1)
        b_vax = (1 - mean_delta) * (mean_delta * (1 - mean_delta) / var_vax - 1)
        current = np.random.beta(a_vax, b_vax, mob_samples)
        new_delta.append(current.tolist())

        mean_omicron = vaccination_by_state_omicron.loc[state][d]
        a_vax = mean_omicron * (mean_omicron * (1 - mean_omicron) / var_vax - 1)
        b_vax = (1 - mean_omicron) * (mean_omicron * (1 - mean_omicron) / var_vax - 1)
        current = np.random.beta(a_vax, b_vax, mob_samples)
        new_omicron.append(current.tolist())

    vacc_sims_delta = np.vstack(new_delta)
    vacc_sims_omicron = np.vstack(new_omicron)

    dd_vacc = [
        last_fit_date + timedelta(days=x)
        for x in range(1, n_forecast + extra_days_vacc + 1)
    ]

    for j, var in enumerate(
        predictors
        + ["md_prop"]
        + ["masks_prop"]
        + ["vaccination_delta"]
        + ["vaccination_omicron"]
    ):
        # Record data
        axs = axes[j]
        if (state == "AUS") and (var == "md_prop"):
            continue

        if var == "md_prop":
            outdata["type"].extend([var] * len(dd_md))
            outdata["state"].extend([state] * len(dd_md))
            outdata["date"].extend([d.strftime("%Y-%m-%d") for d in dd_md])
            outdata["mean"].extend(np.mean(md_sims, axis=1))
            outdata["std"].extend(np.std(md_sims, axis=1))

        elif var == "masks_prop":
            outdata["type"].extend([var] * len(dd_masks))
            outdata["state"].extend([state] * len(dd_masks))
            outdata["date"].extend([d.strftime("%Y-%m-%d") for d in dd_masks])
            outdata["mean"].extend(np.mean(masks_sims, axis=1))
            outdata["std"].extend(np.std(masks_sims, axis=1))

        elif var == "vaccination_delta":
            outdata["type"].extend([var] * len(dd_vacc))
            outdata["state"].extend([state] * len(dd_vacc))
            outdata["date"].extend([d.strftime("%Y-%m-%d") for d in dd_vacc])
            outdata["mean"].extend(np.mean(vacc_sims_delta, axis=1))
            outdata["std"].extend(np.std(vacc_sims_delta, axis=1))

        elif var == "vaccination_omicron":
            outdata["type"].extend([var] * len(dd_vacc))
            outdata["state"].extend([state] * len(dd_vacc))
            outdata["date"].extend([d.strftime("%Y-%m-%d") for d in dd_vacc])
            outdata["mean"].extend(np.mean(vacc_sims_omicron, axis=1))
            outdata["std"].extend(np.std(vacc_sims_omicron, axis=1))

        else:
            outdata["type"].extend([var] * len(dd))
            outdata["state"].extend([state] * len(dd))
            outdata["date"].extend([d.strftime("%Y-%m-%d") for d in dd])
            outdata["mean"].extend(np.mean(sims[:, j, :], axis=1))
            outdata["std"].extend(np.std(sims[:, j, :], axis=1))

        if state in plot_states:

            if var == "md_prop":
                # md plot
                axs[rownum, colnum].plot(prop[state].index, prop[state].values, lw=1)
                axs[rownum, colnum].plot(dd_md, np.median(md_sims, axis=1), "k", lw=1)
                axs[rownum, colnum].fill_between(
                    dd_md,
                    np.quantile(md_sims, 0.25, axis=1),
                    np.quantile(md_sims, 0.75, axis=1),
                    color="k",
                    alpha=0.1,
                )

            elif var == "masks_prop":
                # masks plot
                axs[rownum, colnum].plot(masks[state].index, masks[state].values, lw=1)
                axs[rownum, colnum].plot(
                    dd_masks, np.median(masks_sims, axis=1), "k", lw=1
                )
                axs[rownum, colnum].fill_between(
                    dd_masks,
                    np.quantile(masks_sims, 0.25, axis=1),
                    np.quantile(masks_sims, 0.75, axis=1),
                    color="k",
                    alpha=0.1,
                )

            elif var == "vaccination_delta":
                # vaccination plot
                axs[rownum, colnum].plot(
                    vaccination_by_state_delta.loc[
                        state, ~vaccination_by_state_delta.loc[state].isna()
                    ].index,
                    vaccination_by_state_delta.loc[
                        state, ~vaccination_by_state_delta.loc[state].isna()
                    ].values,
                    lw=1,
                )
                axs[rownum, colnum].plot(
                    dd_vacc, np.median(vacc_sims_delta, axis=1), color="C1", lw=1
                )
                axs[rownum, colnum].fill_between(
                    dd_vacc,
                    np.quantile(vacc_sims_delta, 0.25, axis=1),
                    np.quantile(vacc_sims_delta, 0.75, axis=1),
                    color="C1",
                    alpha=0.1,
                )

            elif var == "vaccination_omicron":
                # vaccination plot
                axs[rownum, colnum].plot(
                    vaccination_by_state_omicron.loc[
                        state, ~vaccination_by_state_omicron.loc[state].isna()
                    ].index,
                    vaccination_by_state_omicron.loc[
                        state, ~vaccination_by_state_omicron.loc[state].isna()
                    ].values,
                    lw=1,
                )
                axs[rownum, colnum].plot(
                    dd_vacc, np.median(vacc_sims_omicron, axis=1), color="C1", lw=1
                )
                axs[rownum, colnum].fill_between(
                    dd_vacc,
                    np.quantile(vacc_sims_omicron, 0.25, axis=1),
                    np.quantile(vacc_sims_omicron, 0.75, axis=1),
                    color="C1",
                    alpha=0.1,
                )

            else:
                # all other predictors
                axs[rownum, colnum].plot(
                    dates, df_google[df_google["state"] == state][var].values, lw=1
                )
                axs[rownum, colnum].fill_between(
                    dates,
                    np.percentile(Rmed_array[:, j, :], 25, axis=1),
                    np.percentile(Rmed_array[:, j, :], 75, axis=1),
                    alpha=0.5,
                )

                axs[rownum, colnum].plot(dd, sims_med[:, j], color="C1", lw=1)
                axs[rownum, colnum].fill_between(
                    dd, sims_q25[:, j], sims_q75[:, j], color="C1", alpha=0.1
                )

            # axs[rownum,colnum].axvline(dd[-num_forecast_days], ls = '--', color = 'black', lw=1)            # plotting a vertical line at the end of the data date
            # axs[rownum,colnum].axvline(dd[-(num_forecast_days+truncation_days)], ls = '-.', color='grey', lw=1)            # plotting a vertical line at the forecast date

            axs[rownum, colnum].set_title(state)
            # plotting horizontal line at 1
            axs[rownum, colnum].axhline(1, ls="--", c="k", lw=1)

            axs[rownum, colnum].set_title(state)
            axs[rownum, colnum].tick_params("x", rotation=90)
            axs[rownum, colnum].tick_params("both", labelsize=8)

            # plot the start date of the data and indicators of the data we are actually fitting to (in grey)
            axs[rownum, colnum].axvline(data_date, ls="-.", color="black", lw=1)

            if j < len(predictors):
                axs[rownum, colnum].set_ylabel(
                    predictors[j].replace("_", " ")[:-5], fontsize=7
                )
            elif var == "md_prop":
                axs[rownum, colnum].set_ylabel(
                    "Proportion of respondents\n micro-distancing", fontsize=7
                )
            elif var == "masks_prop":
                axs[rownum, colnum].set_ylabel(
                    "Proportion of respondents\n wearing masks", fontsize=7
                )
            elif var == "vaccination_delta" or var == "vaccination_omicron":
                axs[rownum, colnum].set_ylabel(
                    "Reduction in TP \n from vaccination", fontsize=7
                )

    # historically we want to store the higher variance mobilities
    state_Rmed[state] = Rmed_array
    state_sims[state] = sims

os.makedirs(
    "figs/mobility_forecasts/" 
    + data_date.strftime("%Y-%m-%d") 
    + "_mobility_forecasts", 
    exist_ok=True,
)

for i, fig in enumerate(figs):
    fig.text(0.5, 0.02, "Date", ha="center", va="center", fontsize=15)

    if i < len(predictors):  # this plots the google mobility forecasts
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"
            + data_date.strftime("%Y-%m-%d")
            + "_mobility_forecasts/"
            + str(predictors[i])
            + ".png",
            dpi=400,
        )

    elif i == len(predictors):  # this plots the microdistancing forecasts
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"
            + data_date.strftime("%Y-%m-%d")
            + "_mobility_forecasts/micro_dist.png",
            dpi=400,
        )

    elif i == len(predictors) + 1:  # this plots the microdistancing forecasts
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"
            + data_date.strftime("%Y-%m-%d")
            + "_mobility_forecasts/mask_wearing.png",
            dpi=400,
        )

    elif i == len(predictors) + 2:  # finally this plots the delta VE forecasts
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"
            + data_date.strftime("%Y-%m-%d")
            + "_mobility_forecasts/delta_vaccination.png",
            dpi=400,
        )
    else:  # finally this plots the omicron VE forecasts
        fig.tight_layout()
        fig.savefig(
            "figs/mobility_forecasts/"
            + data_date.strftime("%Y-%m-%d")
            + "_mobility_forecasts/omicron_vaccination.png",
            dpi=400,
        )

df_out = pd.DataFrame.from_dict(outdata)

df_md = df_out.loc[df_out.type == "md_prop"]
df_masks = df_out.loc[df_out.type == "masks_prop"]

df_out = df_out.loc[df_out.type != "vaccination_delta"]
df_out = df_out.loc[df_out.type != "vaccination_omicron"]
df_out = df_out.loc[df_out.type != "md_prop"]
df_out = df_out.loc[df_out.type != "masks_prop"]

df_forecast = pd.pivot_table(
    df_out, columns=["type"], index=["date", "state"], values=["mean"]
)
df_std = pd.pivot_table(
    df_out, columns=["type"], index=["date", "state"], values=["std"]
)
df_forecast_md = pd.pivot_table(
    df_md, columns=["state"], index=["date"], values=["mean"]
)
df_forecast_md_std = pd.pivot_table(
    df_md, columns=["state"], index=["date"], values=["std"]
)
df_forecast_masks = pd.pivot_table(
    df_masks, columns=["state"], index=["date"], values=["mean"]
)
df_forecast_masks_std = pd.pivot_table(
    df_masks, columns=["state"], index=["date"], values=["std"]
)
# align with google order in columns
df_forecast = df_forecast.reindex([("mean", val) for val in predictors], axis=1)
df_std = df_std.reindex([("std", val) for val in predictors], axis=1)
df_forecast.columns = predictors  # remove the tuple name of columns
df_std.columns = predictors
df_forecast = df_forecast.reset_index()
df_std = df_std.reset_index()
df_forecast.date = pd.to_datetime(df_forecast.date)
df_std.date = pd.to_datetime(df_std.date)

df_forecast_md = df_forecast_md.reindex([("mean", state) for state in states], axis=1)
df_forecast_md_std = df_forecast_md_std.reindex(
    [("std", state) for state in states], axis=1
)
df_forecast_md.columns = states
df_forecast_md_std.columns = states
df_forecast_md = df_forecast_md.reset_index()
df_forecast_md_std = df_forecast_md_std.reset_index()
df_forecast_md.date = pd.to_datetime(df_forecast_md.date)
df_forecast_md_std.date = pd.to_datetime(df_forecast_md_std.date)

df_forecast_masks = df_forecast_masks.reindex(
    [("mean", state) for state in states], axis=1
)
df_forecast_masks_std = df_forecast_masks_std.reindex(
    [("std", state) for state in states], axis=1
)
df_forecast_masks.columns = states
df_forecast_masks_std.columns = states
df_forecast_masks = df_forecast_masks.reset_index()
df_forecast_masks_std = df_forecast_masks_std.reset_index()
df_forecast_masks.date = pd.to_datetime(df_forecast_masks.date)
df_forecast_masks_std.date = pd.to_datetime(df_forecast_masks_std.date)

df_R = df_google[["date", "state"] + mov_values + [val + "_std" for val in mov_values]]
df_R = pd.concat([df_R, df_forecast], ignore_index=True, sort=False)
df_R["policy"] = (df_R.date >= "2020-03-20").astype("int8")

df_md = pd.concat([prop, df_forecast_md.set_index("date")])
df_masks = pd.concat([masks, df_forecast_masks.set_index("date")])

# now we read in the ve time series and create an adjusted timeseries from March 1st
# that includes no effect prior
vaccination_by_state = pd.read_csv(
    results_dir + "adjusted_vaccine_ts_delta" + data_date.strftime("%Y-%m-%d") + ".csv",
    parse_dates=["date"],
)
# there are a couple NA's early on in the time series but is likely due to slightly different start dates
vaccination_by_state.fillna(1, inplace=True)
vaccination_by_state = vaccination_by_state[["state", "date", "effect"]]
vaccination_by_state = vaccination_by_state.pivot(
    index="state", columns="date", values="effect"
)  # Convert to matrix form

# initialise a complete dataframe which will be the full VE timeseries plus the forecasted VE
df_ve_delta = pd.DataFrame()
# loop over states and get the offset compoonenets of the full VE
before_vacc_dates = pd.date_range(
    start_date, vaccination_by_state.columns[0] - timedelta(days=1), freq="d"
)
# this is just a df of ones with all the missing dates as indices (8 comes from 8 jurisdictions)
before_vacc_Reff_reduction = pd.DataFrame(np.ones(((1, len(before_vacc_dates)))))
before_vacc_Reff_reduction.columns = before_vacc_dates

for state in states:
    before_vacc_Reff_reduction.index = {state}
    # merge the vaccine data and the 1's dataframes
    df_ve_delta[state] = pd.concat(
        [before_vacc_Reff_reduction.loc[state].T, vaccination_by_state.loc[state].T]
    )

# clip off extra days
df_ve_delta = df_ve_delta[
    df_ve_delta.index <= pd.to_datetime(today) + timedelta(days=num_forecast_days)
]

# save the forecasted vaccination line
df_ve_delta.to_csv(
    results_dir
    + "forecasted_vaccination_delta"
    + data_date.strftime("%Y-%m-%d")
    + ".csv"
)

vaccination_by_state = pd.read_csv(
    results_dir 
    + "adjusted_vaccine_ts_omicron" 
    + data_date.strftime("%Y-%m-%d") 
    + ".csv",
    parse_dates=["date"],
)
# there are a couple NA's early on in the time series but is likely due to slightly different start dates
vaccination_by_state.fillna(1, inplace=True)
vaccination_by_state = vaccination_by_state[["state", "date", "effect"]]
vaccination_by_state = vaccination_by_state.pivot(
    index="state", columns="date", values="effect"
)  # Convert to matrix form

# initialise a complete dataframe which will be the full VE timeseries plus the forecasted VE
df_ve_omicron = pd.DataFrame()
# loop over states and get the offset compoonenets of the full VE
before_vacc_dates = pd.date_range(
    start_date, pd.to_datetime(omicron_start_date) - timedelta(days=1), freq="d"
)
# this is just a df of ones with all the missing dates as indices (8 comes from 8 jurisdictions)
before_vacc_Reff_reduction = pd.DataFrame(np.ones(((1, len(before_vacc_dates)))))
before_vacc_Reff_reduction.columns = before_vacc_dates

for state in states:
    before_vacc_Reff_reduction.index = {state}
    # merge the vaccine data and the 1's dataframes
    df_ve_omicron[state] = pd.concat(
        [
            before_vacc_Reff_reduction.loc[state].T,
            vaccination_by_state.loc[state][
                vaccination_by_state.loc[state].index
                >= pd.to_datetime(omicron_start_date)
            ],
        ]
    )

df_ve_omicron = df_ve_omicron[
    df_ve_omicron.index <= pd.to_datetime(today) + timedelta(days=num_forecast_days)
]
# save the forecasted vaccination line
df_ve_omicron.to_csv(
    results_dir
    + "forecasted_vaccination_omicron"
    + data_date.strftime("%Y-%m-%d")
    + ".csv"
)

print("============")
print("Plotting forecasted estimates")
print("============")

expo_decay = True
theta_md = np.tile(df_samples["theta_md"].values, (df_md["NSW"].shape[0], 1))

fig, ax = plt.subplots(figsize=(12, 9), nrows=4, ncols=2, sharex=True, sharey=True)
for i, state in enumerate(plot_states):
    # np.random.normal(df_md[state].values, df_md_std.values)
    prop_sim = df_md[state].values
    if expo_decay:
        md = ((1 + theta_md).T ** (-1 * prop_sim)).T
    else:
        md = 2 * expit(-1 * theta_md * prop_sim[:, np.newaxis])

    row = i // 2
    col = i % 2

    ax[row, col].plot(
        df_md[state].index, np.median(md, axis=1), label="Microdistancing"
    )
    ax[row, col].fill_between(
        df_md[state].index,
        np.quantile(md, 0.25, axis=1),
        np.quantile(md, 0.75, axis=1),
        label="Microdistancing",
        alpha=0.4,
        color="C0",
    )
    ax[row, col].fill_between(
        df_md[state].index,
        np.quantile(md, 0.05, axis=1),
        np.quantile(md, 0.95, axis=1),
        label="Microdistancing",
        alpha=0.4,
        color="C0",
    )
    ax[row, col].set_title(state)
    ax[row, col].tick_params("x", rotation=45)

    ax[row, col].set_xticks(
        [df_md[state].index.values[-n_forecast - extra_days_md]],
        minor=True,
    )
    ax[row, col].xaxis.grid(which="minor", linestyle="-.", color="grey", linewidth=1)

fig.text(
    0.03,
    0.5,
    "Multiplicative effect \n of micro-distancing $M_d$",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=20,
)

fig.text(0.5, 0.04, "Date", ha="center", va="center", fontsize=20)

plt.tight_layout(rect=[0.05, 0.04, 1, 1])

fig.savefig(
    "figs/"
    + "mobility_forecasts/" 
    + data_date.strftime("%Y-%m-%d") 
    + "_mobility_forecasts/md_factor.png",
    dpi=144,
)

theta_masks = np.tile(df_samples["theta_masks"].values, (df_masks["NSW"].shape[0], 1))

fig, ax = plt.subplots(figsize=(12, 9), nrows=4, ncols=2, sharex=True, sharey=True)

for i, state in enumerate(plot_states):
    # np.random.normal(df_md[state].values, df_md_std.values)
    masks_prop_sim = df_masks[state].values
    if expo_decay:
        mask_wearing_factor = ((1 + theta_masks).T ** (-1 * masks_prop_sim)).T
    else:
        mask_wearing_factor = 2 * expit(
            -1 * theta_masks * masks_prop_sim[:, np.newaxis]
        )

    row = i // 2
    col = i % 2

    ax[row, col].plot(
        df_masks[state].index,
        np.median(mask_wearing_factor, axis=1),
        label="Microdistancing",
    )
    ax[row, col].fill_between(
        df_masks[state].index,
        np.quantile(mask_wearing_factor, 0.25, axis=1),
        np.quantile(mask_wearing_factor, 0.75, axis=1),
        label="Microdistancing",
        alpha=0.4,
        color="C0",
    )
    ax[row, col].fill_between(
        df_masks[state].index,
        np.quantile(mask_wearing_factor, 0.05, axis=1),
        np.quantile(mask_wearing_factor, 0.95, axis=1),
        label="Microdistancing",
        alpha=0.4,
        color="C0",
    )
    ax[row, col].set_title(state)
    ax[row, col].tick_params("x", rotation=45)

    ax[row, col].set_xticks(
        [df_masks[state].index.values[-n_forecast - extra_days_masks]], minor=True
    )
    ax[row, col].xaxis.grid(which="minor", linestyle="-.", color="grey", linewidth=1)

fig.text(
    0.03,
    0.5,
    "Multiplicative effect \n of mask-wearing $M_d$",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=20,
)

fig.text(0.5, 0.04, "Date", ha="center", va="center", fontsize=20)

plt.tight_layout(rect=[0.05, 0.04, 1, 1])
fig.savefig(
    "figs/"
    + "mobility_forecasts/" 
    + data_date.strftime("%Y-%m-%d") 
    + "_mobility_forecasts/mask_wearing_factor.png",
    dpi=144,
)

df_R = df_R.sort_values("date")
# samples = df_samples.sample(n_samples)  # test on sample of 2
# keep all samples 
samples = df_samples.iloc[:mob_samples, :]

def sigmoid(t, r, tau, m0, m1):
    y = m0 + (m1 - m0) / (1 + np.exp(-r * (t - tau)))
    return y.T
# for strain in ("Delta", "Omicron"):

# samples = df_samples

# flags for advanced scenario modelling
advanced_scenario_modelling = False
save_for_SA = False
# since this can be useful, predictor ordering is:
# ['retail_and_recreation_7days', 'grocery_and_pharmacy_7days', 'parks_7days', 'transit_stations_7days', 'workplaces_7days']
typ = "R_L"
forecast_type = ["R_L"]

for strain in ("Delta", "Omicron"):
    
    state_Rs = {
        "state": [],
        "date": [],
        "type": [],
        "median": [],
        "lower": [],
        "upper": [],
        "bottom": [],
        "top": [],
        "mean": [],
        "std": [],
    }

    ban = "2020-03-20"
    # VIC and NSW allow gatherings of up to 20 people, other jurisdictions allow for
    new_pol = "2020-06-01"

    expo_decay = True

    # start and end date for the third wave
    # Subtract 10 days to avoid right truncation
    third_end_date = data_date - pd.Timedelta(days=truncation_days)

    typ_state_R = {}
    mob_forecast_date = df_forecast.date.min()

    state_key = {
        "ACT": "1",
        "NSW": "2",
        "NT": "3",
        "QLD": "4",
        "SA": "5",
        "TAS": "6",
        "VIC": "7",
        "WA": "8",
    }

    total_N_p_third_omicron = 0
    for v in third_date_range.values():
        tmp = sum(v >= pd.to_datetime(omicron_start_date))
        # add a plus one for inclusion of end date (the else 0 is due to QLD having no Omicron potential)
        total_N_p_third_omicron += tmp if tmp > 0 else 0
    
    state_R = {}
    
    for (kk, state) in enumerate(states):
        # sort df_R by date so that rows are dates. rows are dates, columns are predictors
        df_state = df_R.loc[df_R.state == state]
        dd = df_state.date
        post_values = samples[predictors].values.T
        prop_sim = df_md[state].values
        # grab vaccination data
        vacc_ts_delta = df_ve_delta[state]
        vacc_ts_omicron = df_ve_omicron[state]
        # take right size of md to be N by N
        theta_md = np.tile(samples["theta_md"].values, (df_state.shape[0], 1))
        theta_masks = np.tile(samples["theta_masks"].values, (df_state.shape[0], 1))
        r = samples["r[" + str(kk + 1) + "]"].values
        tau = samples["tau[" + str(kk + 1) + "]"].values
        m0 = samples["m0[" + str(kk + 1) + "]"].values
        m1 = samples["m1[" + str(kk + 1) + "]"].values
        # m1 = 1.0
        md = ((1 + theta_md).T ** (-1 * prop_sim)).T
        masks = ((1 + theta_masks).T ** (-1 * masks_prop_sim)).T
        third_states_indices = {
            state: index + 1 for (index, state) in enumerate(third_states)
        }
        third_days = {k: v.shape[0] for (k, v) in third_date_range.items()}
        third_days_cumulative = np.append(
            [0], np.cumsum([v for v in third_days.values()])
        )
        vax_idx_ranges = {
            k: range(third_days_cumulative[i], third_days_cumulative[i + 1])
            for (i, k) in enumerate(third_days.keys())
        }
        third_days_tot = sum(v for v in third_days.values())
        # get the sampled vaccination effect (this will be incomplete as it's only over the fitting period)
        sampled_vax_effects_all = samples[
            ["ve_delta[" + str(j + 1) + "]" for j in range(third_days_tot)]
        ].T
        vacc_tmp = sampled_vax_effects_all.iloc[vax_idx_ranges[state], :]
        # now we layer in the posterior vaccine multiplier effect which ill be a (T,mob_samples) array
        # get before and after fitting and tile them
        vacc_ts_data_before = pd.concat(
            [vacc_ts_delta.loc[vacc_ts_delta.index < third_date_range[state][0]]]
            * mob_samples,
            axis=1,
        ).to_numpy()
        vacc_ts_data_after = pd.concat(
            [vacc_ts_delta.loc[vacc_ts_delta.index > third_date_range[state][-1]]]
            * mob_samples,
            axis=1,
        ).to_numpy()
        # merge in order
        vacc_ts_delta = np.vstack(
            [vacc_ts_data_before, vacc_tmp, vacc_ts_data_after]
        )
        # construct a range of dates for omicron which starts at the maximum of the start date for that state or the Omicron start date
        third_omicron_date_range = {
            k: pd.date_range(
                start=max(v[0], pd.to_datetime(omicron_start_date)), end=v[-1]
            ).values
            for (k, v) in third_date_range.items()
        }
        third_omicron_days = {
            k: v.shape[0] for (k, v) in third_omicron_date_range.items()
        }
        third_omicron_days_cumulative = np.append(
            [0], np.cumsum([v for v in third_omicron_days.values()])
        )
        omicron_ve_idx_ranges = {
            k: range(
                third_omicron_days_cumulative[i],
                third_omicron_days_cumulative[i + 1],
            )
            for (i, k) in enumerate(third_omicron_days.keys())
        }
        third_omicron_days_tot = sum(v for v in third_omicron_days.values())
        # get the sampled vaccination effect (this will be incomplete as it's only over the fitting period)
        sampled_vax_effects_all = (
            samples[
                ["ve_omicron[" + str(j + 1) + "]" for j in range(third_omicron_days_tot)]
            ].T
        )
        vacc_tmp = sampled_vax_effects_all.iloc[omicron_ve_idx_ranges[state], :]
        # now we layer in the posterior vaccine multiplier effect which ill be a (T,mob_samples) array

        # get before and after fitting and tile them
        vacc_ts_data_before = pd.concat(
            [
                vacc_ts_omicron.loc[
                    vacc_ts_omicron.index < third_omicron_date_range[state][0]
                ]
            ]
            * mob_samples,
            axis=1,
        ).to_numpy()
        vacc_ts_data_after = pd.concat(
            [
                vacc_ts_omicron.loc[
                    vacc_ts_omicron.index > third_date_range[state][-1]
                ]
            ]
            * mob_samples,
            axis=1,
        ).to_numpy()
        # merge in order
        vacc_ts_omicron = np.vstack(
            [vacc_ts_data_before, vacc_tmp, vacc_ts_data_after]
        )

        # setup some variables for handling the omicron starts
        third_states_indices = {
            state: index + 1 for (index, state) in enumerate(third_states)
        }
        omicron_start_day = (
            pd.to_datetime(omicron_start_date) - pd.to_datetime(start_date)
        ).days
        days_into_omicron = np.cumsum(
            np.append(
                [0],
                [
                    (v >= pd.to_datetime(omicron_start_date)).sum()
                    for v in third_date_range.values()
                ],
            )
        )

        idx = {}
        kk = 0
        for k in third_date_range.keys():
            idx[k] = range(days_into_omicron[kk], days_into_omicron[kk + 1])
            kk += 1
            
        # tile the reduction in vaccination effect for omicron (i.e. VE is (1+r)*VE)
        voc_vacc_product = np.zeros_like(vacc_ts_delta)
        # calculate the voc effects
        voc_multiplier_delta = samples["voc_effect_delta"].values
        voc_multiplier_omicron = samples["voc_effect_omicron"].values

        # sample the right R_L
        sim_R = samples["R_Li[" + state_key[state] + "]"].values

        for n in range(mob_samples):
            # add gaussian noise to predictors before forecast
            # df_state.loc[
            df_state.loc[df_state.date < mob_forecast_date, predictors] = (
                state_Rmed[state][:, :, n] / 100
            )

            # add gaussian noise to predictors after forecast
            df_state.loc[df_state.date >= mob_forecast_date, predictors] = (
                state_sims[state][:, :, n] / 100
            )

            ## ADVANCED SCENARIO MODELLING - USE ONLY FOR POINT ESTIMATES
            # set non-grocery values to 0
            if advanced_scenario_modelling:
                df_state.loc[:, predictors[0]] = 0
                df_state.loc[:, predictors[2]] = 0
                df_state.loc[:, predictors[3]] = 0
                df_state.loc[:, predictors[4]] = 0

            df1 = df_state.loc[df_state.date <= ban]
            X1 = df1[predictors]  # N by K
            md[: X1.shape[0], :] = 1

            if n == 0:
                # initialise arrays (loggodds)
                # N by K times (Nsamples by K )^T = Ndate by Nsamples
                logodds = X1 @ post_values[:, n]

                df2 = df_state.loc[
                    (df_state.date > ban) & (df_state.date < new_pol)
                ]
                df3 = df_state.loc[df_state.date >= new_pol]
                X2 = df2[predictors]
                X3 = df3[predictors]
                logodds = np.append(logodds, X2 @ post_values[:, n], axis=0)
                logodds = np.append(logodds, X3 @ post_values[:, n], axis=0)

            else:
                # concatenate to pre-existing logodds martrix
                logodds1 = X1 @ post_values[:, n]

                df2 = df_state.loc[
                    (df_state.date > ban) & (df_state.date < new_pol)
                ]
                df3 = df_state.loc[df_state.date >= new_pol]
                X2 = df2[predictors]
                X3 = df3[predictors]
                prop2 = df_md.loc[ban:new_pol, state].values
                prop3 = df_md.loc[new_pol:, state].values
                logodds2 = X2 @ post_values[:, n]
                logodds3 = X3 @ post_values[:, n]
                logodds_sample = np.append(logodds1, logodds2, axis=0)
                logodds_sample = np.append(logodds_sample, logodds3, axis=0)

                # concatenate to previous
                logodds = np.vstack((logodds, logodds_sample))
                
        # create an matrix of mob_samples realisations which is an indicator of the voc (delta right now)
        # which will be 1 up until the voc_start_date and then it will be values from the posterior sample
        voc_multiplier_alpha = samples["voc_effect_alpha"].values
        voc_multiplier_delta = samples["voc_effect_delta"].values
        voc_multiplier_omicron = samples["voc_effect_omicron"].values

        # number of days into omicron forecast
        tt = 0

        # loop over days in third wave and apply the appropriate form (i.e. decay or not)
        # note that in here we apply the entire sample to the vaccination data to create a days by samples array
        tmp_date = pd.to_datetime("2020-03-01")

        # get the correct Omicron start date 
        # omicron_start_date_tmp = np.maximum(
        #     pd.to_datetime(omicron_start_date),
        #     pd.to_datetime(third_date_range[state][0]),
        # )
        
        omicron_start_date_tmp = pd.to_datetime(omicron_start_date)

        omicron_start_day_tmp = (
            pd.to_datetime(omicron_start_date_tmp) - pd.to_datetime(start_date)
        ).days

        for ii in range(mob_samples):
            # if before omicron introduced in a jurisdiction, we consider what period we're at: 
            # 1. Wildtype
            # 2. Alpha
            # 3. Delta
            voc_vacc_product[:, ii] = vacc_ts_delta[:, ii]
            
            idx_start = df_state.loc[df_state.date < alpha_start_date].shape[0]
            idx_end = df_state.loc[df_state.date < delta_start_date].shape[0]
            voc_vacc_product[idx_start:idx_end, ii] *= voc_multiplier_alpha[ii]
            
            idx_start = idx_end
            idx_end = df_state.loc[df_state.date < omicron_start_date_tmp].shape[0]
            voc_vacc_product[idx_start:idx_end, ii] *= voc_multiplier_delta[ii]
            
            idx_start = idx_end
            idx_end = np.shape(voc_vacc_product)[0]
            
            if strain == "Delta":
                voc_vacc_product[idx_start:idx_end, ii] *= voc_multiplier_delta[ii]
            elif strain == "Omicron":
                # if omicron we need to account for the Omicron VE prior to the introduction of 
                # omicron in mid November 
                voc_vacc_product[idx_start:idx_end, ii] = (
                    vacc_ts_omicron[idx_start:idx_end, ii] * voc_multiplier_omicron[ii]
                )


        # save the components of the TP
        pd.DataFrame(sim_R).to_csv(results_dir + "baseline_R_L_" + strain + ".csv")
        pd.DataFrame(md).to_csv(results_dir + "md_" + strain + ".csv")
        pd.DataFrame(masks).to_csv(results_dir + "masks_" + strain + ".csv")
        macro = 2 * expit(logodds.T)
        pd.DataFrame(macro).to_csv(results_dir + "macro_" + strain + ".csv")
        pd.DataFrame(voc_vacc_product).to_csv(results_dir + "voc_vacc_product_" + strain + ".csv")
        
        # calculate TP
        R_L = (
            2 * expit(logodds.T)
            * md
            * masks
            * sim_R
            * voc_vacc_product
        )
    
        # now we increase TP by 15% based on school reopening (this code can probably be reused 
        # but inferring it would be pretty difficult
        # due to lockdowns and various interruptions since March 2020)
        if scenarios[state] == "school_opening_2022":
            R_L[dd.values >= pd.to_datetime(scenario_dates[state]), :] = (
                1.15 * R_L[dd.values >= pd.to_datetime(scenario_dates[state]), :]
            )

        # calculate summary stats
        R_L_med = np.median(R_L, axis=1)
        R_L_lower = np.percentile(R_L, 25, axis=1)
        R_L_upper = np.percentile(R_L, 75, axis=1)
        R_L_bottom = np.percentile(R_L, 5, axis=1)
        R_L_top = np.percentile(R_L, 95, axis=1)

        # R_L
        state_Rs["state"].extend([state] * df_state.shape[0])
        state_Rs["type"].extend([typ] * df_state.shape[0])
        state_Rs["date"].extend(dd.values)  # repeat mob_samples times?
        state_Rs["lower"].extend(R_L_lower)
        state_Rs["median"].extend(R_L_med)
        state_Rs["upper"].extend(R_L_upper)
        state_Rs["top"].extend(R_L_top)
        state_Rs["bottom"].extend(R_L_bottom)
        state_Rs["mean"].extend(np.mean(R_L, axis=1))
        state_Rs["std"].extend(np.std(R_L, axis=1))

        state_R[state] = R_L

    # generate a summary for the R_I
    for state in states:
        # R_I
        if strain == "Delta":
            R_I = samples["R_I"].values[:df_state.shape[0]]
        elif strain == "Omicron":
            # if Omicron period, then we need to multiply in the VoC effect as there's a period 
            # in the fitting where Delta and Omicron overlap (i.e. R_I = R_I * P(t) where P(t) is 
            # a product term).
            R_I = samples["R_I_omicron"].values[:df_state.shape[0]] 
            
        state_Rs["state"].extend([state] * df_state.shape[0])
        state_Rs["type"].extend(["R_I"] * df_state.shape[0])
        state_Rs["date"].extend(dd.values)
        state_Rs["lower"].extend(np.repeat(np.percentile(R_I, 25), df_state.shape[0]))
        state_Rs["median"].extend(np.repeat(np.median(R_I), df_state.shape[0]))
        state_Rs["upper"].extend(np.repeat(np.percentile(R_I, 75), df_state.shape[0]))
        state_Rs["top"].extend(np.repeat(np.percentile(R_I, 95), df_state.shape[0]))
        state_Rs["bottom"].extend(np.repeat(np.percentile(R_I, 5), df_state.shape[0]))
        state_Rs["mean"].extend(np.repeat(np.mean(R_I), df_state.shape[0]))
        state_Rs["std"].extend(np.repeat(np.std(R_I), df_state.shape[0]))

    df_Rhats = pd.DataFrame().from_dict(state_Rs)
    df_Rhats = df_Rhats.set_index(["state", "date", "type"])

    d = pd.DataFrame()
    for state in states:
        for i, typ in enumerate(forecast_type):
            if i == 0:
                t = pd.DataFrame.from_dict(state_R[state])
                t["date"] = dd.values
                t["state"] = state
                t["type"] = typ
            else:
                temp = pd.DataFrame.from_dict(state_R[state])
                temp["date"] = dd.values
                temp["state"] = state
                temp["type"] = typ
                t = t.append(temp)
        # R_I
        if strain == "Delta": 
            # use the Delta import reproduction number before Omicron starts
            i = pd.DataFrame(np.tile(samples["R_I"].values, (len(dd.values), 1)))
        elif strain == "Omicron":
            # use the Omicron import reproduction number after Omicron starts
            i = pd.DataFrame(np.tile(samples["R_I_omicron"].values, (len(dd.values), 1)))
        
        i["date"] = dd.values
        i["type"] = "R_I"
        i["state"] = state

        t = t.append(i)

        d = d.append(t)

    d = d.set_index(["state", "date", "type"])
    df_Rhats = df_Rhats.join(d)
    df_Rhats = df_Rhats.reset_index()
    df_Rhats.state = df_Rhats.state.astype(str)
    df_Rhats.type = df_Rhats.type.astype(str)

    fig, ax = plt.subplots(figsize=(12, 9), nrows=4, ncols=2, sharex=True, sharey=True)

    for i, state in enumerate(plot_states):

        row = i // 2
        col = i % 2

        plot_df = df_Rhats.loc[(df_Rhats.state == state) & (df_Rhats.type == "R_L")]
        # split the TP into pre data date and after
        plot_df_backcast = plot_df.loc[plot_df["date"] <= data_date]
        plot_df_forecast = plot_df.loc[plot_df["date"] > data_date]
        # plot the backcast TP
        ax[row, col].plot(plot_df_backcast.date, plot_df_backcast["median"], color="C0")
        ax[row, col].fill_between(
            plot_df_backcast.date,
            plot_df_backcast["lower"],
            plot_df_backcast["upper"],
            alpha=0.4,
            color="C0",
        )
        ax[row, col].fill_between(
            plot_df_backcast.date,
            plot_df_backcast["bottom"],
            plot_df_backcast["top"],
            alpha=0.4,
            color="C0",
        )
        # plot the forecast TP
        ax[row, col].plot(plot_df_forecast.date, plot_df_forecast["median"], color="C1")
        ax[row, col].fill_between(
            plot_df_forecast.date,
            plot_df_forecast["lower"],
            plot_df_forecast["upper"],
            alpha=0.4,
            color="C1",
        )
        ax[row, col].fill_between(
            plot_df_forecast.date,
            plot_df_forecast["bottom"],
            plot_df_forecast["top"],
            alpha=0.4,
            color="C1",
        )

        ax[row, col].tick_params("x", rotation=90)
        ax[row, col].set_title(state)
        ax[row, col].set_yticks(
            [1],
            minor=True,
        )
        ax[row, col].set_yticks([0, 2, 4, 6], minor=False)
        ax[row, col].set_yticklabels([0, 2, 4, 6], minor=False)
        ax[row, col].yaxis.grid(which="minor", linestyle="--", color="black", linewidth=2)
        ax[row, col].set_ylim((0, 6))

        # ax[row, col].set_xticks([plot_df.date.values[-n_forecast]], minor=True)
        ax[row, col].axvline(data_date, ls="-.", color="black", lw=1)
        # plot window start date
        plot_window_start_date = min(
            pd.to_datetime(today) - timedelta(days=6 * 30),
            sim_start_date - timedelta(days=truncation_days),
        )

        # create a plot window over the last six months
        ax[row, col].set_xlim(
            plot_window_start_date,
            pd.to_datetime(today) + timedelta(days=num_forecast_days),
        )
        # plot the start date
        ax[row, col].axvline(sim_start_date, ls="--", color="green", lw=2)
        ax[row, col].xaxis.grid(which="minor", linestyle="-.", color="grey", linewidth=2)

    fig.text(
        0.03,
        0.5,
        "Transmission potential",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=20,
    )
    fig.text(0.525, 0.02, "Date", va="center", ha="center", fontsize=20)
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])

    plt.savefig(
        "figs/mobility_forecasts/"
        + data_date.strftime("%Y-%m-%d")
        + "_mobility_forecasts/TP_6_month_" 
        + strain
        + data_date.strftime("%Y-%m-%d")
        + ".png",
        dpi=144,
    )

    fig, ax = plt.subplots(figsize=(12, 9), nrows=4, ncols=2, sharex=True, sharey=True)

    for i, state in enumerate(plot_states):

        row = i // 2
        col = i % 2

        plot_df = df_Rhats.loc[(df_Rhats.state == state) & (df_Rhats.type == "R_L")]
        # split the TP into pre data date and after
        plot_df_backcast = plot_df.loc[plot_df["date"] <= data_date]
        plot_df_forecast = plot_df.loc[plot_df["date"] > data_date]
        # plot the backcast TP
        ax[row, col].plot(plot_df_backcast.date, plot_df_backcast["median"], color="C0")
        ax[row, col].fill_between(
            plot_df_backcast.date,
            plot_df_backcast["lower"],
            plot_df_backcast["upper"],
            alpha=0.4,
            color="C0",
        )
        ax[row, col].fill_between(
            plot_df_backcast.date,
            plot_df_backcast["bottom"],
            plot_df_backcast["top"],
            alpha=0.4,
            color="C0",
        )
        # plot the forecast TP
        ax[row, col].plot(plot_df_forecast.date, plot_df_forecast["median"], color="C1")
        ax[row, col].fill_between(
            plot_df_forecast.date,
            plot_df_forecast["lower"],
            plot_df_forecast["upper"],
            alpha=0.4,
            color="C1",
        )
        ax[row, col].fill_between(
            plot_df_forecast.date,
            plot_df_forecast["bottom"],
            plot_df_forecast["top"],
            alpha=0.4,
            color="C1",
        )

        ax[row, col].tick_params("x", rotation=90)
        ax[row, col].set_title(state)
        ax[row, col].set_yticks(
            [1],
            minor=True,
        )
        ax[row, col].set_yticks([0, 2, 4, 6], minor=False)
        ax[row, col].set_yticklabels([0, 2, 4, 6], minor=False)
        ax[row, col].yaxis.grid(which="minor", linestyle="--", color="black", linewidth=2)
        ax[row, col].set_ylim((0, 6))

        # ax[row, col].set_xticks([plot_df.date.values[-n_forecast]], minor=True)
        ax[row, col].axvline(data_date, ls="-.", color="black", lw=1)
        # plot window start date
        plot_window_start_date = min(
            pd.to_datetime(today) - timedelta(days=12 * 30),
            sim_start_date - timedelta(days=truncation_days),
        )

        # create a plot window over the last six months
        ax[row, col].set_xlim(
            plot_window_start_date,
            pd.to_datetime(today) + timedelta(days=num_forecast_days),
        )
        # plot the start date
        ax[row, col].axvline(sim_start_date, ls="--", color="green", lw=2)
        ax[row, col].xaxis.grid(which="minor", linestyle="-.", color="grey", linewidth=2)

    fig.text(
        0.03,
        0.5,
        "Transmission potential",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=20,
    )
    fig.text(0.525, 0.02, "Date", va="center", ha="center", fontsize=20)
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])

    print("============")
    print("Saving results")
    print("============")

    plt.savefig(
        "figs/mobility_forecasts/"
        + data_date.strftime("%Y-%m-%d")
        + "_mobility_forecasts/TP_12_month_"
        + strain
        + data_date.strftime("%Y-%m-%d")
        + ".png",
        dpi=144,
    )

    # save values for the functional omicron related proportions for each state
    prop_omicron_vars = ("r", "tau", "m0", "m1")

    for (kk, state) in enumerate(states):
        # sort df_R by date so that rows are dates. rows are dates, columns are predictors
        df_state = df_R.loc[df_R.state == state]
        for v in prop_omicron_vars:
            # take right size of the values to be N by N
            y = samples[v + "[" + str(kk + 1) + "]"].values
            
            pd.DataFrame(y[:mob_samples]).to_csv(
                results_dir
                + v
                + "_"
                + state
                + data_date.strftime("%Y-%m-%d")
                + ".csv"
            )

    # now we save the sampled TP paths
    # convert the appropriate sampled susceptible depletion factors to a csv and save them for simulation
    # NOTE: this will not save an updated median, mean etc for the R_I's. We don't use it so it's not 
    # really important but it should be noted for later if we are comparing things. The step function 
    # R_I -> R_I_omicron, is noticeable and shouldn't be overlooked.
    df_Rhats = df_Rhats[
        ["state", "date", "type", "median", "bottom", "lower", "upper", "top"]
        + [i for i in range(mob_samples)]
    ]

    # # save the file as a csv (easier to handle in Julia for now)
    df_Rhats.to_csv(
        results_dir 
        + "soc_mob_R_" 
        + strain 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )