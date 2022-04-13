import glob
import os
import sys
from tracemalloc import start

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
print("Adjusting TP forecasts using data from", data_date)
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

omicron_start_day = (pd.to_datetime(omicron_start_date) - pd.to_datetime(start_date)).days

for strain in ("Delta", "Omicron"):
    """
    Run adjustment model for the local TP estimates. This will adjust the local component of the
    TP
    """

    print("=========================")
    print("Running TP adjustment model...")
    print("=========================")

    df_forecast2 = pd.read_csv(
        results_dir 
        + "soc_mob_R_"
        + strain 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )
    # read in Reff samples
    df_Reff = pd.read_csv(
        "results/EpyReff/Reff_" 
        + strain 
        + "_samples" 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv",
        parse_dates=["INFECTION_DATES"],
    )

    inferred_prop_imports = pd.read_csv(
        results_dir
        + "rho_samples" 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv", 
        parse_dates=["date"],
    )

    # read in the case data and note that we want this to be infection dates to match up to Reff changes
    case_data = read_in_NNDSS(
        data_date, apply_delay_at_read=True, apply_inc_at_read=True
    )
    case_data = case_data[["date_inferred", "STATE", "imported", "local"]]
    # this is the forecasted TP dataframe, without R_L type
    df_forecast2_new = df_forecast2.loc[df_forecast2.type != "R_L"]
    end_date = pd.to_datetime(today) + timedelta(days=num_forecast_days)
    states_to_adjust = ["NSW", "QLD", "SA", "VIC", "TAS", "WA", "ACT", "NT"]

    # read in the samples for weighting between TP and Reff.
    samples2 = pd.read_csv(
        results_dir 
        + "posterior_sample_" 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )

    # extract the import values
    if strain == "Delta":
        R_I = samples2.R_I.to_numpy()
        R_I_omicron = samples2.R_I_omicron.to_numpy()
        voc_effect = samples2.voc_effect_delta.to_numpy()
    elif strain == "Omicron":
        # extract the import values
        R_I_omicron = samples2.R_I_omicron.to_numpy()
        voc_effect = samples2.voc_effect_omicron.to_numpy()

    def calculate_Reff_local(
        Reff, 
        R_I, 
        R_I_omicron, 
        voc_effect, 
        prop_import, 
        omicron_start_day,
    ):
        """
        Apply the same mixture model idea as per the TP model to get
        R_eff^L = (R_eff - rho * RI)/(1 - rho)
        and use this to weight the TP historically.
        """

        # calculate this all in one step. Note that we set the Reff to -1 if
        # the prop_import = 1 as in that instance the relationship breaks due to division by 0.
        Reff_local = np.zeros(shape=Reff.shape[0])
        
        for n in range(len(Reff_local)):
            # adjust the Reff based on the time period of interest
            if n < omicron_start_day:
                R_I_tmp = R_I
            else:
                R_I_tmp = R_I_omicron * voc_effect
            
            if prop_import[n] < 1:
                Reff_local[n] = (Reff[n] - prop_import[n] * R_I_tmp) / (1 - prop_import[n]) 
            else:
                Reff_local[n] = 0
        
        # Reff_local = [
        #     (Reff[t] - prop_import[t] * R_I) / (1 - prop_import[t])
        #     if prop_import[t] < 1 else -1 for t in range(Reff.shape[0])
        # ]

        return Reff_local


    last_date_for_reff = (
        pd.to_datetime(data_date) 
        - pd.Timedelta(days=truncation_days + n_days_nowcast_TP_adjustment - 1)
    )

    print("==============")
    print("The last date the Reff estimate is used is", last_date_for_reff)
    print("==============")

    for state in states:

        # filter case data by state
        case_data_state = case_data.loc[case_data.STATE == state]
        # take a sum of cases each day (this does not fill out missing days)
        df_cases = case_data_state.groupby(["date_inferred", "STATE"]).agg(sum)
        df_cases = df_cases.reset_index()
        df_cases = df_cases.set_index("date_inferred")
        # now we want to fill out indices by adding 0's on days with 0 cases and ensuring we go right up to the current truncated date
        idx = pd.date_range(
            pd.to_datetime("2020-03-01"),
            last_date_for_reff,
        )
        
        is_omicron = np.array(idx >= pd.to_datetime(omicron_start_date))
        df_cases = df_cases.reindex(idx, fill_value=0)

        # filter the TP and Reff by state
        df_forecast2_state_R_L = df_forecast2.loc[
            ((df_forecast2.state == state) & (df_forecast2.type == "R_L"))
        ]
        df_Reff_state = df_Reff.loc[df_Reff.STATE == state]

        # take a rolling average of the cases over the interval of consideration
        idx = (pd.to_datetime(df_forecast2_state_R_L.date) >= pd.to_datetime("2020-03-01")) & (
            pd.to_datetime(df_forecast2_state_R_L.date) <= last_date_for_reff
        )
        df_forecast2_state_R_L_sims = df_forecast2_state_R_L.iloc[:, 9:].loc[idx]

        Reff = df_Reff_state.loc[
            (df_Reff_state.INFECTION_DATES >= pd.to_datetime("2020-03-01"))
            & (df_Reff_state.INFECTION_DATES<= last_date_for_reff)
        ].iloc[:, :-2]

        # take 7-day moving averages for the local, imported, and total cases
        ma_period = 7
        df_cases_local = df_cases["local"]
        df_cases_imported = df_cases["imported"]
        df_cases_local_ma = df_cases_local.rolling(7, min_periods=1).mean()
        # only want to use indices over the fitting horizon, after this point we rely on the TP model
        idx = (df_cases.index >= pd.to_datetime("2020-03-01")) & (
            df_cases.index <= last_date_for_reff
        )
        df_cases_local = df_cases_local[idx]
        df_cases_imported = df_cases_imported[idx]
        df_cases_local_ma = df_cases_local_ma[idx]
        # dictionary to store sampled Rt paths
        Rt = {}

        ratio_import_to_local = df_cases_imported / (df_cases_local + df_cases_imported)
        # set nan or infs to 0
        ratio_import_to_local.replace([np.nan, np.inf], 0, inplace=True)
        ratio_import_to_local = ratio_import_to_local.rolling(7, min_periods=1).mean()
        # now replace the fitted period with the correct proportions
        inferred_prop_imports_state = (
            inferred_prop_imports
            .loc[inferred_prop_imports.state == state]
            .iloc[:,1:]
            .set_index("date")
            .mean(axis=1)
        )
        ratio_import_to_local_combined = pd.Series(
            inferred_prop_imports_state[i] 
            if i in inferred_prop_imports_state.index else ratio_import_to_local[i]
            for i in ratio_import_to_local.index
        )
        ratio_import_to_local_combined.index = ratio_import_to_local.index
        ratio_import_to_local_combined = ratio_import_to_local_combined.to_numpy()

        n_Reff_samples = Reff.shape[1]

        # loop over the TP paths for a state
        for (n, col) in enumerate(df_forecast2_state_R_L_sims):
            if state in states_to_adjust:
                # sample a Reff path from EpyReff (there are only 2000 of these)
                Reff_sample = Reff.iloc[:, n % n_Reff_samples].to_numpy()
                TP_local = np.array(df_forecast2_state_R_L_sims[col])
                # Index by col % n_samples as we would be cycling the values in the R_I
                Reff_local = calculate_Reff_local(
                    Reff_sample, 
                    R_I[int(col) % mob_samples], 
                    R_I_omicron[int(col) % mob_samples], 
                    voc_effect[int(col) % mob_samples],
                    ratio_import_to_local_combined,
                    omicron_start_day=omicron_start_day,
                )
                omega = pd.Series(
                    (
                        np.random.beta(35, L_ma) if L_ma >= 5 else 1
                        for L_ma in df_cases_local_ma.to_numpy()
                    ),
                    index=df_cases_local_ma.index,
                )

                # apply the mixture modelling and the adjustment to ensure we don't get negative
                Rt[col] = np.maximum(0, (1 - omega) * Reff_local + omega * TP_local)

        # store Rt in a dataframe
        Rt = pd.DataFrame.from_dict(Rt, orient="index", columns=df_cases_local_ma.index)

        # next step is to stich the adjusted df back with the forecasting of TP
        idx = pd.to_datetime(df_forecast2_state_R_L.date) > last_date_for_reff
        df_forecast2_after = df_forecast2_state_R_L.iloc[:, 9:].loc[idx].T
        # concatenate updated df with the forecasted TP
        df_full = pd.concat([Rt, df_forecast2_after], axis=1)
        # transpose the full df for consistent structuring
        df_full = df_full.T
        # calculate the summary statistics as per the original df
        df_full["bottom"] = np.percentile(df_full, 5, axis=1)
        df_full["lower"] = np.percentile(df_full, 25, axis=1)
        df_full["median"] = np.percentile(df_full, 50, axis=1)
        df_full["upper"] = np.percentile(df_full, 75, axis=1)
        df_full["top"] = np.percentile(df_full, 95, axis=1)
        df_full["mean"] = np.mean(df_full, axis=1)
        df_full["std"] = np.std(df_full, axis=1)
        # put date back in
        df_full["date"] = pd.date_range(start_date, periods=df_full.shape[0])
        df_full["state"] = [state] * df_full.shape[0]
        df_full["type"] = ["R_L"] * df_full.shape[0]
        # reset indices
        df_full.reset_index(drop=True, inplace=True)
        # merge df back with the other variables
        df_forecast2_new = pd.concat([df_forecast2_new, df_full], axis=0)

    fig, ax = plt.subplots(figsize=(12, 9), nrows=4, ncols=2, sharex=True, sharey=True)

    for i, state in enumerate(plot_states):

        row = i // 2
        col = i % 2

        plot_df = df_forecast2_new.loc[
            (df_forecast2_new.state == state) & (df_forecast2_new.type == "R_L")
        ]

        # split the TP into pre data date and after
        plot_df_backcast = plot_df.loc[plot_df["date"] <= data_date]
        plot_df_forecast2 = plot_df.loc[plot_df["date"] > data_date]
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
        ax[row, col].plot(plot_df_forecast2.date, plot_df_forecast2["median"], color="C1")
        ax[row, col].fill_between(
            plot_df_forecast2.date,
            plot_df_forecast2["lower"],
            plot_df_forecast2["upper"],
            alpha=0.4,
            color="C1",
        )
        ax[row, col].fill_between(
            plot_df_forecast2.date,
            plot_df_forecast2["bottom"],
            plot_df_forecast2["top"],
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
        + "_mobility_forecasts/TP_6_month_adjusted_"
        + strain
        + data_date.strftime("%Y-%m-%d")
        + ".png",
        dpi=144,
    )

    print("=========================")
    print("Saving forecasting results...")
    print("=========================")

    # trim off some columns that will be poorly formatted 
    df_forecast2_new = df_forecast2_new.iloc[:, 1:-2]
    # reformat dates
    df_forecast2_new["date"] = pd.to_datetime(df_forecast2_new["date"])

    df_forecast2_new.to_csv(
        results_dir 
        + "soc_mob_R_adjusted_" 
        + strain 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )