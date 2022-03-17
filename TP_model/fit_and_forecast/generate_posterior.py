######### imports #########
from ast import arg
from datetime import timedelta
import sys

sys.path.insert(0, "TP_model")
sys.path.insert(0, "TP_model/fit_and_forecast")
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
import matplotlib
from math import ceil
import pickle
import stan

matplotlib.use("Agg")

from params import (
    truncation_days,
    download_google_automatically,
    third_start_date,
    alpha_start_date, 
    omicron_start_date,
    omicron_dominance_date,
    pop_sizes,
    num_forecast_days,
    p_detect_delta, 
    p_detect_omicron, 
)

def process_vax_data_array(
    data_date, 
    third_states, 
    third_end_date, 
    variant="Delta",
    print_latest_date_in_ts=False,
):
    """
    Processes the vaccination data to an array for either the Omicron or Delta strain.
    """
    # Load in vaccination data by state and date
    vaccination_by_state = pd.read_csv(
        "data/vaccine_effect_timeseries_" + data_date.strftime("%Y-%m-%d") + ".csv",
        parse_dates=["date"],
    )
    # there are a couple NA's early on in the time series but is likely due to slightly 
    # different start dates
    vaccination_by_state.fillna(1, inplace=True)
    vaccination_by_state = vaccination_by_state.loc[
        vaccination_by_state["variant"] == variant
    ]
    vaccination_by_state = vaccination_by_state[["state", "date", "effect"]]

    if print_latest_date_in_ts: 
        # display the latest available date in the NSW data (will be the same date between states)
        print(
            "Latest date in vaccine data is {}".format(
                vaccination_by_state[vaccination_by_state.state == "NSW"].date.values[-1]
            )
        )

    # Get only the dates we need + 1 (this serves as the initial value)
    vaccination_by_state = vaccination_by_state[
        (
            vaccination_by_state.date
            >= pd.to_datetime(third_start_date) - timedelta(days=1)
        )
        & (vaccination_by_state.date <= third_end_date)
    ]
    vaccination_by_state = vaccination_by_state[
        vaccination_by_state["state"].isin(third_states)
    ]  # Isolate fitting states
    vaccination_by_state = vaccination_by_state.pivot(
        index="state", columns="date", values="effect"
    )  # Convert to matrix form

    # If we are missing recent vaccination data, fill it in with the most recent available data.
    latest_vacc_data = vaccination_by_state.columns[-1]
    if latest_vacc_data < pd.to_datetime(third_end_date):
        vaccination_by_state = pd.concat(
            [vaccination_by_state]
            + [
                pd.Series(vaccination_by_state[latest_vacc_data], name=day)
                for day in pd.date_range(start=latest_vacc_data, end=third_end_date)
            ],
            axis=1,
        )

    # Convert to simple array only useful to pass to stan (index 1 onwards)
    vaccination_by_state_array = vaccination_by_state.iloc[:, 1:].to_numpy()

    return vaccination_by_state_array


def get_data_for_posterior(data_date):
    """
    Read in the various datastreams and combine the samples into a dictionary that we then
    dump to a pickle file.
    """

    print("Performing inference on state level Reff")
    data_date = pd.to_datetime(data_date)  # Define data date
    print("Data date is {}".format(data_date.strftime("%d%b%Y")))
    fit_date = pd.to_datetime(data_date - timedelta(days=truncation_days))
    print("Last date in fitting {}".format(fit_date.strftime("%d%b%Y")))

    # * Note: 2020-09-09 won't work (for some reason)
    # read in microdistancing survey data
    surveys = pd.DataFrame()
    path = "data/md/Barometer wave*.csv"
    for file in glob.glob(path):
        surveys = surveys.append(pd.read_csv(file, parse_dates=["date"]))

    surveys = surveys.sort_values(by="date")
    print("Latest Microdistancing survey is {}".format(surveys.date.values[-1]))

    surveys["state"] = surveys["state"].map(states_initials).fillna(surveys["state"])
    surveys["proportion"] = surveys["count"] / surveys.respondents
    surveys.date = pd.to_datetime(surveys.date)

    always = surveys.loc[surveys.response == "Always"].set_index(["state", "date"])
    always = always.unstack(["state"])
    # If you get an error here saying 'cannot create a new series when the index is not unique', 
    # then you have a duplicated md file.

    idx = pd.date_range("2020-03-01", pd.to_datetime("today"))
    always = always.reindex(idx, fill_value=np.nan)
    always.index.name = "date"

    # fill back to earlier and between weeks.
    # Assume survey on day x applies for all days up to x - 6
    always = always.fillna(method="bfill")
    # assume values continue forward if survey hasn't completed
    always = always.fillna(method="ffill")
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
    survey_counts_base = (
        pd.pivot_table(data=always, index="date", columns="state", values="count")
        .drop(["Australia", "Other"], axis=1)
        .astype(int)
    )

    survey_respond_base = (
        pd.pivot_table(data=always, index="date", columns="state", values="respondents")
        .drop(["Australia", "Other"], axis=1)
        .astype(int)
    )

    # read in and process mask wearing data
    mask_wearing = pd.DataFrame()
    path = "data/face_coverings/face_covering_*_.csv"
    for file in glob.glob(path):
        mask_wearing = mask_wearing.append(pd.read_csv(file, parse_dates=["date"]))

    mask_wearing = mask_wearing.sort_values(by="date")
    print("Latest Mask wearing survey is {}".format(mask_wearing.date.values[-1]))

    mask_wearing["state"] = (
        mask_wearing["state"].map(states_initials).fillna(mask_wearing["state"])
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
    # assume values continue forward if survey hasn't completed
    mask_wearing_always = mask_wearing_always.fillna(method="ffill")
    mask_wearing_always = mask_wearing_always.stack(["state"])

    # Zero out before first survey 20th March
    mask_wearing_always = mask_wearing_always.reset_index().set_index("date")
    mask_wearing_always.loc[:"2020-03-20", "count"] = 0
    mask_wearing_always.loc[:"2020-03-20", "respondents"] = 0
    mask_wearing_always.loc[:"2020-03-20", "proportion"] = 0

    mask_wearing_X = pd.pivot_table(
        data=mask_wearing_always, index="date", columns="state", values="proportion"
    )
    mask_wearing_counts_base = pd.pivot_table(
        data=mask_wearing_always, index="date", columns="state", values="count"
    ).astype(int)
    mask_wearing_respond_base = pd.pivot_table(
        data=mask_wearing_always, index="date", columns="state", values="respondents"
    ).astype(int)

    df_Reff = pd.read_csv(
        "results/EpyReff/Reff_delta" + data_date.strftime("%Y-%m-%d") + "tau_5.csv",
        parse_dates=["INFECTION_DATES"],
    )
    df_Reff["date"] = df_Reff.INFECTION_DATES
    df_Reff["state"] = df_Reff.STATE
    
    df_Reff_omicron = pd.read_csv(
        "results/EpyReff/Reff_omicron" + data_date.strftime("%Y-%m-%d") + "tau_5.csv",
        parse_dates=["INFECTION_DATES"],
    )
    df_Reff_omicron["date"] = df_Reff_omicron.INFECTION_DATES
    df_Reff_omicron["state"] = df_Reff_omicron.STATE
    
    # relabel some of the columns to avoid replication in the merged dataframe 
    col_names_replace = {
        "mean": "mean_omicron",
        "lower": "lower_omicron",
        "upper": "upper_omicron",
        "top": "top_omicron",
        "bottom": "bottom_omicron",
        "std": "std_omicron",
    }

    df_Reff_omicron.rename(col_names_replace, axis=1, inplace=True)

    # read in NNDSS/linelist data
    # If this errors it may be missing a leading zero on the date.
    df_state = read_in_cases(
        case_file_date=data_date.strftime("%d%b%Y"),
        apply_delay_at_read=True,
        apply_inc_at_read=True,
    )

    df_Reff = df_Reff.merge(
        df_state,
        how="left",
        left_on=["state", "date"],
        right_on=["STATE", "date_inferred"],
    )  # how = left to use Reff days, NNDSS missing dates

    # merge in the omicron stuff 
    df_Reff = df_Reff.merge(
        df_Reff_omicron, 
        how="left", 
        left_on=["state", "date"],
        right_on=["state", "date"],
    )
    
    df_Reff["rho_moving"] = df_Reff.groupby(["state"])["rho"].transform(
        lambda x: x.rolling(7, 1).mean()
    )  # minimum number of 1

    # some days have no cases, so need to fillna
    df_Reff["rho_moving"] = df_Reff.rho_moving.fillna(method="bfill")

    # counts are already aligned with infection date by subtracting a random incubation period
    df_Reff["local"] = df_Reff.local.fillna(0)
    df_Reff["imported"] = df_Reff.imported.fillna(0)

    ######### Read in Google mobility results #########
    sys.path.insert(0, "../")

    df_google = read_in_google(local=not download_google_automatically, moving=True)
    df = df_google.merge(
        df_Reff[
            [
                "date",
                "state",
                "mean",
                "lower",
                "upper",
                "top",
                "bottom",
                "std",
                "mean_omicron",
                "lower_omicron",
                "upper_omicron",
                "top_omicron",
                "bottom_omicron",
                "std_omicron",
                "rho",
                "rho_moving",
                "local",
                "imported",
            ]
        ],
        on=["date", "state"],
        how="inner",
    )

    ######### Create useable dataset #########
    # ACT and NT not in original estimates, need to extrapolated sorting keeps consistent 
    # with sort in data_by_state
    # Note that as we now consider the third wave for ACT, we include it in the third 
    # wave fitting only!
    states_to_fit_all_waves = sorted(
        ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
    )
    first_states = sorted(["NSW", "VIC", "QLD", "SA", "WA", "TAS"])
    fit_post_March = True
    ban = "2020-03-20"
    start_date = "2020-03-01"
    end_date = "2020-03-31"

    # data for the first wave
    first_date_range = {
        "NSW": pd.date_range(start="2020-03-01", end=end_date).values,
        "QLD": pd.date_range(start="2020-03-01", end=end_date).values,
        "SA": pd.date_range(start="2020-03-01", end=end_date).values,
        "TAS": pd.date_range(start="2020-03-01", end=end_date).values,
        "VIC": pd.date_range(start="2020-03-01", end=end_date).values,
        "WA": pd.date_range(start="2020-03-01", end=end_date).values,
    }

    # Second wave inputs
    sec_states = sorted(["NSW"])
    sec_start_date = "2020-06-01"
    sec_end_date = "2021-01-19"

    # choose dates for each state for sec wave
    sec_date_range = {
        "NSW": pd.date_range(start=sec_start_date, end="2021-01-19").values,
    }

    # Third wave inputs
    third_states = sorted(["NSW", "VIC", "ACT", "QLD", "SA", "TAS", "NT", "WA"])
    # Subtract the truncation days to avoid right truncation as we consider infection dates
    # and not symptom onset dates
    third_end_date = data_date - pd.Timedelta(days=truncation_days)

    # a different fitting end date to handle data issues with any particular states
    third_end_date_diff = data_date - pd.Timedelta(days=18 + 7 + 7)

    # choose dates for each state for third wave
    # Note that as we now consider the third wave for ACT, we include it in 
    # the third wave fitting only!
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
    predictors.remove("residential_7days")

    df["post_policy"] = (df.date >= ban).astype(int)

    dfX = df.loc[fit_mask].sort_values("date")
    df2X = df.loc[second_wave_mask].sort_values("date")
    df3X = df.loc[third_wave_mask].sort_values("date")

    dfX["is_first_wave"] = 0
    for state in first_states:
        dfX.loc[dfX.state == state, "is_first_wave"] = (
            dfX.loc[dfX.state == state]
            .date.isin(first_date_range[state])
            .astype(int)
            .values
        )

    df2X["is_sec_wave"] = 0
    for state in sec_states:
        df2X.loc[df2X.state == state, "is_sec_wave"] = (
            df2X.loc[df2X.state == state]
            .date.isin(sec_date_range[state])
            .astype(int)
            .values
        )

    # used to index what dates are featured in omicron AND third wave
    omicron_date_range = pd.date_range(start=omicron_start_date, end=third_end_date)

    df3X["is_third_wave"] = 0
    for state in third_states:
        df3X.loc[df3X.state == state, "is_third_wave"] = (
            df3X.loc[df3X.state == state]
            .date.isin(third_date_range[state])
            .astype(int)
            .values
        )
        # condition on being in third wave AND omicron
        df3X.loc[df3X.state == state, "is_omicron_wave"] = (
            (
                df3X.loc[df3X.state == state].date.isin(omicron_date_range)
                * df3X.loc[df3X.state == state].date.isin(third_date_range[state])
            )
            .astype(int)
            .values
        )

    data_by_state = {}
    sec_data_by_state = {}
    third_data_by_state = {}

    for value in ["mean", "std", "local", "imported"]:
        data_by_state[value] = pd.pivot(
            dfX[["state", value, "date"]],
            index="date",
            columns="state",
            values=value,
        ).sort_index(axis="columns")

        # account for dates pre pre second wave
        if df2X.loc[df2X.state == sec_states[0]].shape[0] == 0:
            print("making empty")
            sec_data_by_state[value] = pd.DataFrame(columns=sec_states).astype(float)
        else:
            sec_data_by_state[value] = pd.pivot(
                df2X[["state", value, "date"]],
                index="date",
                columns="state",
                values=value,
            ).sort_index(axis="columns")
        # account for dates pre pre third wave
        if df3X.loc[df3X.state == third_states[0]].shape[0] == 0:
            print("making empty")
            third_data_by_state[value] = pd.DataFrame(columns=third_states).astype(
                float
            )
        else:
            third_data_by_state[value] = pd.pivot(
                df3X[["state", value, "date"]],
                index="date",
                columns="state",
                values=value,
            ).sort_index(axis="columns")
            
    # now add in the summary stats for Omicron Reff 
    for value in ["mean_omicron", "std_omicron"]:
        if df3X.loc[df3X.state == third_states[0]].shape[0] == 0:
            print("making empty")
            third_data_by_state[value] = pd.DataFrame(columns=third_states).astype(
                float
            )
        else:
            third_data_by_state[value] = pd.pivot(
                df3X[["state", value, "date"]],
                index="date",
                columns="state",
                values=value,
            ).sort_index(axis="columns")
        

    # FIRST PHASE
    mobility_by_state = []
    mobility_std_by_state = []
    count_by_state = []
    respond_by_state = []
    mask_wearing_count_by_state = []
    mask_wearing_respond_by_state = []
    include_in_first_wave = []

    # filtering survey responses to dates before this wave fitting
    survey_respond = survey_respond_base.loc[: dfX.date.values[-1]]
    survey_counts = survey_counts_base.loc[: dfX.date.values[-1]]
    mask_wearing_respond = mask_wearing_respond_base.loc[: dfX.date.values[-1]]
    mask_wearing_counts = mask_wearing_counts_base.loc[: dfX.date.values[-1]]

    for state in first_states:
        mobility_by_state.append(dfX.loc[dfX.state == state, predictors].values / 100)
        mobility_std_by_state.append(
            dfX.loc[dfX.state == state, [val + "_std" for val in predictors]].values
            / 100
        )
        count_by_state.append(survey_counts.loc[start_date:end_date, state].values)
        respond_by_state.append(survey_respond.loc[start_date:end_date, state].values)
        mask_wearing_count_by_state.append(
            mask_wearing_counts.loc[start_date:end_date, state].values
        )
        mask_wearing_respond_by_state.append(
            mask_wearing_respond.loc[start_date:end_date, state].values
        )
        include_in_first_wave.append(
            dfX.loc[dfX.state == state, "is_first_wave"].values
        )

    # SECOND PHASE
    sec_mobility_by_state = []
    sec_mobility_std_by_state = []
    sec_count_by_state = []
    sec_respond_by_state = []
    sec_mask_wearing_count_by_state = []
    sec_mask_wearing_respond_by_state = []
    include_in_sec_wave = []

    # filtering survey responses to dates before this wave fitting
    survey_respond = survey_respond_base.loc[: df2X.date.values[-1]]
    survey_counts = survey_counts_base.loc[: df2X.date.values[-1]]
    mask_wearing_respond = mask_wearing_respond_base.loc[: df2X.date.values[-1]]
    mask_wearing_counts = mask_wearing_counts_base.loc[: df2X.date.values[-1]]

    for state in sec_states:
        sec_mobility_by_state.append(
            df2X.loc[df2X.state == state, predictors].values / 100
        )
        sec_mobility_std_by_state.append(
            df2X.loc[df2X.state == state, [val + "_std" for val in predictors]].values
            / 100
        )
        sec_count_by_state.append(
            survey_counts.loc[sec_start_date:sec_end_date, state].values
        )
        sec_respond_by_state.append(
            survey_respond.loc[sec_start_date:sec_end_date, state].values
        )
        sec_mask_wearing_count_by_state.append(
            mask_wearing_counts.loc[sec_start_date:sec_end_date, state].values
        )
        sec_mask_wearing_respond_by_state.append(
            mask_wearing_respond.loc[sec_start_date:sec_end_date, state].values
        )
        include_in_sec_wave.append(df2X.loc[df2X.state == state, "is_sec_wave"].values)

    # THIRD WAVE
    third_mobility_by_state = []
    third_mobility_std_by_state = []
    third_count_by_state = []
    third_respond_by_state = []
    third_mask_wearing_count_by_state = []
    third_mask_wearing_respond_by_state = []
    include_in_third_wave = []
    include_in_omicron_wave = []

    # filtering survey responses to dates before this wave fitting
    survey_respond = survey_respond_base.loc[: df3X.date.values[-1]]
    survey_counts = survey_counts_base.loc[: df3X.date.values[-1]]
    mask_wearing_respond = mask_wearing_respond_base.loc[: df3X.date.values[-1]]
    mask_wearing_counts = mask_wearing_counts_base.loc[: df3X.date.values[-1]]

    for state in third_states:
        third_mobility_by_state.append(
            df3X.loc[df3X.state == state, predictors].values / 100
        )
        third_mobility_std_by_state.append(
            df3X.loc[df3X.state == state, [val + "_std" for val in predictors]].values
            / 100
        )
        third_count_by_state.append(
            survey_counts.loc[third_start_date:third_end_date, state].values
        )
        third_respond_by_state.append(
            survey_respond.loc[third_start_date:third_end_date, state].values
        )
        third_mask_wearing_count_by_state.append(
            mask_wearing_counts.loc[third_start_date:third_end_date, state].values
        )
        third_mask_wearing_respond_by_state.append(
            mask_wearing_respond.loc[third_start_date:third_end_date, state].values
        )
        include_in_third_wave.append(
            df3X.loc[df3X.state == state, "is_third_wave"].values
        )
        include_in_omicron_wave.append(
            df3X.loc[df3X.state == state, "is_omicron_wave"].values
        )

    # policy boolean flag for after travel ban in each wave
    policy = dfX.loc[
        dfX.state == first_states[0], "post_policy"
    ]  # this is the post ban policy
    policy_sec_wave = [1] * df2X.loc[df2X.state == sec_states[0]].shape[0]
    policy_third_wave = [1] * df3X.loc[df3X.state == third_states[0]].shape[0]

    # read in the vaccination data
    delta_vaccination_by_state_array = process_vax_data_array(
        data_date=data_date,
        third_states=third_states,
        third_end_date=third_end_date,
        variant="Delta",
        print_latest_date_in_ts=True,
    )
    omicron_vaccination_by_state_array = process_vax_data_array(
        data_date=data_date,
        third_states=third_states,
        third_end_date=third_end_date,
        variant="Omicron",
    )

    # Make state by state arrays
    state_index = {state: i + 1 for i, state in enumerate(states_to_fit_all_waves)}

    # dates to apply alpha in the second wave (this won't allow for VIC to be added as 
    # the date_ranges are different)
    apply_alpha_sec_wave = (
        sec_date_range["NSW"] >= pd.to_datetime(alpha_start_date)
    ).astype(int)
    omicron_start_day = (
        pd.to_datetime(omicron_start_date) - pd.to_datetime(third_start_date)
    ).days
    omicron_dominance_day = (
        pd.to_datetime(omicron_dominance_date) - pd.to_datetime(third_start_date)
    ).days
    heterogeneity_start_day = (
        pd.to_datetime("2021-08-20") - pd.to_datetime(third_start_date)
    ).days
    
    # number of days we fit the average VE over 
    tau_vax_block_size = 7

    # get pop size array
    pop_size_array = []
    for s in states_to_fit_all_waves:
        pop_size_array.append(pop_sizes[s])

    # input data block for stan model
    input_data = {
        "j_total": len(states_to_fit_all_waves),
        
        "N": dfX.loc[dfX.state == first_states[0]].shape[0],
        "K": len(predictors),
        "j_first_wave": len(first_states),
        "Reff": data_by_state["mean"].values,
        "Mob": mobility_by_state,
        "Mob_std": mobility_std_by_state,
        "sigma2": data_by_state["std"].values ** 2,
        "policy": policy.values,
        "local": data_by_state["local"].values,
        "imported": data_by_state["imported"].values,
        
        "N_sec_wave": df2X.loc[df2X.state == sec_states[0]].shape[0],
        "j_sec_wave": len(sec_states),
        "Reff_sec_wave": sec_data_by_state["mean"].values,
        "Mob_sec_wave": sec_mobility_by_state,
        "Mob_sec_wave_std": sec_mobility_std_by_state,
        "sigma2_sec_wave": sec_data_by_state["std"].values ** 2,
        "policy_sec_wave": policy_sec_wave,
        "local_sec_wave": sec_data_by_state["local"].values,
        "imported_sec_wave": sec_data_by_state["imported"].values,
        "apply_alpha_sec_wave": apply_alpha_sec_wave,
        
        "N_third_wave": df3X.loc[df3X.state == third_states[0]].shape[0],
        "j_third_wave": len(third_states),
        "Reff_third_wave": third_data_by_state["mean"].values,
        "Reff_omicron_wave": third_data_by_state["mean_omicron"].values,
        "Mob_third_wave": third_mobility_by_state,
        "Mob_third_wave_std": third_mobility_std_by_state,
        "sigma2_third_wave": third_data_by_state["std"].values ** 2,
        "sigma2_omicron_wave": third_data_by_state["std_omicron"].values ** 2,
        "policy_third_wave": policy_third_wave,
        "local_third_wave": third_data_by_state["local"].values,
        "imported_third_wave": third_data_by_state["imported"].values,
        
        "count_md": count_by_state,
        "respond_md": respond_by_state,
        "count_md_sec_wave": sec_count_by_state,
        "respond_md_sec_wave": sec_respond_by_state,
        "count_md_third_wave": third_count_by_state,
        "respond_md_third_wave": third_respond_by_state,
        
        "count_masks": mask_wearing_count_by_state,
        "respond_masks": mask_wearing_respond_by_state,
        "count_masks_sec_wave": sec_mask_wearing_count_by_state,
        "respond_masks_sec_wave": sec_mask_wearing_respond_by_state,
        "count_masks_third_wave": third_mask_wearing_count_by_state,
        "respond_masks_third_wave": third_mask_wearing_respond_by_state,
        
        "map_to_state_index_first": [state_index[state] for state in first_states],
        "map_to_state_index_sec": [state_index[state] for state in sec_states],
        "map_to_state_index_third": [state_index[state] for state in third_states],
        
        "total_N_p_sec": sum([sum(x) for x in include_in_sec_wave]).item(),
        "total_N_p_third": sum([sum(x) for x in include_in_third_wave]).item(),
        
        "include_in_first_wave": include_in_first_wave,
        "include_in_sec_wave": include_in_sec_wave,
        "include_in_third_wave": include_in_third_wave,
        
        "pos_starts_sec": np.cumsum([sum(x) for x in include_in_sec_wave])
            .astype(int)
            .tolist(),
        "pos_starts_third": np.cumsum([sum(x) for x in include_in_third_wave])
            .astype(int)
            .tolist(),
        
        "ve_delta_data": delta_vaccination_by_state_array,
        "ve_omicron_data": omicron_vaccination_by_state_array,
        
        "omicron_start_day": omicron_start_day,
        "omicron_dominance_day": omicron_dominance_day,
        "include_in_omicron_wave": include_in_omicron_wave,
        "total_N_p_third_omicron": int(
            sum([sum(x) for x in include_in_omicron_wave]).item()
        ),
        "pos_starts_third_omicron": np.cumsum([sum(x) for x in include_in_omicron_wave])
            .astype(int)
            .tolist(),
        'tau_vax_block_size': tau_vax_block_size, 
        'total_N_p_third_blocks': int(
            sum([int(ceil(sum(x)/tau_vax_block_size)) for x in include_in_third_wave])
        ),
        'pos_starts_third_blocks': np.cumsum(
            [int(ceil(sum(x)/tau_vax_block_size)) for x in include_in_third_wave]
        ).astype(int),
        'total_N_p_third_omicron_blocks': int(
            sum([int(ceil(sum(x)/tau_vax_block_size)) for x in include_in_omicron_wave])
        ),
        'pos_starts_third_omicron_blocks': np.cumsum(
            [int(ceil(sum(x)/tau_vax_block_size)) for x in include_in_omicron_wave]
        ).astype(int),
        "pop_size_array": pop_size_array,
        "heterogeneity_start_day": heterogeneity_start_day,
        "p_detect_delta": p_detect_delta, 
        "p_detect_omicron": p_detect_omicron, 
    }

    # dump the dictionary to a json filex
    a_file = open("results/stan_input_data.pkl", "wb")
    pickle.dump(input_data, a_file)
    a_file.close()

    return None


def run_stan(
    data_date, 
    num_chains=4, 
    num_samples=1000, 
    num_warmup_samples=500, 
    custom_file_name="",
):
    """
    Read the input_data.json in and run the stan model.
    """

    data_date = pd.to_datetime(data_date)

    # read in the input data as a dictionary
    a_file = open("results/stan_input_data.pkl", "rb")
    input_data = pickle.load(a_file)
    a_file.close()

    # make results and figs dir
    figs_dir = (
        "figs/stan_fit/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/"
        + custom_file_name 
        + "/"
    )
    
    results_dir = (
        "results/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/" 
        + custom_file_name 
        + "/"
    )
    
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # to run the inference set run_inference to True in params
    if run_inference:

        # import the stan model as a string
        with open("TP_model/fit_and_forecast/rho_model_gamma.stan") as f:
            rho_model_gamma = f.read()

        # compile the stan model
        posterior = stan.build(rho_model_gamma, data=input_data)
        fit = posterior.sample(
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup_samples,
        )
        
        df_fit = fit.to_frame()
        df_fit.to_csv(
            results_dir
            + "posterior_sample_" 
            + data_date.strftime("%Y-%m-%d") 
            + ".csv"
        )
        
        filename = "stan_posterior_fit" + data_date.strftime("%Y-%m-%d") + ".txt"
        with open(figs_dir + filename, "w") as f:
            print(
                az.summary(
                    fit,
                    var_names=[
                        "bet",
                        "R_I",
                        "R_I_omicron",
                        "R_L",
                        "R_Li",
                        "theta_md",
                        "sig",
                        "voc_effect_alpha",
                        "voc_effect_delta",
                        "voc_effect_omicron",
                        "susceptible_depletion_factor",
                    ],
                ),
                file=f,
            )

    return None


def plot_and_save_posterior_samples(data_date, custom_file_name=""):
    """
    Runs the full suite of plotting.
    """

    data_date = pd.to_datetime(data_date)  # Define data date
    figs_dir = (
        "figs/stan_fit/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/"
        + custom_file_name 
        + "/"
    )

    # read in the posterior sample
    samples_mov_gamma = pd.read_csv(
        "results/"
        + data_date.strftime("%Y-%m-%d") 
        + "/"
        + custom_file_name 
        + "/"
        + "posterior_sample_" 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )

    # * Note: 2020-09-09 won't work (for some reason)

    ######### Read in microdistancing (md) surveys #########
    surveys = pd.DataFrame()
    path = "data/md/Barometer wave*.csv"
    for file in glob.glob(path):
        surveys = surveys.append(pd.read_csv(file, parse_dates=["date"]))

    surveys = surveys.sort_values(by="date")
    print("Latest Microdistancing survey is {}".format(surveys.date.values[-1]))

    surveys["state"] = surveys["state"].map(states_initials).fillna(surveys["state"])
    surveys["proportion"] = surveys["count"] / surveys.respondents
    surveys.date = pd.to_datetime(surveys.date)

    always = surveys.loc[surveys.response == "Always"].set_index(["state", "date"])
    always = always.unstack(["state"])
    # If you get an error here saying 'cannot create a new series when the index is not unique', 
    # then you have a duplicated md file.

    idx = pd.date_range("2020-03-01", pd.to_datetime("today"))
    always = always.reindex(idx, fill_value=np.nan)
    always.index.name = "date"

    # fill back to earlier and between weeks.
    # Assume survey on day x applies for all days up to x - 6
    always = always.fillna(method="bfill")
    # assume values continue forward if survey hasn't completed
    always = always.fillna(method="ffill")
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
    survey_counts_base = (
        pd.pivot_table(data=always, index="date", columns="state", values="count")
        .drop(["Australia", "Other"], axis=1)
        .astype(int)
    )

    survey_respond_base = (
        pd.pivot_table(data=always, index="date", columns="state", values="respondents")
        .drop(["Australia", "Other"], axis=1)
        .astype(int)
    )

    ## read in and process mask wearing data
    mask_wearing = pd.DataFrame()
    path = "data/face_coverings/face_covering_*_.csv"
    for file in glob.glob(path):
        mask_wearing = mask_wearing.append(pd.read_csv(file, parse_dates=["date"]))

    mask_wearing = mask_wearing.sort_values(by="date")
    print("Latest Mask wearing survey is {}".format(mask_wearing.date.values[-1]))

    mask_wearing["state"] = (
        mask_wearing["state"].map(states_initials).fillna(mask_wearing["state"])
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
    # assume values continue forward if survey hasn't completed
    mask_wearing_always = mask_wearing_always.fillna(method="ffill")
    mask_wearing_always = mask_wearing_always.stack(["state"])

    # Zero out before first survey 20th March
    mask_wearing_always = mask_wearing_always.reset_index().set_index("date")
    mask_wearing_always.loc[:"2020-03-20", "count"] = 0
    mask_wearing_always.loc[:"2020-03-20", "respondents"] = 0
    mask_wearing_always.loc[:"2020-03-20", "proportion"] = 0

    mask_wearing_X = pd.pivot_table(
        data=mask_wearing_always, index="date", columns="state", values="proportion"
    )
    mask_wearing_counts_base = pd.pivot_table(
        data=mask_wearing_always, index="date", columns="state", values="count"
    ).astype(int)
    mask_wearing_respond_base = pd.pivot_table(
        data=mask_wearing_always, index="date", columns="state", values="respondents"
    ).astype(int)

    df_Reff = pd.read_csv(
        "results/EpyReff/Reff_delta" + data_date.strftime("%Y-%m-%d") + "tau_5.csv",
        parse_dates=["INFECTION_DATES"],
    )
    df_Reff["date"] = df_Reff.INFECTION_DATES
    df_Reff["state"] = df_Reff.STATE
    
    df_Reff_omicron = pd.read_csv(
        "results/EpyReff/Reff_omicron" + data_date.strftime("%Y-%m-%d") + "tau_5.csv",
        parse_dates=["INFECTION_DATES"],
    )
    df_Reff_omicron["date"] = df_Reff_omicron.INFECTION_DATES
    df_Reff_omicron["state"] = df_Reff_omicron.STATE
    
    # relabel some of the columns to avoid replication in the merged dataframe 
    col_names_replace = {
        "mean": "mean_omicron",
        "lower": "lower_omicron",
        "upper": "upper_omicron",
        "top": "top_omicron",
        "bottom": "bottom_omicron",
        "std": "std_omicron",
    }

    df_Reff_omicron.rename(col_names_replace, axis=1, inplace=True)

    # read in NNDSS/linelist data
    # If this errors it may be missing a leading zero on the date.
    df_state = read_in_cases(
        case_file_date=data_date.strftime("%d%b%Y"),
        apply_delay_at_read=True,
        apply_inc_at_read=True,
    )

    df_Reff = df_Reff.merge(
        df_state,
        how="left",
        left_on=["state", "date"],
        right_on=["STATE", "date_inferred"],
    )  # how = left to use Reff days, NNDSS missing dates

    # merge in the omicron stuff 
    df_Reff = df_Reff.merge(
        df_Reff_omicron, 
        how="left", 
        left_on=["state", "date"],
        right_on=["state", "date"],
    )
    
    df_Reff["rho_moving"] = df_Reff.groupby(["state"])["rho"].transform(
        lambda x: x.rolling(7, 1).mean()
    )  # minimum number of 1

    # some days have no cases, so need to fillna
    df_Reff["rho_moving"] = df_Reff.rho_moving.fillna(method="bfill")

    # counts are already aligned with infection date by subtracting a random incubation period
    df_Reff["local"] = df_Reff.local.fillna(0)
    df_Reff["imported"] = df_Reff.imported.fillna(0)

    ######### Read in Google mobility results #########
    sys.path.insert(0, "../")

    df_google = read_in_google(local=not download_google_automatically, moving=True)
    df = df_google.merge(
        df_Reff[
            [
                "date",
                "state",
                "mean",
                "lower",
                "upper",
                "top",
                "bottom",
                "std",
                "mean_omicron",
                "lower_omicron",
                "upper_omicron",
                "top_omicron",
                "bottom_omicron",
                "std_omicron",
                "rho",
                "rho_moving",
                "local",
                "imported",
            ]
        ],
        on=["date", "state"],
        how="inner",
    )

    # ACT and NT not in original estimates, need to extrapolated sorting keeps consistent 
    # with sort in data_by_state
    # Note that as we now consider the third wave for ACT, we include it in the third 
    # wave fitting only!
    states_to_fit_all_waves = sorted(
        ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
    )

    first_states = sorted(["NSW", "VIC", "QLD", "SA", "WA", "TAS"])
    fit_post_March = True
    ban = "2020-03-20"
    start_date = "2020-03-01"
    end_date = "2020-03-31"

    # data for the first wave
    first_date_range = {
        "NSW": pd.date_range(start="2020-03-01", end=end_date).values,
        "QLD": pd.date_range(start="2020-03-01", end=end_date).values,
        "SA": pd.date_range(start="2020-03-01", end=end_date).values,
        "TAS": pd.date_range(start="2020-03-01", end=end_date).values,
        "VIC": pd.date_range(start="2020-03-01", end=end_date).values,
        "WA": pd.date_range(start="2020-03-01", end=end_date).values,
    }

    # Second wave inputs
    # sec_states = sorted(['NSW', 'VIC'])
    sec_states = sorted(["NSW"])
    sec_start_date = "2020-06-01"
    sec_end_date = "2021-01-19"

    # choose dates for each state for sec wave
    sec_date_range = {
        "NSW": pd.date_range(start=sec_start_date, end="2021-01-19").values,
        # 'VIC': pd.date_range(start=sec_start_date, end='2020-10-28').values,
    }

    # Third wave inputs
    third_states = sorted(["NSW", "VIC", "ACT", "QLD", "SA", "TAS", "NT", "WA"])
    # third_states = sorted(['NSW', 'VIC', 'ACT', 'QLD', 'SA', 'NT'])
    # Subtract the truncation days to avoid right truncation as we consider infection dates
    # and not symptom onset dates
    third_end_date = data_date - pd.Timedelta(days=truncation_days)

    # to handle SA data issues
    # third_end_date_diff = data_date - pd.Timedelta(days=18 + 7 + 7)a

    # choose dates for each state for third wave
    # Note that as we now consider the third wave for ACT, we include it in the third 
    # wave fitting only!
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
    predictors.remove("residential_7days")

    df["post_policy"] = (df.date >= ban).astype(int)

    dfX = df.loc[fit_mask].sort_values("date")
    df2X = df.loc[second_wave_mask].sort_values("date")
    df3X = df.loc[third_wave_mask].sort_values("date")

    dfX["is_first_wave"] = 0
    for state in first_states:
        dfX.loc[dfX.state == state, "is_first_wave"] = (
            dfX.loc[dfX.state == state]
            .date.isin(first_date_range[state])
            .astype(int)
            .values
        )

    df2X["is_sec_wave"] = 0
    for state in sec_states:
        df2X.loc[df2X.state == state, "is_sec_wave"] = (
            df2X.loc[df2X.state == state]
            .date.isin(sec_date_range[state])
            .astype(int)
            .values
        )

    # used to index what dates are also featured in omicron
    omicron_date_range = pd.date_range(start=omicron_start_date, end=third_end_date)

    df3X["is_third_wave"] = 0
    for state in third_states:
        df3X.loc[df3X.state == state, "is_third_wave"] = (
            df3X.loc[df3X.state == state]
            .date.isin(third_date_range[state])
            .astype(int)
            .values
        )
        # condition on being in third wave AND omicron
        df3X.loc[df3X.state == state, "is_omicron_wave"] = (
            (
                df3X.loc[df3X.state == state].date.isin(omicron_date_range)
                * df3X.loc[df3X.state == state].date.isin(third_date_range[state])
            )
            .astype(int)
            .values
        )

    data_by_state = {}
    sec_data_by_state = {}
    third_data_by_state = {}

    for value in ["mean", "std", "local", "imported"]:
        data_by_state[value] = pd.pivot(
            dfX[["state", value, "date"]], index="date", columns="state", values=value
        ).sort_index(axis="columns")

        # account for dates pre pre second wave
        if df2X.loc[df2X.state == sec_states[0]].shape[0] == 0:
            print("making empty")
            sec_data_by_state[value] = pd.DataFrame(columns=sec_states).astype(float)
        else:
            sec_data_by_state[value] = pd.pivot(
                df2X[["state", value, "date"]],
                index="date",
                columns="state",
                values=value,
            ).sort_index(axis="columns")
        # account for dates pre pre third wave
        if df3X.loc[df3X.state == third_states[0]].shape[0] == 0:
            print("making empty")
            third_data_by_state[value] = pd.DataFrame(columns=third_states).astype(
                float
            )
        else:
            third_data_by_state[value] = pd.pivot(
                df3X[["state", value, "date"]],
                index="date",
                columns="state",
                values=value,
            ).sort_index(axis="columns")
            
            
    # now add in the summary stats for Omicron Reff 
    for value in ["mean_omicron", "std_omicron"]:
        if df3X.loc[df3X.state == third_states[0]].shape[0] == 0:
            print("making empty")
            third_data_by_state[value] = pd.DataFrame(columns=third_states).astype(
                float
            )
        else:
            third_data_by_state[value] = pd.pivot(
                df3X[["state", value, "date"]],
                index="date",
                columns="state",
                values=value,
            ).sort_index(axis="columns")

    # FIRST PHASE
    mobility_by_state = []
    mobility_std_by_state = []
    count_by_state = []
    respond_by_state = []
    mask_wearing_count_by_state = []
    mask_wearing_respond_by_state = []
    include_in_first_wave = []

    # filtering survey responses to dates before this wave fitting
    survey_respond = survey_respond_base.loc[: dfX.date.values[-1]]
    survey_counts = survey_counts_base.loc[: dfX.date.values[-1]]
    mask_wearing_respond = mask_wearing_respond_base.loc[: dfX.date.values[-1]]
    mask_wearing_counts = mask_wearing_counts_base.loc[: dfX.date.values[-1]]

    for state in first_states:
        mobility_by_state.append(dfX.loc[dfX.state == state, predictors].values / 100)
        mobility_std_by_state.append(
            dfX.loc[dfX.state == state, [val + "_std" for val in predictors]].values
            / 100
        )
        count_by_state.append(survey_counts.loc[start_date:end_date, state].values)
        respond_by_state.append(survey_respond.loc[start_date:end_date, state].values)
        mask_wearing_count_by_state.append(
            mask_wearing_counts.loc[start_date:end_date, state].values
        )
        mask_wearing_respond_by_state.append(
            mask_wearing_respond.loc[start_date:end_date, state].values
        )
        include_in_first_wave.append(
            dfX.loc[dfX.state == state, "is_first_wave"].values
        )

    # SECOND PHASE
    sec_mobility_by_state = []
    sec_mobility_std_by_state = []
    sec_count_by_state = []
    sec_respond_by_state = []
    sec_mask_wearing_count_by_state = []
    sec_mask_wearing_respond_by_state = []
    include_in_sec_wave = []

    # filtering survey responses to dates before this wave fitting
    survey_respond = survey_respond_base.loc[: df2X.date.values[-1]]
    survey_counts = survey_counts_base.loc[: df2X.date.values[-1]]
    mask_wearing_respond = mask_wearing_respond_base.loc[: df2X.date.values[-1]]
    mask_wearing_counts = mask_wearing_counts_base.loc[: df2X.date.values[-1]]

    for state in sec_states:
        sec_mobility_by_state.append(
            df2X.loc[df2X.state == state, predictors].values / 100
        )
        sec_mobility_std_by_state.append(
            df2X.loc[df2X.state == state, [val + "_std" for val in predictors]].values
            / 100
        )
        sec_count_by_state.append(
            survey_counts.loc[sec_start_date:sec_end_date, state].values
        )
        sec_respond_by_state.append(
            survey_respond.loc[sec_start_date:sec_end_date, state].values
        )
        sec_mask_wearing_count_by_state.append(
            mask_wearing_counts.loc[sec_start_date:sec_end_date, state].values
        )
        sec_mask_wearing_respond_by_state.append(
            mask_wearing_respond.loc[sec_start_date:sec_end_date, state].values
        )
        include_in_sec_wave.append(df2X.loc[df2X.state == state, "is_sec_wave"].values)

    # THIRD WAVE
    third_mobility_by_state = []
    third_mobility_std_by_state = []
    third_count_by_state = []
    third_respond_by_state = []
    third_mask_wearing_count_by_state = []
    third_mask_wearing_respond_by_state = []
    include_in_third_wave = []
    include_in_omicron_wave = []

    # filtering survey responses to dates before this wave fitting
    survey_respond = survey_respond_base.loc[: df3X.date.values[-1]]
    survey_counts = survey_counts_base.loc[: df3X.date.values[-1]]
    mask_wearing_respond = mask_wearing_respond_base.loc[: df3X.date.values[-1]]
    mask_wearing_counts = mask_wearing_counts_base.loc[: df3X.date.values[-1]]

    for state in third_states:
        third_mobility_by_state.append(
            df3X.loc[df3X.state == state, predictors].values / 100
        )
        third_mobility_std_by_state.append(
            df3X.loc[df3X.state == state, [val + "_std" for val in predictors]].values
            / 100
        )
        third_count_by_state.append(
            survey_counts.loc[third_start_date:third_end_date, state].values
        )
        third_respond_by_state.append(
            survey_respond.loc[third_start_date:third_end_date, state].values
        )
        third_mask_wearing_count_by_state.append(
            mask_wearing_counts.loc[third_start_date:third_end_date, state].values
        )
        third_mask_wearing_respond_by_state.append(
            mask_wearing_respond.loc[third_start_date:third_end_date, state].values
        )
        include_in_third_wave.append(
            df3X.loc[df3X.state == state, "is_third_wave"].values
        )
        include_in_omicron_wave.append(
            df3X.loc[df3X.state == state, "is_omicron_wave"].values
        )

    # Make state by state arrays
    state_index = {state: i for i, state in enumerate(states_to_fit_all_waves)}

    # get pop size array
    pop_size_array = []
    for s in states_to_fit_all_waves:
        pop_size_array.append(pop_sizes[s])

    # First phase
    # rho calculated at data entry
    if isinstance(df_state.index, pd.MultiIndex):
        df_state = df_state.reset_index()

    states = sorted(["NSW", "QLD", "VIC", "TAS", "SA", "WA", "ACT", "NT"])
    fig, ax = plt.subplots(figsize=(24, 9), ncols=len(states), sharey=True)

    states_to_fitd = {state: i + 1 for i, state in enumerate(first_states)}

    for i, state in enumerate(states):
        if state in first_states:
            dates = df_Reff.loc[
                (df_Reff.date >= start_date)
                & (df_Reff.state == state)
                & (df_Reff.date <= end_date)
            ].date
            rho_samples = samples_mov_gamma[
                [
                    "brho." + str(j + 1) + "." + str(states_to_fitd[state])
                    for j in range(dfX.loc[dfX.state == first_states[0]].shape[0])
                ]
            ]
            ax[i].plot(dates, rho_samples.median(), label="fit", color="C0")
            ax[i].fill_between(
                dates,
                rho_samples.quantile(0.25),
                rho_samples.quantile(0.75),
                color="C0",
                alpha=0.4,
            )

            ax[i].fill_between(
                dates,
                rho_samples.quantile(0.05),
                rho_samples.quantile(0.95),
                color="C0",
                alpha=0.4,
            )
        else:
            sns.lineplot(
                x="date_inferred",
                y="rho",
                data=df_state.loc[
                    (df_state.date_inferred >= start_date)
                    & (df_state.STATE == state)
                    & (df_state.date_inferred <= end_date)
                ],
                ax=ax[i],
                color="C1",
                label="data",
            )

        sns.lineplot(
            x="date",
            y="rho",
            data=df_Reff.loc[
                (df_Reff.date >= start_date)
                & (df_Reff.state == state)
                & (df_Reff.date <= end_date)
            ],
            ax=ax[i],
            color="C1",
            label="data",
        )
        sns.lineplot(
            x="date",
            y="rho_moving",
            data=df_Reff.loc[
                (df_Reff.date >= start_date)
                & (df_Reff.state == state)
                & (df_Reff.date <= end_date)
            ],
            ax=ax[i],
            color="C2",
            label="moving",
        )

        dates = dfX.loc[dfX.state == first_states[0]].date

        ax[i].tick_params("x", rotation=90)
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[i].set_title(state)

    ax[0].set_ylabel("Proportion of imported cases")
    plt.legend()
    plt.savefig(
        figs_dir + data_date.strftime("%Y-%m-%d") + "rho_first_phase.png", dpi=144
    )

    # Second phase
    if df2X.shape[0] > 0:
        fig, ax = plt.subplots(
            figsize=(24, 9), ncols=len(sec_states), sharey=True, squeeze=False
        )
        states_to_fitd = {state: i + 1 for i, state in enumerate(sec_states)}
        pos = 0
        for i, state in enumerate(sec_states):
            # Google mobility only up to a certain date, so take only up to that value
            dates = df2X.loc[
                (df2X.state == state) & (df2X.is_sec_wave == 1)
            ].date.values
            rho_samples = samples_mov_gamma[
                [
                    "brho_sec_wave." + str(j + 1)
                    for j in range(
                        pos, pos + df2X.loc[df2X.state == state].is_sec_wave.sum()
                    )
                ]
            ]
            pos = pos + df2X.loc[df2X.state == state].is_sec_wave.sum()

            ax[0, i].plot(dates, rho_samples.median(), label="fit", color="C0")
            ax[0, i].fill_between(
                dates,
                rho_samples.quantile(0.25),
                rho_samples.quantile(0.75),
                color="C0",
                alpha=0.4,
            )

            ax[0, i].fill_between(
                dates,
                rho_samples.quantile(0.05),
                rho_samples.quantile(0.95),
                color="C0",
                alpha=0.4,
            )

            sns.lineplot(
                x="date_inferred",
                y="rho",
                data=df_state.loc[
                    (df_state.date_inferred >= sec_start_date)
                    & (df_state.STATE == state)
                    & (df_state.date_inferred <= sec_end_date)
                ],
                ax=ax[0, i],
                color="C1",
                label="data",
            )
            sns.lineplot(
                x="date",
                y="rho",
                data=df_Reff.loc[
                    (df_Reff.date >= sec_start_date)
                    & (df_Reff.state == state)
                    & (df_Reff.date <= sec_end_date)
                ],
                ax=ax[0, i],
                color="C1",
                label="data",
            )
            sns.lineplot(
                x="date",
                y="rho_moving",
                data=df_Reff.loc[
                    (df_Reff.date >= sec_start_date)
                    & (df_Reff.state == state)
                    & (df_Reff.date <= sec_end_date)
                ],
                ax=ax[0, i],
                color="C2",
                label="moving",
            )

            dates = dfX.loc[dfX.state == sec_states[0]].date

            ax[0, i].tick_params("x", rotation=90)
            ax[0, i].xaxis.set_major_locator(plt.MaxNLocator(4))
            ax[0, i].set_title(state)

        ax[0, 0].set_ylabel("Proportion of imported cases")
        plt.legend()
        plt.savefig(
            figs_dir + data_date.strftime("%Y-%m-%d") + "rho_sec_phase.png", dpi=144
        )

    df_rho_third_all_states = pd.DataFrame()
    df_rho_third_tmp = pd.DataFrame()
    # Third  phase
    if df3X.shape[0] > 0:
        fig, ax = plt.subplots(
            figsize=(9, 24), nrows=len(third_states), sharex=True, squeeze=False
        )
        states_to_fitd = {state: i + 1 for i, state in enumerate(third_states)}
        pos = 0
        for i, state in enumerate(third_states):
            # Google mobility only up to a certain date, so take only up to that value
            dates = df3X.loc[
                (df3X.state == state) & (df3X.is_third_wave == 1)
            ].date.values
            rho_samples = samples_mov_gamma[
                [
                    "brho_third_wave." + str(j + 1)
                    for j in range(
                        pos, pos + df3X.loc[df3X.state == state].is_third_wave.sum()
                    )
                ]
            ]
            pos = pos + df3X.loc[df3X.state == state].is_third_wave.sum()
            
            df_rho_third_tmp = rho_samples.T
            df_rho_third_tmp["date"] = dates           
            df_rho_third_tmp["state"] = state
            
            df_rho_third_all_states = pd.concat([df_rho_third_all_states, df_rho_third_tmp])

            ax[i, 0].plot(dates, rho_samples.median(), label="fit", color="C0")
            ax[i, 0].fill_between(
                dates,
                rho_samples.quantile(0.25),
                rho_samples.quantile(0.75),
                color="C0",
                alpha=0.4,
            )

            ax[i, 0].fill_between(
                dates,
                rho_samples.quantile(0.05),
                rho_samples.quantile(0.95),
                color="C0",
                alpha=0.4,
            )

            sns.lineplot(
                x="date_inferred",
                y="rho",
                data=df_state.loc[
                    (df_state.date_inferred >= third_start_date)
                    & (df_state.STATE == state)
                    & (df_state.date_inferred <= third_end_date)
                ],
                ax=ax[i, 0],
                color="C1",
                label="data",
            )
            sns.lineplot(
                x="date",
                y="rho",
                data=df_Reff.loc[
                    (df_Reff.date >= third_start_date)
                    & (df_Reff.state == state)
                    & (df_Reff.date <= third_end_date)
                ],
                ax=ax[i, 0],
                color="C1",
                label="data",
            )
            sns.lineplot(
                x="date",
                y="rho_moving",
                data=df_Reff.loc[
                    (df_Reff.date >= third_start_date)
                    & (df_Reff.state == state)
                    & (df_Reff.date <= third_end_date)
                ],
                ax=ax[i, 0],
                color="C2",
                label="moving",
            )

            dates = dfX.loc[dfX.state == third_states[0]].date

            ax[i, 0].tick_params("x", rotation=90)
            ax[i, 0].xaxis.set_major_locator(plt.MaxNLocator(4))
            ax[i, 0].set_title(state)
            ax[i, 0].set_ylabel("Proportion of imported cases")

        plt.legend()
        plt.savefig(
            figs_dir + data_date.strftime("%Y-%m-%d") + "rho_third_phase.png", dpi=144,
        )

    df_rho_third_all_states.to_csv(
        "results/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/"
        + custom_file_name 
        + "/"
        + "rho_samples" 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )

    # plotting 
    fig, ax = plt.subplots(figsize=(12, 9))

    # sample from the priors for RL and RI
    samples_mov_gamma["R_L_prior"] = np.random.gamma(
        1.8 * 1.8 / 0.05, 0.05 / 1.8, size=samples_mov_gamma.shape[0]
    )
    samples_mov_gamma["R_I_prior"] = np.random.gamma(
        0.5 ** 2 / 0.2, 0.2 / 0.5, size=samples_mov_gamma.shape[0]
    )

    samples_mov_gamma["R_L_national"] = np.random.gamma(
        samples_mov_gamma.R_L.values ** 2 / samples_mov_gamma.sig.values,
        samples_mov_gamma.sig.values / samples_mov_gamma.R_L.values,
    )

    sns.violinplot(
        x="variable",
        y="value",
        data=pd.melt(
            samples_mov_gamma[[col for col in samples_mov_gamma if "R" in col]]
        ),
        ax=ax,
        cut=0,
    )

    ax.set_yticks(
        [1],
        minor=True,
    )
    ax.set_yticks([0, 2, 3], minor=False)
    ax.set_yticklabels([0, 2, 3], minor=False)
    ax.set_ylim((0, 3))
    # state labels in alphabetical
    ax.set_xticklabels(
        [
            "R_I",
            "R_I_omicron",
            "R_L0 mean",
            "R_L0 ACT",
            "R_L0 NSW",
            "R_L0 NT",
            "R_L0 QLD",
            "R_L0 SA",
            "R_L0 TAS",
            "R_L0 VIC",
            "R_L0 WA",
            "R_L0 prior",
            "R_I prior",
            "R_L0 national",
        ]
    )
    ax.set_xlabel("")
    ax.set_ylabel("Effective reproduction number")
    ax.tick_params("x", rotation=90)
    ax.yaxis.grid(which="minor", linestyle="--", color="black", linewidth=2)
    plt.tight_layout()
    plt.savefig(figs_dir + data_date.strftime("%Y-%m-%d") + "R_priors.png", dpi=144)

    # Making a new figure that doesn't include the priors
    fig, ax = plt.subplots(figsize=(12, 9))

    small_plot_cols = ["R_Li." + str(i) for i in range(1, 9)] + ["R_I"]

    sns.violinplot(
        x="variable",
        y="value",
        data=pd.melt(samples_mov_gamma[small_plot_cols]),
        ax=ax,
        cut=0,
    )

    ax.set_yticks(
        [1],
        minor=True,
    )
    ax.set_yticks([0, 2, 3], minor=False)
    ax.set_yticklabels([0, 2, 3], minor=False)
    ax.set_ylim((0, 3))
    # state labels in alphabetical
    ax.set_xticklabels(
        [
            "$R_L0$ ACT",
            "$R_L0$ NSW",
            "$R_L0$ NT",
            "$R_L0$ QLD",
            "$R_L0$ SA",
            "$R_L0$ TAS",
            "$R_L0$ VIC",
            "$R_L0$ WA",
            "$R_I$",
        ]
    )
    ax.tick_params("x", rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel("Effective reproduction number")
    ax.yaxis.grid(which="minor", linestyle="--", color="black", linewidth=2)
    plt.tight_layout()
    plt.savefig(
        figs_dir + data_date.strftime("%Y-%m-%d") + "R_priors_(without_priors).png",
        dpi=288,
    )

    # Making a new figure that doesn't include the priors
    fig, ax = plt.subplots(figsize=(12, 9))
    samples_mov_gamma["voc_effect_third_prior"] = np.random.gamma(
        1.5 * 1.5 / 0.05, 0.05 / 1.5, size=samples_mov_gamma.shape[0]
    )
    small_plot_cols = [
        "voc_effect_third_prior",
        "voc_effect_delta",
        "voc_effect_omicron",
    ]

    sns.violinplot(
        x="variable",
        y="value",
        data=pd.melt(samples_mov_gamma[small_plot_cols]),
        ax=ax,
        cut=0,
    )

    ax.set_yticks([1], minor=True)
    # ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3], minor=False)
    # ax.set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3], minor=False)
    # ax.set_ylim((0, 1))
    # state labels in alphabetical
    ax.set_xticklabels(["VoC (prior)", "VoC (Delta)", "VoC (Omicron)"])
    # ax.tick_params('x', rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel("value")
    ax.yaxis.grid(which="minor", linestyle="--", color="black", linewidth=2)
    plt.tight_layout()
    plt.savefig(
        figs_dir + data_date.strftime("%Y-%m-%d") + "voc_effect_posteriors.png",
        dpi=288,
    )

    posterior = samples_mov_gamma[["bet." + str(i + 1) for i in range(len(predictors))]]

    split = True
    md = "power"  # samples_mov_gamma.md.values

    posterior.columns = [val for val in predictors]
    long = pd.melt(posterior)

    fig, ax2 = plt.subplots(figsize=(12, 9))

    ax2 = sns.violinplot(x="variable", y="value", data=long, ax=ax2, color="C0")

    ax2.plot([0] * len(predictors), linestyle="dashed", alpha=0.6, color="grey")
    ax2.tick_params(axis="x", rotation=90)

    ax2.set_title("Coefficients of mobility indices")
    ax2.set_xlabel("Social mobility index")
    ax2.set_xticklabels([var[:-6] for var in predictors])
    ax2.set_xticklabels(
        [
            "Retail and Recreation",
            "Grocery and Pharmacy",
            "Parks",
            "Transit Stations",
            "Workplaces",
        ]
    )
    ax2.tick_params("x", rotation=15)
    plt.tight_layout()

    plt.savefig(
        figs_dir + data_date.strftime("%Y-%m-%d") + "mobility_posteriors.png",
        dpi=288,
    )

    # plot the TP's
    RL_by_state = {
        state: samples_mov_gamma["R_Li." + str(i + 1)].values
        for state, i in state_index.items()
    }
    ax3 = predict_plot(
        samples_mov_gamma,
        df.loc[(df.date >= start_date) & (df.date <= end_date)],
        moving=True,
        grocery=True,
        rho=first_states,
    )
    for ax in ax3:
        for a in ax:
            a.set_ylim((0, 2.5))
            a.set_xlim((pd.to_datetime(start_date), pd.to_datetime(end_date)))

    plt.savefig(
        figs_dir + data_date.strftime("%Y-%m-%d") + "Reff_first_phase.png",
        dpi=144,
    )

    if df2X.shape[0] > 0:
        df["is_sec_wave"] = 0
        for state in sec_states:
            df.loc[df.state == state, "is_sec_wave"] = (
                df.loc[df.state == state]
                .date.isin(sec_date_range[state])
                .astype(int)
                .values
            )
        # plot only if there is second phase data - have to have second_phase=True
        ax4 = predict_plot(
            samples_mov_gamma,
            df.loc[(df.date >= sec_start_date) & (df.date <= sec_end_date)],
            moving=True,
            grocery=True,
            rho=sec_states,
            second_phase=True,
        )
        for ax in ax4:
            for a in ax:
                a.set_ylim((0, 2.5))

        plt.savefig(
            figs_dir + data_date.strftime("%Y-%m-%d") + "Reff_sec_phase.png", dpi=144
        )

        # remove plots from memory
        fig.clear()
        plt.close(fig)

    # Load in vaccination data by state and date
    vaccination_by_state = pd.read_csv(
        "data/vaccine_effect_timeseries_" + data_date.strftime("%Y-%m-%d") + ".csv",
        parse_dates=["date"],
    )
    # there are a couple NA's early on in the time series but is likely due to slightly 
    # different start dates
    vaccination_by_state.fillna(1, inplace=True)
    # we take the whole set of estimates up to the end of the forecast period 
    # (with 10 days padding which won't be used in the forecast)
    vaccination_by_state = vaccination_by_state[
        (
            vaccination_by_state.date
            >= pd.to_datetime(third_start_date) - timedelta(days=1)
        )
        & (
            vaccination_by_state.date
            <= pd.to_datetime(data_date) + timedelta(days=num_forecast_days + 10)
        )
    ]
    vaccination_by_state_delta = vaccination_by_state.loc[
        vaccination_by_state["variant"] == "Delta"
    ][["state", "date", "effect"]]
    vaccination_by_state_omicron = vaccination_by_state.loc[
        vaccination_by_state["variant"] == "Omicron"
    ][["state", "date", "effect"]]

    vaccination_by_state_delta = vaccination_by_state_delta.pivot(
        index="state", columns="date", values="effect"
    )  # Convert to matrix form
    vaccination_by_state_omicron = vaccination_by_state_omicron.pivot(
        index="state", columns="date", values="effect"
    )  # Convert to matrix form

    # If we are missing recent vaccination data, fill it in with the most recent available data.
    latest_vacc_data = vaccination_by_state_omicron.columns[-1]
    if latest_vacc_data < pd.to_datetime(third_end_date):
        vaccination_by_state_delta = pd.concat(
            [vaccination_by_state_delta]
            + [
                pd.Series(vaccination_by_state_delta[latest_vacc_data], name=day)
                for day in pd.date_range(start=latest_vacc_data, end=third_end_date)
            ],
            axis=1,
        )
        vaccination_by_state_omicron = pd.concat(
            [vaccination_by_state_omicron]
            + [
                pd.Series(vaccination_by_state_omicron[latest_vacc_data], name=day)
                for day in pd.date_range(start=latest_vacc_data, end=third_end_date)
            ],
            axis=1,
        )

    # get the dates for vaccination
    dates = vaccination_by_state_delta.columns

    third_days = {k: v.shape[0] for (k, v) in third_date_range.items()}
    third_days_cumulative = np.append([0], np.cumsum([v for v in third_days.values()]))
    delta_ve_idx_ranges = {
        k: range(third_days_cumulative[i], third_days_cumulative[i + 1])
        for (i, k) in enumerate(third_days.keys())
    }
    third_days_tot = sum(v for v in third_days.values())

    # construct a range of dates for omicron which starts at the maximum of the start date 
    # for that state or the Omicron start date
    third_omicron_date_range = {
        k: pd.date_range(
            start=max(v[0], pd.to_datetime(omicron_start_date)), end=v[-1]
        ).values
        for (k, v) in third_date_range.items()
    }
    third_omicron_days = {k: v.shape[0] for (k, v) in third_omicron_date_range.items()}
    third_omicron_days_cumulative = np.append(
        [0], np.cumsum([v for v in third_omicron_days.values()])
    )
    omicron_ve_idx_ranges = {
        k: range(third_omicron_days_cumulative[i], third_omicron_days_cumulative[i + 1])
        for (i, k) in enumerate(third_omicron_days.keys())
    }
    third_omicron_days_tot = sum(v for v in third_omicron_days.values())

    # extrac the samples
    delta_ve_samples = samples_mov_gamma[
        ["ve_delta." + str(j + 1) for j in range(third_days_tot)]
    ].T
    omicron_ve_samples = samples_mov_gamma[
        ["ve_omicron." + str(j + 1) for j in range(third_omicron_days_tot)]
    ].T

    # now we plot and save the adjusted ve time series to be read in by the forecasting
    plot_adjusted_ve(
        data_date,
        samples_mov_gamma,
        states,
        vaccination_by_state_delta,
        third_states,
        third_date_range,
        delta_ve_samples,
        delta_ve_idx_ranges,
        figs_dir,
        "delta",
        custom_file_name=custom_file_name, 
    )

    plot_adjusted_ve(
        data_date,
        samples_mov_gamma,
        states,
        vaccination_by_state_omicron,
        third_states,
        third_omicron_date_range,
        omicron_ve_samples,
        omicron_ve_idx_ranges,
        figs_dir,
        "omicron",
        custom_file_name=custom_file_name, 
    )

    # extract the prop of omicron to delta and save
    total_N_p_third_omicron = int(sum([sum(x) for x in include_in_omicron_wave]).item())
    # prop_omicron_to_delta = samples_mov_gamma[
    #     ["prop_omicron_to_delta." + str(j + 1) for j in range(total_N_p_third_omicron)]
    # ]
    # pd.DataFrame(
    #     prop_omicron_to_delta.to_csv(
    #         "results/prop_omicron_to_delta" + data_date.strftime("%Y-%m-%d") + ".csv"
    #     )
    # )

    if df3X.shape[0] > 0:
        df["is_third_wave"] = 0
        for state in third_states:
            df.loc[df.state == state, "is_third_wave"] = (
                df.loc[df.state == state]
                .date.isin(third_date_range[state])
                .astype(int)
                .values
            )

        # plot only if there is third phase data - have to have third_phase=True
        ax4 = predict_plot(
            samples_mov_gamma,
            df.loc[(df.date >= third_start_date) & (df.date <= third_end_date)],
            moving=True,
            grocery=True,
            rho=third_states,
            third_phase=True,
        )  # by states....

        for ax in ax4:
            for a in ax:
                a.set_ylim((0, 2.5))
                # a.set_xlim((start_date,end_date))

        plt.savefig(
            figs_dir + data_date.strftime("%Y-%m-%d") + "Reff_third_phase_combined.png",
            dpi=144,
        )

        # remove plots from memory
        fig.clear()
        plt.close(fig)
        
    if df3X.shape[0] > 0:
        df["is_third_wave"] = 0
        for state in third_states:
            df.loc[df.state == state, "is_third_wave"] = (
                df.loc[df.state == state]
                .date.isin(third_date_range[state])
                .astype(int)
                .values
            )

        # plot only if there is third phase data - have to have third_phase=True
        ax4 = predict_plot(
            samples_mov_gamma,
            df.loc[(df.date >= third_start_date) & (df.date <= third_end_date)],
            moving=True,
            grocery=True,
            rho=third_states,
            third_phase=True,
            third_plot_type="delta"
        )  # by states....

        for ax in ax4:
            for a in ax:
                a.set_ylim((0, 2.5))
                # a.set_xlim((start_date,end_date))

        plt.savefig(
            figs_dir + data_date.strftime("%Y-%m-%d") + "Reff_third_phase_delta.png",
            dpi=144,
        )

        # remove plots from memory
        fig.clear()
        plt.close(fig)
        
    if df3X.shape[0] > 0:
        df["is_omicron_wave"] = 0
        for state in third_states:
            df.loc[df.state == state, "is_omicron_wave"] = (
                df.loc[df.state == state]
                .date.isin(third_omicron_date_range[state])
                .astype(int)
                .values
            )

        # plot only if there is third phase data - have to have third_phase=True
        ax4 = predict_plot(
            samples_mov_gamma,
            df.loc[(df.date >= omicron_start_date) & (df.date <= third_end_date)],
            moving=True,
            grocery=True,
            rho=third_states,
            third_phase=True,
            third_plot_type="omicron"
        )  # by states....

        for ax in ax4:
            for a in ax:
                a.set_ylim((0, 2.5))
                # a.set_xlim((start_date,end_date))

        plt.savefig(
            figs_dir + data_date.strftime("%Y-%m-%d") + "Reff_third_phase_omicron.png",
            dpi=144,
        )

        # remove plots from memory
        fig.clear()
        plt.close(fig)

    
    # plot the omicron proportion 
    
    # create a range of dates from the beginning of Omicron to use for producing the Omicron
    # proportion
    omicron_date_range = pd.date_range(
        omicron_start_date, pd.to_datetime(data_date) + timedelta(45)
    )
    prop_omicron_to_delta = np.array([])
    # create array of times to plot against 
    t = np.tile(range(len(omicron_date_range)), (samples_mov_gamma.shape[0], 1)).T

    fig, ax = plt.subplots(figsize=(15, 12), nrows=4, ncols=2, sharex=True, sharey=True)

    for (i, state) in enumerate(third_states):
        m0 = np.tile(samples_mov_gamma.loc[:, "m0." + str(i + 1)], (len(omicron_date_range), 1))
        m1 = np.tile(samples_mov_gamma.loc[:, "m1." + str(i + 1)], (len(omicron_date_range), 1))
        r = np.tile(samples_mov_gamma.loc[:, "r." + str(i + 1)], (len(omicron_date_range), 1))
        tau = np.tile(samples_mov_gamma.loc[:, "tau." + str(i + 1)] , (len(omicron_date_range), 1))
        
        omicron_start_date_tmp = max(
            pd.to_datetime(omicron_start_date), third_date_range[state][0]
        )
        omicron_date_range_tmp = pd.date_range(
            omicron_start_date_tmp, third_date_range[state][-1]
        )
        
        if state in {"TAS", "WA", "NT"}:
            prop_omicron_to_delta_tmp = m1 
        else:
            prop_omicron_to_delta_tmp = m0 + (m1 - m0) / (1 + np.exp(-r * (t - tau)))
        
        ax[i // 2, i % 2].plot(
            omicron_date_range, 
            np.median(prop_omicron_to_delta_tmp, axis=1),
        )
        
        ax[i // 2, i % 2].fill_between(
            omicron_date_range,
            np.quantile(prop_omicron_to_delta_tmp, 0.05, axis=1),
            np.quantile(prop_omicron_to_delta_tmp, 0.95, axis=1),
            alpha=0.2,
        )
        
        ax[i // 2, i % 2].axvline(
            omicron_date_range_tmp[0], ls="--", c="k", lw=1
        )
        
        ax[i // 2, i % 2].axvline(
            omicron_date_range_tmp[-1], ls="--", c="k", lw=1
        )
        
        ax[i // 2, i % 2].set_title(state)
        ax[i // 2, i % 2].xaxis.set_major_locator(plt.MaxNLocator(3))
        ax[i // 2, 0].set_ylabel("Proportion of Omicron\ncases to Delta")
        
        if len(prop_omicron_to_delta) == 0:
            prop_omicron_to_delta = prop_omicron_to_delta_tmp[:, -len(omicron_date_range_tmp):]
        else:
            prop_omicron_to_delta = np.hstack(
                (
                    prop_omicron_to_delta, 
                    prop_omicron_to_delta_tmp[:, -len(omicron_date_range_tmp):],
                )   
            )

    fig.tight_layout()
    
    plt.savefig(
        figs_dir + data_date.strftime("%Y-%m-%d") + "omicron_proportion.png", dpi=144
    )
    
    # need to rotate to put into a good format
    prop_omicron_to_delta = prop_omicron_to_delta.T

    df_prop_omicron_to_delta = pd.DataFrame(
        prop_omicron_to_delta, 
        columns=[
            "prop_omicron_to_delta." + str(i+1) for i in range(prop_omicron_to_delta.shape[1])
        ]
    )

    df_prop_omicron_to_delta.to_csv(
        "results/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/"
        + custom_file_name 
        + "/"
        + "prop_omicron_to_delta" 
        + data_date.strftime("%Y-%m-%d") 
        + ".csv"
    )

    # saving the final processed posterior samples to h5 for generate_RL_forecasts.py
    var_to_csv = predictors
    samples_mov_gamma[predictors] = samples_mov_gamma[
        ["bet." + str(i + 1) for i in range(len(predictors))]
    ]
    var_to_csv = [
        "R_I",
        "R_I_omicron",
        "R_L",
        "sig",
        "theta_md",
        "voc_effect_alpha",
        "voc_effect_delta",
        "voc_effect_omicron",
        "susceptible_depletion_factor",
    ]
    var_to_csv = (
        var_to_csv
        + predictors
        + ["R_Li." + str(i + 1) for i in range(len(states_to_fit_all_waves))]
    )
    var_to_csv = var_to_csv + ["ve_delta." + str(j + 1) for j in range(third_days_tot)]
    var_to_csv = var_to_csv + [
        "ve_omicron." + str(j + 1) for j in range(third_omicron_days_tot)
    ]
    var_to_csv = var_to_csv + ["r." + str(j + 1) for j in range(len(third_states))]
    var_to_csv = var_to_csv + ["tau." + str(j + 1) for j in range(len(third_states))]
    var_to_csv = var_to_csv + ["m0." + str(j + 1) for j in range(len(third_states))]
    var_to_csv = var_to_csv + ["m1." + str(j + 1) for j in range(len(third_states))]

    # save the posterior
    samples_mov_gamma[var_to_csv].to_hdf(
        "results/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/"
        + custom_file_name 
        + "/"
        + "soc_mob_posterior" 
        + data_date.strftime("%Y-%m-%d") 
        + ".h5",
        key="samples",
    )

    return None


def main(data_date, run_inference=True):
    """
    Runs the stan model in parts to cut down on memory.
    """
    # some parameters for HMC
    custom_file_name = str(round(p_detect_omicron * 100)) + "_case_ascertainment"
        
    if run_inference:     
        num_chains = 4
        num_samples = 2000
        num_warmup_samples = 1000
        # num_samples = 200
        # num_warmup_samples = 200
        get_data_for_posterior(data_date=data_date)
        
        run_stan(
            data_date=data_date,
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup_samples=num_warmup_samples,
            custom_file_name=custom_file_name,
        )
        
    plot_and_save_posterior_samples(
        data_date=data_date, 
        custom_file_name=custom_file_name
    )

    return None


if __name__ == "__main__":
    """
    If we are running the script here (which is always) then this ensures things run appropriately.
    """
    data_date = argv[1]
    
    if len(argv) == 2:
        run_inference = True
    elif len(argv) > 2:
        if argv[2] == "False": 
            run_inference = False
        
    main(data_date, run_inference)