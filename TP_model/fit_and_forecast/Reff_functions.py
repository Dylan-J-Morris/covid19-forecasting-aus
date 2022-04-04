import os
import sys
from numpy.core.numeric import zeros_like
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-poster")
# I hate this too but it allows everything to use the same helper functions.
sys.path.insert(0, "model")
from helper_functions import read_in_NNDSS
from Reff_constants import *

def read_in_posterior(date):
    """
    read in samples from posterior from inference
    """
    df = pd.read_hdf(
        "results/"
        + date 
        + "/soc_mob_posterior" 
        + date 
        + ".h5", 
        key="samples"
    )

    return df


def read_in_google(Aus_only=True, local=True, moving=False):
    """
    Read in the Google data set
    """
    if local:
        if type(local) == str:
            df = pd.read_csv(local, parse_dates=["date"])
        elif type(local) == bool:
            local = "data/Global_Mobility_Report.csv"
            df = pd.read_csv(local, parse_dates=["date"])
    else:
        # Download straight from the web
        df = pd.read_csv(
            "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv",
            parse_dates=["date"],
        )
        # Make it save automatically.
        df.to_csv("data/Global_Mobility_Report.csv", index=False)

    if Aus_only:
        df = df.loc[df.country_region_code == "AU"]
        # Change state column to state initials
        df["state"] = df.sub_region_1.map(
            lambda x: states_initials[x] if not pd.isna(x) else "AUS"
        )

    df = df.loc[df.sub_region_2.isna()]
    if moving:
        # generate moving average columns in reverse
        df = df.sort_values(by="date")
        mov_values = []
        for val in value_vars:
            mov_values.append(val[:-29] + "_7days")
            df[mov_values[-1]] = df.groupby(["state"])[val].transform(
                lambda x: x[::-1].rolling(7, 1).mean()[::-1]
            )  # minimumnumber of 1

            # minimum of 7 days for std, forward fill the rest
            df[mov_values[-1] + "_std"] = df.groupby(["state"])[val].transform(
                lambda x: x[::-1].rolling(7, 7).std()[::-1]
            )
            # fill final values as std doesn't work with single value
            df[mov_values[-1] + "_std"] = df.groupby("state")[
                mov_values[-1] + "_std"
            ].fillna(method="ffill")

    # show latest date
    print("Latest date in Google indices " + str(df.date.values[-1]))
    return df


def predict_plot(
    samples,
    df,
    moving=True,
    grocery=True,
    rho=None,
    second_phase=False,
    third_phase=False,
    third_plot_type="combined",
):
    """
    Produce posterior predictive plots for all states using the inferred mu_hat. This should run 
    regardless of the form of the model as it only requires the mu_hat parameter which is 
    calculated inside stan (the TP model fitted to the Reff). 
    """

    value_vars = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    value_vars.remove("residential_percent_change_from_baseline")

    if not grocery:
        value_vars.remove("grocery_and_pharmacy_percent_change_from_baseline")
    if moving:
        value_vars = [val[:-29] + "_7days" for val in value_vars]

    # all states
    fig, ax = plt.subplots(figsize=(15, 12), ncols=4, nrows=2, sharex=True, sharey=True)

    states = sorted(list(states_initials.keys()))

    if not third_phase:
        states.remove("Northern Territory")
        states.remove("Australian Capital Territory")

    # no R_eff modelled for these states, skip
    # counter for brho_v
    pos = 0
    for i, state in enumerate(states):

        df_state = df.loc[df.sub_region_1 == state]
        if second_phase:
            df_state = df_state.loc[df_state.is_sec_wave == 1]
        elif third_phase:
            df_state = df_state.loc[df_state.is_third_wave == 1]

        # directly plot the fitted TP values 
        states_to_fitd = {s: i + 1 for i, s in enumerate(rho)}
        
        if not second_phase and not third_phase:
            mu_hat = samples[
                [
                    "mu_hat["
                    + str(j + 1)
                    + ","
                    + str(states_to_fitd[states_initials[state]])
                    + "]"
                    for j in range(df_state.shape[0])
                ]
            ].values.T
        elif second_phase:
            mu_hat = samples[
                [
                    "mu_hat_sec_wave[" + str(j + 1) + "]"
                    for j in range(
                        pos,
                        pos
                        + df.loc[
                            df.state == states_initials[state]
                        ].is_sec_wave.sum(),
                    )
                ]
            ].values.T
            pos = (
                pos
                + df.loc[
                    df.state == states_initials[state]
                ].is_sec_wave.sum()
            )
        elif third_phase:
            if third_plot_type == "combined":
                mu_hat = samples[
                    [
                        "mu_hat_third_wave[" + str(j + 1) + "]"
                        for j in range(
                            pos,
                            pos
                            + df.loc[
                                df.state == states_initials[state]
                            ].is_third_wave.sum(),
                        )
                    ]
                ].values.T
                pos = (
                    pos
                    + df.loc[
                        df.state == states_initials[state]
                    ].is_third_wave.sum()
                )
            elif third_plot_type == "delta":
                mu_hat = samples[
                    [
                        "mu_hat_delta_only[" + str(j + 1) + "]"
                        for j in range(
                            pos,
                            pos
                            + df.loc[
                                df.state == states_initials[state]
                            ].is_third_wave.sum(),
                        )
                    ]
                ].values.T
                pos = (
                    pos
                    + df.loc[
                        df.state == states_initials[state]
                    ].is_third_wave.sum()
                )
            elif third_plot_type == "omicron":
                mu_hat = samples[
                    [
                        "mu_hat_omicron_only[" + str(j + 1) + "]"
                        for j in range(
                            pos,
                            pos
                            + df.loc[
                                df.state == states_initials[state]
                            ].is_omicron_wave.sum(),
                        )
                    ]
                ].values.T
                pos = (
                    pos
                    + df.loc[
                        df.state == states_initials[state]
                    ].is_omicron_wave.sum()
                )
            
        
        df_hat = pd.DataFrame(mu_hat.T)
        
        # df_hat.to_csv('mu_hat_' + state + '.csv')

        if states_initials[state] not in rho:
            if i // 4 == 1:
                ax[i // 4, i % 4].tick_params(axis="x", rotation=90)
            continue
        
        if not third_phase: 
            # plot actual R_eff
            ax[i // 4, i % 4].plot(
                df_state.date, df_state["mean"], label="$R_{eff}$", color="C1"
            )
            ax[i // 4, i % 4].fill_between(
                df_state.date, df_state["bottom"], df_state["top"], color="C1", alpha=0.3
            )
            ax[i // 4, i % 4].fill_between(
                df_state.date, df_state["lower"], df_state["upper"], color="C1", alpha=0.3
            )
        elif third_phase:
            if third_plot_type in ("combined", "omicron"):
                # plot actual R_eff
                ax[i // 4, i % 4].plot(
                    df_state.date, df_state["mean_omicron"], label="$R_{eff}$", color="C1"
                )
                ax[i // 4, i % 4].fill_between(
                    df_state.date, 
                    df_state["bottom_omicron"], 
                    df_state["top_omicron"], 
                    color="C1", 
                    alpha=0.3
                )
                ax[i // 4, i % 4].fill_between(
                    df_state.date, 
                    df_state["lower_omicron"], 
                    df_state["upper_omicron"], 
                    color="C1", 
                    alpha=0.3
                )
            else:
                # plot actual R_eff
                ax[i // 4, i % 4].plot(
                    df_state.date, df_state["mean"], label="$R_{eff}$", color="C1"
                )
                ax[i // 4, i % 4].fill_between(
                    df_state.date, 
                    df_state["bottom"], 
                    df_state["top"], 
                    color="C1", 
                    alpha=0.3
                )
                ax[i // 4, i % 4].fill_between(
                    df_state.date, 
                    df_state["lower"], 
                    df_state["upper"], 
                    color="C1", 
                    alpha=0.3
                )
            
            
        ax[i // 4, i % 4].plot(
            df_state.date, df_hat.quantile(0.5, axis=0), label="$\hat{\mu}$", color="C0"
        )
        ax[i // 4, i % 4].fill_between(
            df_state.date,
            df_hat.quantile(0.25, axis=0),
            df_hat.quantile(0.75, axis=0),
            color="C0",
            alpha=0.3,
        )
        ax[i // 4, i % 4].fill_between(
            df_state.date,
            df_hat.quantile(0.05, axis=0),
            df_hat.quantile(0.95, axis=0),
            color="C0",
            alpha=0.3,
        )
        ax[i // 4, i % 4].set_title(state)
        # grid line at R_eff =1
        ax[i // 4, i % 4].axhline(1, ls="--", c="k", lw=1)
        ax[i // 4, i % 4].set_yticks([0, 1, 2], minor=False)
        ax[i // 4, i % 4].set_yticklabels([0, 1, 2], minor=False)
        ax[i // 4, i % 4].yaxis.grid(
            which="minor", linestyle="--", color="black", linewidth=2
        )
        ax[i // 4, i % 4].set_ylim((0, 2.5))
        
        if i // 4 == 1:
            ax[i // 4, i % 4].tick_params(axis="x", rotation=90)

    plt.legend()
    
    return ax


def plot_adjusted_ve(
    data_date,
    samples_mov_gamma,
    states,
    vaccination_by_state,
    third_states,
    third_date_range,
    ve_samples,
    ve_idx_ranges,
    figs_dir,
    strain,
):

    """
    A function to process the inferred VE. This will save an updated timeseries which 
    is the mean posterior estimates.
    """

    fig, ax = plt.subplots(figsize=(15, 12), ncols=2, nrows=4, sharey=True, sharex=True)
    # temporary state vector

    # make a dataframe for the adjusted vacc_ts
    df_vacc_ts_adjusted = pd.DataFrame()

    # for i, state in enumerate(third_states):
    for i, state in enumerate(states):
        # for i, state in enumerate(states_tmp):
        # grab states vaccination data
        vacc_ts_data = vaccination_by_state.loc[state]

        # apply different vaccine form depending on if NSW
        if state in third_states:
            # get the sampled vaccination effect (this will be incomplete as it's only 
            # over the fitting period)
            vacc_tmp = ve_samples.iloc[ve_idx_ranges[state], :]
            # get before and after fitting and tile them
            vacc_ts_data_before = pd.concat(
                [vacc_ts_data.loc[vacc_ts_data.index < third_date_range[state][0]]]
                * samples_mov_gamma.shape[0],
                axis=1,
            )
            vacc_ts_data_after = pd.concat(
                [vacc_ts_data.loc[vacc_ts_data.index > third_date_range[state][-1]]]
                * samples_mov_gamma.shape[0],
                axis=1,
            )
            # rename columns for easy merging
            vacc_ts_data_before.columns = vacc_tmp.columns
            vacc_ts_data_after.columns = vacc_tmp.columns
            # merge in order
            vacc_ts = pd.concat(
                [vacc_ts_data_before, vacc_tmp, vacc_ts_data_after],
                axis=0,
                ignore_index=True,
            )
            vacc_ts.set_index(vacc_ts_data.index[: vacc_ts.shape[0]], inplace=True)

        else:
            # just tile the data
            vacc_ts = pd.concat(
                [vacc_ts_data] * samples_mov_gamma.shape[0],
                axis=1,
            )
            # reset the index to be the dates for easier information handling
            vacc_ts.set_index(vacc_ts_data.index, inplace=True)
            # need to name columns samples for consistent indexing
            vacc_ts.columns = range(0, samples_mov_gamma.shape[0])

        dates = vacc_ts.index
        vals = vacc_ts.median(axis=1).values
        state_vec = np.repeat([state], vals.shape[0])
        df_vacc_ts_adjusted = pd.concat(
            [
                df_vacc_ts_adjusted,
                pd.DataFrame({"state": state_vec, "date": dates, "effect": vals}),
            ]
        )

        # create zero array to fill in with the full vaccine effect model
        vacc_eff = np.zeros_like(vacc_ts)

        # Note that in here we apply the entire sample to the vaccination data 
        # to create a days by samples array
        for ii in range(vacc_eff.shape[0]):
            vacc_eff[ii] = vacc_ts.iloc[ii, :]

        row = i % 4
        col = i // 4

        ax[row, col].plot(
            dates,
            vaccination_by_state.loc[state][: dates.shape[0]].values,
            label="data",
            color="C1",
        )
        ax[row, col].plot(dates, np.median(vacc_eff, axis=1), label="fit", color="C0")
        ax[row, col].fill_between(
            dates,
            np.quantile(vacc_eff, 0.25, axis=1),
            np.quantile(vacc_eff, 0.75, axis=1),
            color="C0",
            alpha=0.4,
        )
        ax[row, col].fill_between(
            dates,
            np.quantile(vacc_eff, 0.05, axis=1),
            np.quantile(vacc_eff, 0.95, axis=1),
            color="C0",
            alpha=0.4,
        )
        # plot the start and end of the fitting
        if state in third_states:
            ax[row, col].axvline(third_date_range[state][0], ls="--", color="red", lw=1)
            ax[row, col].axvline(third_date_range[state][-1], ls="--", color="red", lw=1)
        ax[row, col].set_title(state)
        ax[row, col].tick_params(axis="x", rotation=90)

    ax[1, 0].set_ylabel("reduction in TP from vaccination")

    df_vacc_ts_adjusted.to_csv(
        "results/" 
        + data_date.strftime("%Y-%m-%d") 
        + "/adjusted_vaccine_ts_"
        + strain
        + data_date.strftime("%Y-%m-%d")
        + ".csv",
        index=False,
    )

    plt.savefig(
        figs_dir
        + data_date.strftime("%Y-%m-%d")
        + "_"
        + strain
        + "_ve_reduction_in_TP.png",
        dpi=144,
    )

    # remove plots from memory
    fig.clear()
    plt.close(fig)

    return None


def modify_cases_by_CAR(df_state):
    """
    Apply a scaling to the daily reported cases by accounting for a ~75% detection probability pre
    15/12/2021 and 0.33 following that. To improve the transition, we assume that detection 
    probability decreases from 0.75 to 0.5 over 7 days beginning 9/12/2021.
    """
    
    from params import p_detect_delta
    
    start_date = np.min(df_state.index)
    end_date = np.max(df_state.index)
    
    CAR_normal_before = pd.date_range(start_date, "2021-12-12")
    CAR_decrease_range = pd.date_range("2021-12-13", "2021-12-19")
    CAR_low_range = pd.date_range("2021-12-20", "2022-01-16")
    CAR_increase_range = pd.date_range("2022-01-17", "2022-01-23")
    CAR_normal_after = pd.date_range("2022-01-24", end_date)
    
    CAR_dates = np.concatenate(
        (
            CAR_normal_before, 
            CAR_decrease_range,
            CAR_low_range, 
            CAR_increase_range, 
            CAR_normal_after,
        )
    )
    
    # get baseline CAR
    CAR = p_detect_delta * np.ones(CAR_dates.shape)
    # determine index arrays for the various phases assumed
    decrease_bool = (CAR_dates >= CAR_decrease_range[0]) & (CAR_dates <= CAR_decrease_range[-1])
    low_bool = (CAR_dates > CAR_decrease_range[-1]) & (CAR_dates < CAR_increase_range[0])
    increase_bool = (CAR_dates >= CAR_increase_range[0]) & (CAR_dates <= CAR_increase_range[-1])
    # adjust the CAR based on the time varying assumptions by approximating the step change 
    # linearly 
    CAR[decrease_bool] = np.linspace(p_detect_delta, 0.333, 7)
    CAR[low_bool] = 0.333
    CAR[increase_bool] = np.linspace(0.333, p_detect_delta, 7)
    
    # scale the cases up each day by the CAR
    for d in df_state.index:
        df_state.loc[d, "local"] = np.ceil(1 / CAR[CAR_dates == d] * df_state.loc[d, "local"])
        
    # reset the indices so that the merging works as intended
    df_state.reset_index(drop=False, inplace=True)
    # the above worked with onset dates so we map back by (approx) the mean IP across Delta 
    # and Omicron (around 5 days noting that this is only needed for the inferred import 
    # proportion and susceptible depletion in the stan model so a day off isn't
    # too concerning)
    df_state["date_inferred"] = df_state["date_inferred"] - pd.Timedelta(days=5)
    
    return None 


def modify_cases_by_CAR_old(df_state):
    """
    Apply a scaling to the daily reported cases by accounting for a ~75% detection probability pre
    15/12/2021 and 0.5 following that. To improve the transition, we assume that detection 
    probability decreases from 0.75 to 0.5 over 7 days beginning 9/12/2021.
    """
    
    from params import p_detect_delta, omicron_dominance_date
    
    start_date = np.min(df_state.index)
    end_date = np.max(df_state.index)
    
    CAR_normal_before = pd.date_range(start_date, "2021-12-08")
    CAR_decrease_range = pd.date_range("2021-12-09", "2021-12-15")
    CAR_normal_after = pd.date_range("2021-12-16", end_date)
    
    CAR_dates = np.concatenate(
        (
            CAR_normal_before, 
            CAR_decrease_range,
            CAR_normal_after,
        )
    )
    
    # get baseline CAR
    CAR = p_detect_delta * np.ones(CAR_dates.shape)
    # determine index arrays for the various phases assumed
    decrease_bool = (CAR_dates >= CAR_decrease_range[0]) & (CAR_dates <= CAR_decrease_range[-1])
    after_bool = CAR_dates > CAR_decrease_range[-1]
    # adjust the CAR based on the time varying assumptions 
    CAR[decrease_bool] = np.linspace(p_detect_delta, 0.5, 7)
    CAR[after_bool] = 0.5
    
    # scale the cases up each day by the CAR
    for d in df_state.index:
        df_state.loc[d, "local"] = np.ceil(1 / CAR[CAR_dates == d] * df_state.loc[d, "local"])
        
    # reset the indices so that the merging works as intended
    df_state.reset_index(drop=False, inplace=True)
    # the above worked with onset dates so we map back by (approx) the mean IP across Delta 
    # and Omicron (around 5 days noting that this is only needed for the inferred import 
    # proportion and susceptible depletion in the stan model so a day off isn't
    # too concerning)
    df_state["date_inferred"] = df_state["date_inferred"] - pd.Timedelta(days=5)
        
    return None 


def read_in_cases(
    case_file_date, 
    apply_delay_at_read=False, 
    apply_inc_at_read=False,
    apply_case_ascertainment_increase=False,
):
    """
    Read in NNDSS data and from data, find rho
    """
    
    df_NNDSS = read_in_NNDSS(
        case_file_date,
        apply_delay_at_read=apply_delay_at_read,
        apply_inc_at_read=apply_inc_at_read,
    )
    
    df_state_all = pd.DataFrame()
    
    states = df_NNDSS["STATE"].unique()
    
    for state in states:
        # boolean array for isolating the current state
        ind = df_NNDSS["STATE"] == state
        # aggregate data into daily local / imported cases 
        df_state = (
            df_NNDSS[ind][["date_inferred", "imported", "local"]]
            .groupby(["date_inferred"])
            .sum()
        )
    
        if apply_case_ascertainment_increase:
            modify_cases_by_CAR_old(df_state=df_state)
            
            # modify_cases_by_CAR(df_state=df_state)
        
        # add state column back in 
        df_state["STATE"] = state    
        
        # calculate the raw proportion of imported to local cases based off the detection 
        # probability
        df_state["rho"] = [
            0 if (i + l == 0) else i / (i + l)
            for l, i in zip(df_state.local, df_state.imported)
        ]
        
        # concatenate with full array 
        df_state_all = pd.concat((df_state_all, df_state))


    return df_state_all
