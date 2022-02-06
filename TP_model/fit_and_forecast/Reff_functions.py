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
from params import (
    alpha_start_date,
    delta_start_date,
    omicron_start_date,
    vaccination_start_date,
)


def read_in_posterior(date):
    """
    read in samples from posterior from inference
    """
    df = pd.read_hdf("results/soc_mob_posterior" + date + ".h5", key="samples")

    return df


def read_in_google(Aus_only=True, local=False, moving=False):
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
    third_date_range=None,
    third_omicron_date_range=None,
    split=True,
    moving=True,
    grocery=True,
    R=2.2,
    md_arg=None,
    R_I=None,
    ban="2020-03-16",
    var=None,
    rho=None,
    prop=None,
    masks_prop=None,
    second_phase=False,
    third_phase=False,
    third_states=None,
    prop_omicron_to_delta=None,
):
    """
    Produce posterior predictive plots for all states
    """
    from scipy.special import expit
    from params import third_start_date

    os.makedirs("results/third_wave_fit/", exist_ok=True)

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

        masks_prop_sim = masks_prop[states_initials[state]].values[
            : df_state.shape[0]
        ]

        samples_sim = samples.sample(1000)
        post_values = samples_sim[
            ["bet." + str(i + 1) for i in range(len(value_vars))]
        ].values.T
        prop_sim = prop[states_initials[state]].values[: df_state.shape[0]]

        if split:

            # split model with parameters pre and post policy
            df1 = df_state.loc[df_state.date <= ban]
            df2 = df_state.loc[df_state.date > ban]
            X1 = df1[value_vars] / 100  # N by K
            X2 = df2[value_vars] / 100
            # N by K times (Nsamples by K )^T = N by N
            logodds = X1 @ post_values

            theta_md = samples_sim.theta_md.values  # 1 by samples shape
            # each row is a date, column a new sample
            theta_md = np.tile(theta_md, (df_state.shape[0], 1))
            md = ((1 + theta_md).T ** (-1 * prop_sim)).T
            # set preban md values to 1
            md[: logodds.shape[0]] = 1
            
            theta_masks = samples_sim.theta_masks.values  # 1 by samples shape
            # each row is a date, column a new sample
            theta_masks = np.tile(theta_masks, (df_state.shape[0], 1))
            masks = ((1 + theta_masks).T ** (-1 * masks_prop_sim)).T
            # set preban mask values to 1
            masks[: logodds.shape[0]] = 1

            # make logodds by appending post ban values
            logodds = np.append(logodds, X2 @ post_values, axis=0)

            # grab posterior sampled vaccination effects here and multiply by the daily efficacy
            if third_phase and states_initials[state] in third_states:
                third_days = {k: v.shape[0] for (k, v) in third_date_range.items()}
                third_days_cumulative = np.append(
                    [0], np.cumsum([v for v in third_days.values()])
                )
                vax_idx_ranges = {
                    k: range(third_days_cumulative[i], third_days_cumulative[i + 1])
                    for (i, k) in enumerate(third_days.keys())
                }
                third_days_tot = sum(v for v in third_days.values())

                # construct a range of dates for omicron which starts at the maximum of the start date for that state or the Omicron start date
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
                delta_ve_samples = samples_sim[
                    ["ve_delta." + str(j + 1) for j in range(third_days_tot)]
                ].T
                omicron_ve_samples = samples_sim[
                    ["ve_omicron." + str(j + 1) for j in range(third_omicron_days_tot)]
                ].T
                delta_ve_tmp = delta_ve_samples.iloc[
                    vax_idx_ranges[states_initials[state]], :
                ]
                omicron_ve_tmp = omicron_ve_samples.iloc[
                    omicron_ve_idx_ranges[states_initials[state]], :
                ]

                omicron_start_day = (
                    pd.to_datetime(omicron_start_date)
                    - third_date_range[states_initials[state]][0]
                ).days
                omicron_start_day = 0 if omicron_start_day < 0 else omicron_start_day

                # create zero array to fill in with the full vaccine effect model
                vacc_post = np.zeros_like(delta_ve_tmp)

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

                m = prop_omicron_to_delta.iloc[
                    :, idx[states_initials[state]]
                ].to_numpy()
                m = m[: vacc_post.shape[1]].T

                # note that in here we apply the entire sample to the vaccination data to create a days by samples array
                for ii in range(vacc_post.shape[0]):
                    if ii < omicron_start_day:
                        vacc_post[ii] = delta_ve_tmp.iloc[ii, :]
                    else:
                        jj = ii - omicron_start_day
                        # calculate the raw vax effect
                        vacc_delta = delta_ve_tmp.iloc[ii, :]
                        # indexed by days into omicron
                        vacc_omicron = omicron_ve_tmp.iloc[jj, :]
                        # calculate the full vaccination effect
                        vacc_post[ii] = m[jj] * vacc_omicron + (1 - m[jj]) * vacc_delta

                for ii in range(vacc_post.shape[0]):
                    if (
                        ii
                        < df_state.loc[df_state.date < vaccination_start_date].shape[0]
                    ):
                        vacc_post[ii] = 1.0

        if type(R) == str:  # 'state'
            try:
                sim_R = samples_sim["R_" + states_initials[state]]
            except KeyError:
                # this state not fitted, use gamma prior on initial value
                print("using initial value for state" + state)
                sim_R = np.random.gamma(
                    shape=df.loc[df.date == "2020-03-01", "mean"].mean() ** 2 / 0.2,
                    scale=0.2 / df.loc[df.date == "2020-03-01", "mean"].mean(),
                    size=df_state.shape[0],
                )

        if type(R) == dict:
            if states_initials[state] != ["NT"]:
                # if state, use inferred
                sim_R = np.tile(
                    R[states_initials[state]][: samples_sim.shape[0]],
                    (df_state.shape[0], 1),
                )
            else:
                # if territory, use generic R_L
                sim_R = np.tile(samples_sim.R_L.values, (df_state.shape[0], 1))
        else:
            sim_R = np.tile(samples_sim.R_L.values, (df_state.shape[0], 1))

        if third_phase and states_initials[state] in third_states:
            mu_hat = 2 * md * masks * sim_R * expit(logodds) * vacc_post
        else:
            mu_hat = 2 * md * masks * sim_R * expit(logodds)

        if rho:
            if rho == "data":
                rho_data = np.tile(
                    df_state.rho_moving.values[np.newaxis].T, (1, samples_sim.shape[0])
                )
            else:
                states_to_fitd = {s: i + 1 for i, s in enumerate(rho)}
                if states_initials[state] in states_to_fitd.keys():
                    # transpose as columns are days, need rows to be days
                    if second_phase:
                        # use brho_v

                        rho_data = samples_sim[
                            [
                                "brho_sec_wave." + str(j + 1)
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
                        # use brho_v

                        rho_data = samples_sim[
                            [
                                "brho_third_wave." + str(j + 1)
                                for j in range(
                                    pos,
                                    pos
                                    + df.loc[
                                        df.state == states_initials[state]
                                    ].is_third_wave.sum(),
                                )
                            ]
                        ].values.T

                        voc_multiplier_alpha = samples_sim[
                            ["voc_effect_alpha"]
                        ].values.T
                        voc_multiplier_delta = np.tile(
                            samples_sim[["voc_effect_delta"]].values.T,
                            (mu_hat.shape[0], 1),
                        )
                        voc_multiplier_omicron = np.tile(
                            samples_sim[["voc_effect_omicron"]].values.T,
                            (mu_hat.shape[0], 1),
                        )
                        # now we just modify the values before the introduction of the voc to be 1.0
                        voc_multiplier = np.zeros_like(voc_multiplier_delta)

                        for ii in range(voc_multiplier.shape[0]):
                            if (
                                ii
                                < df_state.loc[df_state.date < alpha_start_date].shape[
                                    0
                                ]
                            ):
                                voc_multiplier[ii] = 1.0
                            elif (
                                ii
                                < df_state.loc[df_state.date < delta_start_date].shape[
                                    0
                                ]
                            ):
                                voc_multiplier[ii] = voc_multiplier_alpha[ii]
                            elif (
                                ii
                                < df_state.loc[
                                    df_state.date < omicron_start_date
                                ].shape[0]
                            ):
                                voc_multiplier[ii] = voc_multiplier_delta[ii]
                            else:
                                jj = (
                                    ii
                                    - df_state.loc[
                                        df_state.date < omicron_start_date
                                    ].shape[0]
                                )
                                voc_multiplier[ii] = (
                                    m[jj] * voc_multiplier_omicron[ii]
                                    + (1 - m[jj]) * voc_multiplier_delta[ii]
                                )

                        # now modify the mu_hat
                        mu_hat *= voc_multiplier

                        pos = (
                            pos
                            + df.loc[
                                df.state == states_initials[state]
                            ].is_third_wave.sum()
                        )

                    else:
                        # first phase
                        rho_data = samples_sim[
                            [
                                "brho."
                                + str(j + 1)
                                + "."
                                + str(states_to_fitd[states_initials[state]])
                                for j in range(df_state.shape[0])
                            ]
                        ].values.T
                else:
                    print("Using data as inference not done on {}".format(state))
                    rho_data = np.tile(
                        df_state.rho_moving.values[np.newaxis].T,
                        (1, samples_sim.shape[0]),
                    )

            R_I_sim = np.tile(samples_sim.R_I.values, (df_state.shape[0], 1))

            R_eff_hat = rho_data * R_I_sim + (1 - rho_data) * mu_hat

        df_hat = pd.DataFrame(R_eff_hat.T)

        if states_initials[state] not in rho:
            if i // 4 == 1:
                ax[i // 4, i % 4].tick_params(axis="x", rotation=90)
            continue
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
        ax[i // 4, i % 4].set_yticks(
            [1],
            minor=True,
        )
        ax[i // 4, i % 4].set_yticks([0, 2, 3], minor=False)
        ax[i // 4, i % 4].set_yticklabels([0, 2, 3], minor=False)
        ax[i // 4, i % 4].yaxis.grid(
            which="minor", linestyle="--", color="black", linewidth=2
        )
        ax[i // 4, i % 4].set_ylim((0, 4))
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
    results_dir,
    strain,
):

    """
    A function to process the inferred VE. This will save an updated timeseries which is the mean posterior
    estimates.
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
            # get the sampled vaccination effect (this will be incomplete as it's only over the fitting period)
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

        # * Note that in here we apply the entire sample to the vaccination data to create a days by samples array
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
        "results/adjusted_vaccine_ts_"
        + strain
        + data_date.strftime("%Y-%m-%d")
        + ".csv",
        index=False,
    )

    plt.savefig(
        results_dir
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


def read_in_cases(case_file_date, apply_delay_at_read=False, apply_inc_at_read=False):
    """
    Read in NNDSS data and from data, find rho
    """
    from datetime import timedelta
    import glob

    df_NNDSS = read_in_NNDSS(
        case_file_date,
        apply_delay_at_read=apply_delay_at_read,
        apply_inc_at_read=apply_inc_at_read,
    )

    df_state = (
        df_NNDSS[["date_inferred", "STATE", "imported", "local"]]
        .groupby(["STATE", "date_inferred"])
        .sum()
    )

    df_state["rho"] = [
        0 if (i + l == 0) else i / (i + l)
        for l, i in zip(df_state.local, df_state.imported)
    ]

    return df_state
