import sys

# I hate this too but it allows everything to use the same helper functions.
sys.path.insert(0, "TP_model")
from helper_functions import read_in_NNDSS, sample_discrete_dist
from scipy.stats import gamma
import glob
from datetime import timedelta
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from params import use_linelist
import matplotlib

matplotlib.use("Agg")

plt.style.use("seaborn-poster")

# Code taken from read_in_cases from Reff_functions. Preprocessing was not helpful for this situation.


def read_cases_lambda(case_file_date):
    """
    Read in NNDSS data
    """
    df_NNDSS = read_in_NNDSS(case_file_date)

    if use_linelist:
        df_interim = df_NNDSS[
            ["date_inferred", "is_confirmation", "STATE", "imported", "local"]
        ]
    else:
        df_interim = df_NNDSS[["date_inferred", "STATE", "imported", "local"]]

    return df_interim


def tidy_cases_lambda(interim_data, remove_territories=True):
    # Remove non-existent notification dates
    interim_data = interim_data[~np.isnat(interim_data.date_inferred)]

    # Filter out territories
    if remove_territories:
        df_linel = interim_data[(interim_data["STATE"] != "NT")]
    else:
        df_linel = interim_data

    # Melt down so that imported and local are no longer columns. Allows multiple draws for infection date.
    # i.e. create linelist data
    if use_linelist:
        df_linel = df_linel.melt(
            id_vars=["date_inferred", "STATE", "is_confirmation"],
            var_name="SOURCE",
            value_name="n_cases",
        )
    else:
        df_linel = df_linel.melt(
            id_vars=["date_inferred", "STATE"], var_name="SOURCE", value_name="n_cases"
        )

    # Reset index or the joining doesn't work
    df_linel = df_linel[df_linel.n_cases != 0]
    df_linel = df_linel.reset_index(drop=True)

    return df_linel


def draw_inf_dates(
    df_linelist,
    inc_disc_pmf=[], 
    rd_disc_pmf=[], 
):
    # these aren't really notification dates, they are a combination of onset and confirmation dates
    notification_dates = df_linelist["date_inferred"]

    # the above are the same size so this works
    nsamples = notification_dates.shape[0]

    # Draw from incubation distribution
    # inc_period = np.random.gamma(shape_inc, scale_inc, size=(nsamples))

    # apply the delay at the point of applying the incubation
    # as we are taking a posterior sample
    # extract boolean indicator of when the confirmation date was used
    is_confirmation_date = df_linelist["is_confirmation"].to_numpy()
    id_nd_diff = (
        sample_discrete_dist(inc_disc_pmf, nsamples) 
        + sample_discrete_dist(rd_disc_pmf, nsamples) * is_confirmation_date
    )
    # Minutes aren't included in df. Take the ceiling because the day runs from 0000 to 2359. This can still be a long vector.
    whole_day_diff = np.ceil(id_nd_diff)
    time_day_diffmat = whole_day_diff.astype("timedelta64[D]").reshape(nsamples)

    # infection dates are just the difference
    infection_dates = notification_dates - time_day_diffmat

    # need to remove the confirmation boolean variable from the df to ensure that the
    # rest of epyreff runs as per normal
    df_linelist = df_linelist.loc[:, df_linelist.columns != "is_confirmation"]
    
    df_inf = df_linelist 
    df_inf["infection_date_inferred"] = infection_dates

    return df_inf


def index_by_infection_date(infections_wide):
    datecolnames = [*infections_wide.columns[4:]]
    df_combined = (
        infections_wide[["STATE", "SOURCE", datecolnames[0], "n_cases"]]
        .groupby(["STATE", datecolnames[0], "SOURCE"])
        .sum()
    )

    # For each column (cn=column number): concatenate each sample as a column.
    for cn in range(1, len(datecolnames)):
        df_addin = (
            infections_wide[["STATE", "SOURCE", datecolnames[cn], "n_cases"]]
            .groupby(["STATE", datecolnames[cn], "SOURCE"])
            .sum()
        )
        df_combined = pd.concat([df_combined, df_addin], axis=1, ignore_index=True)

    # NaNs are inserted for missing values when concatenating. If it's missing, there were zero infections
    df_combined[np.isnan(df_combined)] = 0
    # Rename the index.
    df_combined.index.set_names(["STATE", "INFECTION_DATE", "SOURCE"], inplace=True)

    # INCLUDE ALL DAYS WITH ZERO INFECTIONS IN THE INDEX AS WELL.
    # Reindex to include days with zero total infections.
    local_infs = df_combined.xs("local", level="SOURCE")
    imported_infs = df_combined.xs("imported", level="SOURCE")
    statelist = [*df_combined.index.get_level_values("STATE").unique()]

    # Should all states have the same start date? Current code starts from the first case in each state.
    # For the same start date:
    local_statedict = dict(zip(statelist, np.repeat(None, len(statelist))))
    imported_statedict = dict(zip(statelist, np.repeat(None, len(statelist))))

    # Determine start date as the first infection date for all.
    # start_date = np.datetime64("2020-02-01")
    start_date = df_combined.index.get_level_values("INFECTION_DATE").min()

    # Determine end dates as the last infected date by state.
    index_only = df_combined.index.to_frame()
    index_only = index_only.reset_index(drop=True)
    maxdates = index_only["INFECTION_DATE"].max()

    for aus_state in statelist:
        state_data = local_infs.xs(aus_state, level="STATE")
        # start_date = state_data.index.min()

        # dftest.index=dftest.reindex(alldates, fill_value=0)

        # All days from start_date to the last infection day.
        alldates = pd.date_range(start_date, maxdates)
        local_statedict[aus_state] = state_data.reindex(alldates, fill_value=0)

    for aus_state in statelist:
        state_data = imported_infs.xs(aus_state, level="STATE")
        alldates = pd.date_range(start_date, maxdates)
        imported_statedict[aus_state] = state_data.reindex(alldates, fill_value=0)

    # Convert dictionaries to data frames
    df_local_inc_zeros = pd.concat(local_statedict)
    df_local_inc_zeros["SOURCE"] = "local"
    df_imp_inc_zeros = pd.concat(imported_statedict)
    df_imp_inc_zeros["SOURCE"] = "imported"

    # Merge dataframes and reindex.
    df_inc_zeros = pd.concat([df_local_inc_zeros, df_imp_inc_zeros])

    df_inc_zeros = df_inc_zeros.reset_index()
    df_inc_zeros = df_inc_zeros.groupby(["level_0", "level_1", "SOURCE"]).sum()
    df_inc_zeros.index = df_inc_zeros.index.rename(
        ["STATE", "INFECTION_DATE", "SOURCE"]
    )

    return df_inc_zeros


def generate_lambda(
    infections,
    gen_disc_pmf=[],
    trunc_days=21,
):
    """
    Given array of infection_dates (N_dates by N_samples), where values are possible
    number of cases infected on this day, generate the force of infection Lambda_t,
    a N_dates-tau by N_samples array.
    Default generation interval parameters taken from Ganyani et al 2020.
    """
    disc_gamma = gen_disc_pmf / sum(gen_disc_pmf)
    ws = disc_gamma[:trunc_days]
    lambda_t = np.zeros(shape=(infections.shape[0] - trunc_days + 1, infections.shape[1]))
    lambda_t[:, 0] = np.convolve(infections[:, 0], ws, mode="valid")

    return lambda_t


def lambda_all_states(df_infection, trunc_days=21, **kwargs):
    """
    Use geenrate lambda on every state
    """
    statelist = [*df_infection.index.get_level_values("STATE").unique()]

    lambda_dict = {}
    for state in statelist:
        df_total_infections = df_infection.groupby(["STATE", "INFECTION_DATE"]).agg(sum)
        lambda_dict[state] = generate_lambda(
            df_total_infections.loc[state].values,
            trunc_days=trunc_days,
            **kwargs
        )

    return lambda_dict


def Reff_from_case(
    cases_by_infection, lamb, prior_a=1, prior_b=5, tau=7, samples=1000, trunc_days=21
):
    """
    Using Cori at al. 2013, given case incidence by date of infection, and the force
    of infection \Lambda_t on day t, estimate the effective reproduction number at time
    t with smoothing parameter \tau.

    cases_by_infection: A T by N array, for T days and N samples
    lamb : A T by N array, for T days and N samples
    """
    csum_incidence = np.cumsum(cases_by_infection)
    # remove first few incidences to align with size of lambda
    # Generation interval length 20
    csum_incidence = csum_incidence[(trunc_days - 1) :]
    csum_lambda = np.cumsum(lamb)
    roll_sum_incidence = csum_incidence[tau:] - csum_incidence[:-tau]
    roll_sum_lambda = csum_lambda[tau:] - csum_lambda[:-tau]
    a = prior_a + roll_sum_incidence
    b = 1 / (1 / prior_b + roll_sum_lambda)

    R = np.random.gamma(a, b)  # shape, scale

    # select first column

    # Need to empty R when there is too few cases...

    return a, b, R


def generate_summary(samples, dates_by="rows"):
    """
    Given an array of samples (T by N) where rows index the dates,
    generate summary statistics and quantiles
    """

    if dates_by == "rows":
        # quantiles of the columns
        ax = 1
    else:
        # quantiles of the rows
        ax = 0
    mean = np.mean(samples, axis=ax)
    bottom, lower, median, upper, top = np.quantile(
        samples, (0.05, 0.25, 0.5, 0.75, 0.95), axis=ax
    )
    std = np.std(samples, axis=ax)
    output = {
        "mean": mean,
        "std": std,
        "bottom": bottom,
        "lower": lower,
        "median": median,
        "upper": upper,
        "top": top,
    }
    return output


def plot_Reff(Reff: dict, dates=None, ax_arg=None, truncate=None, **kwargs):
    """
    Given summary statistics of Reff as a dictionary, plot the distribution over time
    """
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-poster")
    from datetime import datetime as dt

    if ax_arg is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig, ax = ax_arg

    color_cycle = ax._get_lines.prop_cycler
    curr_color = next(color_cycle)["color"]
    if dates is None:
        dates = range(len(Reff["mean"]))

    if truncate is None:
        ax.plot(dates, Reff["mean"], color=curr_color, **kwargs)

        ax.fill_between(
            dates, Reff["lower"], Reff["upper"], alpha=0.4, color=curr_color
        )
        ax.fill_between(dates, Reff["bottom"], Reff["top"], alpha=0.4, color=curr_color)
    else:
        ax.plot(
            dates[truncate[0] : truncate[1]],
            Reff["mean"][truncate[0] : truncate[1]],
            color=curr_color,
            **kwargs
        )

        ax.fill_between(
            dates[truncate[0] : truncate[1]],
            Reff["lower"][truncate[0] : truncate[1]],
            Reff["upper"][truncate[0] : truncate[1]],
            alpha=0.4,
            color=curr_color,
        )
        ax.fill_between(
            dates[truncate[0] : truncate[1]],
            Reff["bottom"][truncate[0] : truncate[1]],
            Reff["top"][truncate[0] : truncate[1]],
            alpha=0.4,
            color=curr_color,
        )
        # plt.legend()

    # grid line at R_eff =1
    ax.set_yticks(
        [1],
        minor=True,
    )
    ax.set_yticks([0, 2, 3], minor=False)
    ax.set_yticklabels([0, 2, 3], minor=False)
    ax.yaxis.grid(which="minor", linestyle="--", color="black", linewidth=2)
    ax.tick_params(axis="x", rotation=90)

    return fig, ax


def plot_all_states(
    R_summ_states,
    df_interim,
    dates,
    start="2020-03-01",
    end="2020-08-01",
    save=True,
    date=None,
    tau=7,
    nowcast_truncation=-10,
    omicron_Reff=False,
):
    """
    Plot results over time for all jurisdictions.

    dates: dictionary of (region, date) pairs where date holds the relevant
            dates for plotting cases by inferred symptom-onset
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    states = df_interim.STATE.unique().tolist()
    # states.remove('NT')

    date_min = pd.to_datetime(end) - timedelta(days=3 * 30)

    date_filter = pd.date_range(start=start, end=end)

    # prepare NNDSS cases where here we are plotting the inferred onset data
    df_cases = df_interim.groupby(["date_inferred", "STATE"]).agg(sum)
    df_cases = df_cases.reset_index()

    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(15, 12))

    for i, state in enumerate(states):
        row = i // 4
        col = i % 4

        R_summary = R_summ_states[state]

        fig, ax[row, col] = plot_Reff(
            R_summary,
            dates=dates[state],
            ax_arg=(fig, ax[row, col]),
            truncate=(0, nowcast_truncation),
            label="Our Model",
        )

        fig, ax[row, col] = plot_Reff(
            R_summary,
            dates=dates[state],
            ax_arg=(fig, ax[row, col]),
            truncate=(nowcast_truncation, None),
            label="Nowcast",
        )

        # plot formatting
        ax[row, col].set_title(state)
        ax[row, col].set_ylim((0, 4))
        # ax[row, col].set_xlim((pd.to_datetime(start), pd.to_datetime(end)))
        ax[row, col].set_xlim((date_min, pd.to_datetime(end)))

        # plot cases behind
        ax2 = ax[row, col].twinx()

        ax2.bar(
            df_cases.loc[df_cases.STATE == state, "date_inferred"],
            df_cases.loc[df_cases.STATE == state, "local"]
            + df_cases.loc[df_cases.STATE == state, "imported"],
            color="grey",
            alpha=0.3,
        )
        ax2.bar(
            df_cases.loc[df_cases.STATE == state, "date_inferred"],
            df_cases.loc[df_cases.STATE == state, "local"],
            color="grey",
            alpha=0.8,
        )

        # Set common labels
        fig.text(0.5, 0.01, "Date", ha="center", va="center", fontsize=20)
        fig.text(
            0.08,
            0.5,
            "Effective \nReproduction Number",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=20,
        )
        fig.text(
            0.95,
            0.5,
            "Local Cases",
            ha="center",
            va="center",
            rotation=270,
            fontsize=20,
        )

    if save:
        import os

        os.makedirs("figs/EpyReff/", exist_ok=True)
        if omicron_Reff:
            plt.savefig("figs/EpyReff/Reff_omicron_tau_" + str(tau) + "_" + date + ".png", dpi=300)
        else:
            plt.savefig("figs/EpyReff/Reff_delta_tau_" + str(tau) + "_" + date + ".png", dpi=300)
    return fig, ax