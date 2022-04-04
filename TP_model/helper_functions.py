# A collection of helper functions that are used throughout. This file is aimed to avoid replication of code.

import pandas as pd
from scipy.stats import rv_discrete
import numpy as np

def sample_discrete_dist(dist_disc_unnorm, nsamples):
    """
    Samples from the (unnormalised) discrete distribution nsamples times efficiently using C.
    """
    # this returns a number in the range [1, 21] corresponding 
    res = (
        rv_discrete(values=(range(1, 22), dist_disc_unnorm / sum(dist_disc_unnorm)))
        .rvs(size=nsamples)
    )
    
    return res 


def read_in_NNDSS(
    date_string,
    apply_delay_at_read=False,
    apply_inc_at_read=False,
):
    """
    A general function to read in the NNDSS data. Alternatively this can be manually
    set to read in the linelist instead.
    Args:
        date_string: (str) a string of the date of the data file.

    Returns:
        A dataframe of all NNDSS data.
    """

    from datetime import timedelta
    import glob
    from params import (
        rd_disc_pmf,
        inc_disc_pmf,
        inc_omicron_disc_pmf,
        omicron_dominance_date,
    )

    # The linelist, currently produce by Gerry Ryan, has had the onset dates and local / 
    # imported status vetted by a human. This can be a lot more reliable during an outbreak
    case_file_date = pd.to_datetime(date_string).strftime("%Y-%m-%d")
    path = "data/interim_linelist_" + case_file_date + "*.csv"

    for file in glob.glob(path):  # Allows us to use the * option
        df = pd.read_csv(file, low_memory=False)

    if len(glob.glob(path)) == 0:
        raise FileNotFoundError(
            "Calculated linelist not found. Did you want to use NNDSS or the imputed linelist?"
        )

    # take the representative dates
    df["date_onset"] = pd.to_datetime(df["date_onset"], errors="coerce")
    # create boolean of when confirmation dates used
    df["date_confirmation"] = pd.to_datetime(df["date_confirmation"], errors="coerce")
    df["is_confirmation"] = df["date_onset"].isna()
    # set the known onset dates
    df["date_inferred"] = df["date_onset"]

    if apply_delay_at_read:
        # calculate number of delays to sample
        n_delays = df["date_inferred"].isna().sum()
        # sample that number of delays from the distribution and take the ceiling.
        # This was fitted to the third and second wave data, looking at the common differences
        # between onsets and confirmations
        # missing_onset_date = (df["state"][df["date_inferred"].isna()]).to_numpy()
        # subtract 1 as report delay of 0 days is reasonable
        rd = sample_discrete_dist(rd_disc_pmf, n_delays) - 1
        rd = rd * timedelta(days=1)

        # fill missing days with the confirmation date, noting that this is adjusted when used
        df.loc[df["date_inferred"].isna(), "date_inferred"] = (
            df.loc[df["date_inferred"].isna(), "date_confirmation"] - rd
        )
    else:
        # just apply the confirmation date and let EpyReff handle the delay distribution
        df.loc[df["date_inferred"].isna(), "date_inferred"] = df.loc[
            df["date_inferred"].isna(), "date_confirmation"
        ]

    # now we apply the incubation period to the inferred onset date. Note that this should 
    # never be done in the absence of the delay
    if apply_inc_at_read:
        # assuming that the date_onset field is valid, this is the actual date that 
        # individuals get symptoms
        n_infs = df["date_inferred"].shape[0]
        inc = sample_discrete_dist(inc_disc_pmf, n_infs)
        inc_omicron = sample_discrete_dist(inc_omicron_disc_pmf, n_infs)
        is_omicron_dominant = (
            df["date_inferred"] >= pd.to_datetime(omicron_dominance_date)
        ).to_numpy()
        inc = (1 - is_omicron_dominant) * inc + is_omicron_dominant * inc_omicron

        # need to take the ceiling of the incubation period as otherwise the merging 
        # in generate_posterior doesnt work properly
        inc = inc * timedelta(days=1)
        df["date_inferred"] = df["date_inferred"] - inc

    df["imported"] = [1 if stat == "imported" else 0 for stat in df["import_status"]]
    df["local"] = 1 - df.imported
    df["STATE"] = df["state"]

    return df


def read_in_Reff_file(file_date, results_dir="results/", adjust_TP_forecast=False):
    """
    Read in Reff h5 file produced by generate_RL_forecast.
    """

    if file_date is None:
        raise Exception("Need to provide file date to Reff read.")

    file_date = pd.to_datetime(file_date).strftime("%Y-%m-%d")

    if adjust_TP_forecast:
        df_forecast = pd.read_hdf(
            results_dir + "soc_mob_R_adjusted" + file_date + ".h5", key="Reff"
        )
    else:
        df_forecast = pd.read_hdf(
            results_dir + "soc_mob_R" + file_date + ".h5", key="Reff"
        )

    return df_forecast
