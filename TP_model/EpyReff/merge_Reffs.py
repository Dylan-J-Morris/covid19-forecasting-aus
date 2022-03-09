###
# Merge the Reff files
###

from sys import argv
import sys
sys.path.insert(0, "TP_model")
sys.path.insert(0, "TP_model/EpyReff")
from params import omicron_dominance_date
from epyreff import *
import pandas as pd

date = argv[1]
dt_date = pd.to_datetime(date, format="%Y-%m-%d")
file_date = dt_date.strftime("%Y-%m-%d")

# read in and process summary files used for the fitting 
R = pd.read_csv("results/EpyReff/Reff_delta" + file_date + "tau_5.csv")
R_omicron = pd.read_csv("results/EpyReff/Reff_omicron" + file_date + "tau_5.csv")

R_merge = pd.concat(
    [
        R.loc[R.INFECTION_DATES < omicron_dominance_date], 
        R_omicron.loc[R_omicron.INFECTION_DATES >= omicron_dominance_date],
    ]
)

R_merge.to_csv("results/EpyReff/Reff" + file_date + "tau_5.csv")

# read in and process the Reff samples (same code, just different files)
R = pd.read_csv("results/EpyReff/Reff_delta_samples" + file_date + "tau_5.csv")
R_omicron = pd.read_csv("results/EpyReff/Reff_omicron_samples" + file_date + "tau_5.csv")

R_merge = pd.concat(
    [
        R.loc[R.INFECTION_DATES < omicron_dominance_date], 
        R_omicron.loc[R_omicron.INFECTION_DATES >= omicron_dominance_date],
    ]
)

R_merge.to_csv("results/EpyReff/Reff_samples" + file_date + "tau_5.csv")



