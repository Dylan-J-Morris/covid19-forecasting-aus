"""
Control panel for the fitting and forecasting procedures. The key parameters
are usually needed to be adjusted based on properties of the simulation (i.e. 
what machine is used, number of samples etc). This contains the bulk of the 
assumptions used in the fitting and should be referred to when checking the 
assumptions. 
"""

from scipy.stats import gamma
import numpy as np 
import pandas as pd    

##### Key parameters #####
use_linelist = True
use_TP_adjustment = True
n_days_nowcast_TP_adjustment = 30
# number of forecasted TP samples to save (2000 appears to work well, but could probably get away
# with fewer if memory is an issue).
mob_samples = 2000  
# number of days to remove to stop the issues with the right-truncation
truncation_days = 14
# Number of days after data date to forecast (usually 35)
num_forecast_days = 35

##### Usually unchanged parameters, contains some dates and number of forecast #####
p_detect_delta = 0.75
p_detect_omicron = 0.5  # default

# this is the earliest date the simulations are run from and helps with plotting
sim_start_date = "2021-06-25"
# the start of the third wave is used for a lot of calibration across fitting and forecasting and 
# so is included here
third_start_date = "2021-06-25"
# Date from which to apply the VoC Reff increase from particular strains
alpha_start_date = "2020-12-01"  
delta_start_date = "2021-05-01"  
omicron_start_date = "2021-11-15"  
omicron_only_date = "2022-02-01"  
# the date at which omicron is assumed to be dominant (this is deprecated but kept for logic)
omicron_dominance_date = "2021-12-15"
# vaccination program began mid Feb 2021
vaccination_start_date = "2021-02-21"
# start date for the TP forecasts (might be able to try moving this to ensure we can reduce memory
# usage)?
start_date = "2020-03-01"

##### Simulation parameters/transmission parameters #####
## reporting delay distribution: 
# empirically estimated from the case data using MLE looked at duration between symptom onset 
# and cofnirmation for cases where this was feasible and truncated this to be between 0 and 30 
# (plenty of retropsective cases with negatives etc)
# range(22) = 0:21 
(shape_rd, scale_rd) = (2.33, 1.35)
rd_disc_pmf = [gamma.pdf(x, a=shape_rd, scale=scale_rd) for x in range(22)]

# incubation period: taken from Lauer et al. 2020
(shape_inc, scale_inc) = (5.807, 0.948)
inc_disc_pmf = [gamma.pdf(x, a=shape_inc, scale=scale_inc) for x in range(22)]
# omicron incubation period determined by sampling Delta incubation periods and subtracting 1 
# (then taking those with days > 0.05) and using MLE to fit a Gamma distribution
(shape_inc_omicron, scale_inc_omicron) = (3.581, 1.257)
inc_omicron_disc_pmf = [
    gamma.pdf(x, a=shape_inc_omicron, scale=scale_inc_omicron) for x in range(22)
]

## generation interval:
# generation inteval changed Oct 5 2021
(shape_gen, scale_gen) = (2.75, 1.00)
gen_disc_pmf = [gamma.pdf(x, a=shape_gen, scale=scale_gen) for x in range(22)]
# omicron GI determined by sampling Delta GI and subtracting 1 (then taking those with days > 0.05)
# and using MLE to fit a Gamma distribution
(shape_gen_omicron, scale_gen_omicron) = (1.389, 1.415)
gen_omicron_disc_pmf = [
    gamma.pdf(x, a=shape_gen_omicron, scale=scale_gen_omicron) for x in range(22)
]

# pulled from
# https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release
pop_sizes = {
    "ACT": 432266,
    "NSW": 8189266,
    "NT": 246338,
    "QLD": 5221170,
    "SA": 1773243,
    "TAS": 541479,
    "VIC": 6649159,
    "WA": 2681633,
}

def get_p_detect_old_assumptions(end_date):
    """
    Apply a scaling to the daily reported cases by accounting for a ~75% detection probability pre
    15/12/2021 and 0.5 following that. To improve the transition, we assume that detection 
    probability decreases from 0.75 to 0.5 over 7 days beginning 9/12/2021.
    """
    
    # CA is related to detection of actual cases but our fitting works with infection dates. 
    # deal with this by subtracting 1 mean incubation period from the CA
    CAR_normal_before = pd.date_range(third_start_date, "2021-12-09")
    CAR_decrease_range = pd.date_range("2021-12-10", end_date)
    
    CAR_dates = np.concatenate(
        (
            CAR_normal_before, 
            CAR_decrease_range,
        )
    )
    
    # get baseline CAR
    CAR = 0.75 * np.ones(CAR_dates.shape)
    # apply a step decrease in the CAR 
    after_bool = CAR_dates >= CAR_decrease_range[0]
    CAR[after_bool] = 0.5
    
    return CAR
    
    
def get_p_detect_big_jurisdictions(end_date):
    """
    Apply a scaling to the daily reported cases by accounting for a ~75% detection probability pre
    15/12/2021 and 0.5 following that. To improve the transition, we assume that detection 
    probability decreases from 0.75 to 0.5 over 7 days beginning 9/12/2021.
    """
    
    CAR_normal_before = pd.date_range(third_start_date, "2021-12-07")
    CAR_decrease_range = pd.date_range("2021-12-08", "2022-01-17")
    CAR_normal_after = pd.date_range("2022-01-18", end_date)
    
    CAR_dates = np.concatenate(
        (
            CAR_normal_before, 
            CAR_decrease_range,
            CAR_normal_after,
        )
    )
    
    # get baseline CAR
    CAR = 0.75 * np.ones(CAR_dates.shape)
    # determine index arrays for the various phases assumed
    decrease_bool = (CAR_dates >= CAR_decrease_range[0]) & (CAR_dates <= CAR_decrease_range[-1])
    # adjust the CAR based on the time varying assumptions by approximating the step change 
    # linearly 
    CAR[decrease_bool] = 0.333
    
    return CAR
    
    
def get_p_detect_small_jurisdictions(end_date):
    """
    Apply a scaling to the daily reported cases by accounting for a ~75% detection probability pre
    15/12/2021 and 0.5 following that. To improve the transition, we assume that detection 
    probability decreases from 0.75 to 0.5 over 7 days beginning 9/12/2021.
    """
    
    CAR_dates = pd.date_range(third_start_date, end_date)
    
    # get baseline CAR
    CAR = 0.75 * np.ones(CAR_dates.shape)
    
    return CAR


def get_all_p_detect(end_date, num_days):
    states = sorted(["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])
    
    p_detect = np.zeros((num_days, len(states)))
    
    for i, s in enumerate(states):
        if s in ("NSW", "ACT", "QLD", "VIC"):
            p_detect[:,i] = get_p_detect_big_jurisdictions(end_date)        
        else:
            p_detect[:,i] = get_p_detect_small_jurisdictions(end_date)
    
    return p_detect


def get_all_p_detect_old(end_date, num_days):
    states = sorted(["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])
    
    p_detect = np.zeros((num_days, len(states)))
    
    for i, s in enumerate(states):
        p_detect[:,i] = get_p_detect_old_assumptions(end_date)
    
    return p_detect

