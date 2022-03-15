"""
Control panel for the fitting and forecasting procedures. The key parameters
are usually needed to be adjusted based on properties of the simulation (i.e. 
what machine is used, number of samples etc). This contains the bulk of the 
assumptions used in the fitting and should be referred to when checking the 
assumptions. 
"""

from scipy.stats import gamma

##### Key parameters #####
use_linelist = True
use_local_cases_input = True
use_TP_adjustment = True
testing_inference = False
n_days_nowcast_TP_adjustment = 30
# number of forecasted TP samples to save
num_TP_samples = 2000  
# number of days to remove to stop the issues with the right-truncation
truncation_days = 14
# Number of days after data date to forecast (usually 35)
num_forecast_days = 35

##### Usually unchanged parameters, contains some dates and number of forecast #####
p_detect_delta = 0.725  # increased slightly due to comparisons between Delta and Omicron waves
# p_detect_omicron = 0.64  # default
p_detect_omicron = 0.5  # default
# p_detect_omicron = 0.375
# p_detect_omicron = 0.25

third_start_date = "2021-06-25"
start_date = "2021-06-25"
# Date from which to apply the VoC Reff increase from particular strains (based on Reff model)
alpha_start_date = "2020-12-01"  
delta_start_date = "2021-05-01"  
omicron_start_date = "2021-11-15"  
omicron_dominance_date = "2021-12-15"
# vaccination program began mid Feb 2021
vaccination_start_date = "2021-02-21"

start_dates = {
    "NSW": start_date,
    "QLD": "2021-06-23",
    "SA": "2021-06-23",
    "TAS": "2021-06-23",
    "WA": "2021-06-23",
    "ACT": "2021-06-23",
    "NT": "2021-06-23",
    "VIC": "2021-08-01",
}


# Will download Google data automatically on run. Set to False for repeated runs. 
# False is the preferable setting.
download_google_automatically = False
# assume local cases in the absence of a POI 
assume_local_cases_if_unknown = True

##### Simulation parameters/transmission parameters #####
## reporting delay distribution: 
# empirically estimated from the case data using MLE looked at duration between symptom onset 
# and cofnirmation for cases where this was feasible and truncated this to be between 0 and 30 
# (plenty of retropsective cases with negatives etc)
# (shape_rd, scale_rd) = (1.28, 2.31)
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