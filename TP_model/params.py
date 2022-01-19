"""
Control panel for the fitting and forecasting procedures. The key parameters
are usually needed to be adjusted based on properties of the simulation (i.e. 
what machine is used, number of samples etc). This contains the bulk of the 
assumptions used in the fitting and should be referred to when checking the 
assumptions. 
"""

##### Key parameters

use_linelist = True
use_imputed_linelist = False
on_phoenix = False   # flag for running on phoenix
run_inference = True    # whether the inference should be run
run_inference_only = False
run_TP_adjustment = False
use_TP_adjustment = False
testing_inference = False
n_days_nowcast_TP_adjustment = 45
num_TP_samples = 2000       # number of forecasted TP samples to save 
# number of days to remove to stop the issues with the right-truncation
truncation_days = 16
# Number of days after data date to forecast (usually 35)
num_forecast_days = 35

if on_phoenix:
    ncores = 12     # number of cores to use (this is relevant for the simulation)
else: 
    ncores = 4     # number of cores to use (this is relevant for the simulation)

##### Usually unchanged parameters, contains some dates and number of forecast

third_start_date = '2021-06-15'
start_date = '2021-06-23'
omicron_dominance_date = '2021-12-15'

start_dates = {
    'NSW': start_date,
    'QLD': '2021-06-23',
    'SA': '2021-06-23',
    'TAS': '2021-06-23',
    'WA': '2021-06-23',
    'ACT': '2021-06-23',
    'NT': '2021-06-23',
    'VIC': '2021-08-01',
}

alpha_start_date = '2020-12-01'  # Date from which to apply the VoC Reff increase from alpha (based on Reff model) 
delta_start_date = '2021-05-01'  # Date from which to apply the VoC Reff increase from deltas (based on Reff model)
omicron_start_date = '2021-11-15'  # Date from which to apply the VoC Reff increase from deltas (based on Reff model)
vaccination_start_date = '2021-02-21'

# Will download Google data automatically on run. Set to False for repeated runs. False is the preferable 
# setting.
download_google_automatically = False
assume_local_cases_if_unknown = True

##### Simulation parameters/transmission parameters

# incubation period: taken from Lauer et al. 2020
(shape_inc, scale_inc) = (5.807, 0.948)
# omicron incubation period determined by sampling Delta incubation periods and subtracting 1 (then taking those with days > 0.05) 
# and using MLE to fit a Gamma distribution
(shape_inc_omicron, scale_inc_omicron) = (3.33, 1.34)
offset_inc = 0

## reporting delay distribution: empirically estimated from the case data using MLE
# looked at duration between symptom onset and cofnirmation for cases where this was 
# feasible and truncated this to be between 0 and 30 (plenty of retropsective cases with negatives etc)
(shape_rd, scale_rd) = (1.28, 2.31)
offset_rd = 0

## generation interval: 
# generation inteval changed Oct 5 2021
(shape_gen, scale_gen) = (2.75, 1.00)
# omicron GI determined by sampling Delta GI and subtracting 1 (then taking those with days > 0.05) 
# and using MLE to fit a Gamma distribution
(shape_gen_omicron, scale_gen_omicron) = (1.58, 1.32)
offset_gen = 0

# Heterogeneity parameter for a negative binomial offspring distribution
# informed from:
# Endo A; Centre for the Mathematical Modelling of
# Infectious Diseases COVID-19 Working Group, Abbott S, Kucharski AJ, Funk S. Estimating the overdispersion in
# COVID-19 transmission using outbreak sizes outside China. Wellcome Open Res. 2020 Jul 10;5:67.
# doi:10.12688/wellcomeopenres.15842.3.
k = 0.15

# Also known as qs, this is the probability of detecting an symptomatic case. This will go up during major testing drives. Increasing qs increases the observed outbreak.
local_detection = {
    'NSW': 0.95,
    'QLD': 0.95,
    'SA': 0.95,
    'TAS': 0.95,
    'VIC': 0.95,
    'WA': 0.95,
    'ACT': 0.95,
    'NT': 0.95
}

# Also known as qa, this is the probability of detecting an asymptomatic case.
a_local_detection = {
    'NSW': 0.1,
    'QLD': 0.1,
    'SA': 0.1,
    'TAS': 0.1,
    'VIC': 0.1,
    'WA': 0.1,
    'ACT': 0.1,
    'NT': 0.1
}

# probability of detecting an imported case 
qi_d = {
    'NSW': 0.98,
    'QLD': 0.98,
    'SA': 0.98,
    'TAS': 0.98,
    'VIC': 0.98,
    'WA': 0.98,
    'ACT': 0.98,
    'NT': 0.98
}

# alpha_i is impact of importations after April 15th. These have been set to 1 as we not long believe 
# there are significant differences between hotel quarentine effectiveness between states.
alpha_i_all = 1

alpha_i = {
    'NSW': alpha_i_all,
    'QLD': alpha_i_all,
    'SA': alpha_i_all,
    'TAS': alpha_i_all,
    'VIC': alpha_i_all,
    'WA': alpha_i_all,
    'ACT': alpha_i_all,
    'NT': alpha_i_all
}

# pulled from 
# https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release
pop_sizes = {
    'NSW': 8189266,
    'QLD': 5221170,
    'SA': 1773243,
    'TAS': 541479,
    'VIC': 6649159,
    'WA': 2681633,
    'ACT': 432266,
    'NT': 246338
}
