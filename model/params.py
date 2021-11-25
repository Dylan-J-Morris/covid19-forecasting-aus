##### Key parameters
use_linelist = True
use_imputed_linelist = False
on_phoenix = True   # flag for running on phoenix
run_inference = True    # whether the inference should be run
run_inference_only = False
testing_inference = False
testing_sim = False      # this flag will tet the simulation algorithm

if on_phoenix:
    ncores = 12     # number of cores to use (this is relevant for the simulation)
    # this flag will run the inference only and quit after it's done
else: 
    ncores = 4     # number of cores to use (this is relevant for the simulation)
    # this flag will run the inference only and quit after it's done

##### Usually unchanged parameters, contains some dates and number of forecast
third_start_date = '2021-08-15'
start_date = '2021-06-10'  # Start date of forecast
# start_date = '2021-08-01'  # Start date of forecast
VoC_start_date = '2021-05-01'  # Date from which to apply the VoC Reff increase
vaccination_start_date = '2021-02-22'
# Number of days after data date to forecast (usually 35)
num_forecast_days = 35
# setting this to False lets us check that the soc_mob_R_L_hats look ok without the VoC and Vax effects 
# applied. NEED to set to True in order to apply inferred VoC effect properly. This is predominantly for testing
# with and without Vax and has not been removed in case of need to forecast without Vax effect
apply_voc_to_R_L_hats = True
apply_vacc_to_R_L_hats = True 
# alternative application of voc and vaccination effect -- not removed yet in case we need them -- should be left at False
use_vaccine_effect = False
use_voc_effect = False
# The ratio of true cases to simulation cases below which we insert cases into branching process
case_insertion_threshold = 5
# Will download Google data automatically on run. Set to False for repeated runs. False is the preferable 
# setting.
download_google_automatically = False
assume_local_cases_if_unknown = True
# number of days to remove to stop the issues with the right-truncation
truncation_days = 15

##### Simulation parameters/transmission parameters
# incubation period: taken from Lauer et al. 2020
shape_inc = 5.807  
scale_inc = 0.948
offset_inc = 0
## reporting delay distribution: empirically estimated from the case data using MLE
# looked at duration between symptom onset and cofnirmation for cases where this was 
# feasible and truncated this to be between 0 and 30 (plenty of retropsective cases with negatives etc)
shape_rd = 1.28
scale_rd = 2.31
offset_rd = 0
## generation interval: 
# generation inteval changed Oct 5 2021
# shape_gen = 3.64/3.07
# scale_gen = 3.07
shape_gen = 2.75
scale_gen = 1.00
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
    'NSW': 0.15,
    'QLD': 0.1,
    'SA': 0.1,
    'TAS': 0.1,
    'VIC': 0.15,
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

# alpha_i is impact of importations after April 15th. These have been set to 1 as we not long believe there are significant differences between hotel quarentine effectiveness between states.
alpha_i_all = 1

alpha_i = {'NSW': alpha_i_all,
           'QLD': alpha_i_all,
           'SA': alpha_i_all,
           'TAS': alpha_i_all,
           'VIC': alpha_i_all,
           'WA': alpha_i_all,
           'ACT': alpha_i_all,
           'NT': alpha_i_all}
