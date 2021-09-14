########### Inference parameters
run_inference = True
run_inference_only = False
testing_inference = True        # use this to test the inference methods -- fewer chains and samples -- easy to debug and check summary output

if testing_inference:
    num_chains = 2
    num_samples = 1000
else:
    num_chains = 4
    num_samples = 4000


########### Key parameters
ncores = 4
third_start_date = '2021-06-15'
start_date = '2021-06-01' # Start date of forecast
use_linelist = True # If something goes wrong on a day you can set this to True to use the linelist
VoC_start_date = '2021-05-01' # Date from which to apply the VoC Reff increase
vaccination_start_date = '2021-02-22'
num_forecast_days=35 # Number of days after data date to forecast (usually 35)

# setting this to False lets us check that the soc_mob_R_L_hats look ok without the VoC effect applied
# NEED to set to True in order to apply inferred VoC effect properly
apply_voc_to_R_L_hats = True
apply_vacc_to_R_L_hats = True      


############ Optional parameters - require less frequent tuning 
# alternative application of voc and vaccination effect -- not removed yet in case we need them -- should be left at False
use_vaccine_effect = False
use_voc_effect = False
case_insertion_threshold = 5 # The ratio of true cases to simulation cases below which we insert cases into branching process
download_google_automatically = False           # Will download Google data automatically on run. Set to false for repeated runs.
assume_local_cases_if_unknown = True
# number of days to remove to stop the issues with the right-truncation 
# we have 6 days here cause this is the average amount of the incubation period
# the other 10 come from standard censoring as we are inferring the infection dates
# based on time distributions 
truncation_days = 16            


############# Simulation parameters
# Transmission parameters     
k = 0.15 #  Heterogeneity parameter for a negative binomial offspring distribution

# Also known as qs, this is the probability of detecting an symptomatic case. This will go up during major testing drives. Increasing qs increases the observed outbreak. 
local_detection = {
            'NSW':0.95,
            'QLD':0.95,
            'SA':0.95,
            'TAS':0.95,
            'VIC':0.95,
            'WA':0.95,
            'ACT':0.95,
            'NT':0.95,
        }

# Also known as qa, this is the probability of detecting an asymptomatic case.
a_local_detection = {
            'NSW':0.15,
            'QLD':0.1,
            'SA':0.1,
            'TAS':0.1,
            'VIC':0.15,
            'WA':0.1,
            'ACT':0.1,
            'NT':0.1,
        }

qi_d = {
            'NSW':0.98,
            'QLD':0.98,
            'SA':0.98,
            'TAS':0.98,
            'VIC':0.98,
            'WA':0.98,
            'ACT':0.98,
            'NT':0.98,
        }

# alpha_i is impact of importations after April 15th. These have been set to 1 as we not long believe there are significant differences between hotel quarentine effectiveness between states.
alpha_i_all = 0.5
alpha_i = {
    'NSW':alpha_i_all,
    'QLD':alpha_i_all,
    'SA':alpha_i_all,
    'TAS':alpha_i_all,
    'VIC':alpha_i_all,
    'WA':alpha_i_all,
    'ACT':alpha_i_all,
    'NT':alpha_i_all
    }
