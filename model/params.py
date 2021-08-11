
VoC_start_date = '2021-05-01' # Date from which to apply the VoC Reff increase
use_linelist = True # If something goes wrong on a day you can set this to True to use the linelist
assume_local_cases_if_unknown = True
start_date = '2021-04-15' # Start date of forecast
num_forecast_days=35 # Number of days after data date to forecast (usually 35)

case_insertion_threshold = 5 # The ratio of true cases to simulation cases below which we insert cases into branching process
use_vaccine_effect = False
use_voc_effect = False
download_google_automatically = False # Will download Google data automatically on run. Set to false for repeated runs.

# Transmission parameters     

k = 0.1 #  Heterogeneity parameter for a negative binomial offspring distribution

local_detection = {
            'NSW':0.9,
            'QLD':0.9,
            'SA':0.7,
            'TAS':0.4,
            'VIC':0.35,
            'WA':0.7,
            'ACT':0.95,
            'NT':0.95,
        }

a_local_detection = {
            'NSW':0.05,
            'QLD':0.05,
            'SA':0.05,
            'TAS':0.05,
            'VIC':0.05,
            'WA':0.05,
            'ACT':0.7,
            'NT':0.7,
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

 #alpha_i is impact of importations after April 15th
alpha_i = {
            'NSW':1,
            'QLD':0.1,
            'SA':0.1,
            'TAS':0.5,
            'VIC':1,
            'WA':0.1,
            'ACT':0.1,
            'NT':0.1,
        }