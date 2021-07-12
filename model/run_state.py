from sim_class import *
import pandas as pd
from sys import argv
from numpy.random import beta, gamma
from tqdm import tqdm
import multiprocessing as mp


n_sims=int(argv[1]) #number of sims
from params import start_date, num_forecast_days
state = argv[3]
print("Simulating state " +state)


# Get total number of simulation days
forecast_date = argv[2] # Date of forecast
end_date = pd.to_datetime(forecast_date,format="%Y-%m-%d") + pd.Timedelta(days=num_forecast_days)
end_time = (end_date - pd.to_datetime(start_date,format="%Y-%m-%d")).days # end_time is recorded as a number of days
case_file_date = pd.to_datetime(forecast_date).strftime("%d%b%Y") # Convert date to format used in case file


# If no VoC specified, code will run without alterations.
VoC_flag = ''
if len(argv)>4:
    VoC_flag =  argv[4]

if len(argv) > 5:
    # Add an optional scenario flag to load in specific Reff scenarios and save results. This does not change the run behaviour of the simulations.
    scenario = argv[5]
else:
    scenario = ''
            
local_detection = {
            'NSW':0.9,#0.556,#0.65,
            'QLD':0.9,#0.353,#0.493,#0.74,
            'SA':0.7,#0.597,#0.75,
            'TAS':0.4,#0.598,#0.48,
            'VIC':0.35,#0.558,#0.77,
            'WA':0.7,#0.409,#0.509,#0.66,
            'ACT':0.95,#0.557,#0.65,
            'NT':0.95,#0.555,#0.71
        }

a_local_detection = {
            'NSW':0.05,#0.556,#0.65,
            'QLD':0.05,#0.353,#0.493,#0.74,
            'SA':0.05,#0.597,#0.75,
            'TAS':0.05,#0.598,#0.48,
            'VIC':0.05,#0.558,#0.77,
            'WA':0.05,#0.409,#0.509,#0.66,
            'ACT':0.7,#0.557,#0.65,
            'NT':0.7,#0.555,#0.71
        }

qi_d = {
            'NSW':0.98,#0.758,
            'QLD':0.98,#0.801,
            'SA':0.98,#0.792,
            'TAS':0.98,#0.800,
            'VIC':0.98,#0.735,
            'WA':0.98,#0.792,
            'ACT':0.98,#0.771,
            'NT':0.98,#0.761
    }

##Initialise the number of cases as 1st of March data incidence
if start_date=="2020-03-01":
    current = {
        'ACT':[0,0,0],
        'NSW':[10,0,2], #1
        'NT':[0,0,0],
        'QLD':[2,0,0],
        'SA':[2,0,0],
        'TAS':[0,0,0],
        'VIC':[2,0,0], #1
        'WA':[0,0,0],
    } 
elif start_date=="2020-09-01":
    current = {
        'ACT':[0,0,0],
        'NSW':[3,0,7], #1
        'NT':[0,0,0],
        'QLD':[0,0,3],
        'SA':[0,0,0],
        'TAS':[0,0,0],
        'VIC':[0,0,60], #1
        'WA':[1,0,0],
    }
elif start_date == "2020-12-01":
    current = { # based on locally acquired cases in the days preceding the start date
        'ACT': [0, 0, 0],
        'NSW': [0, 0, 1], 
        'NT': [0, 0, 0],
        'QLD': [0, 0, 1],
        'SA': [0, 0, 0],
        'TAS': [0, 0, 0],
        'VIC': [0, 0, 0], 
        'WA': [0, 0, 0],
    }
else:
    print("Start date not implemented") 

initial_people = ['I']*current[state][0] + \
        ['A']*current[state][1] + \
        ['S']*current[state][2]

people = {}
for i,cat in enumerate(initial_people):
    people[i] = Person(0,0,0,0,cat)


####### Create simulation.py object ########

if state in ['VIC']:
    forecast_object = Forecast(current[state],
    state,start_date,people,
    alpha_i= 1, #alpha_i is impact of importations after April 15th
    qs=local_detection[state],qi=qi_d[state],qa=a_local_detection[state],
    qua_ai=1, forecast_date=forecast_date,
    cases_file_date=case_file_date, 
    VoC_flag = VoC_flag, scenario=scenario
    )
elif state in ['NSW']:
    forecast_object = Forecast(current[state],
    state,start_date,people,
    alpha_i= 1,
    qs=local_detection[state],qi=qi_d[state],qa=a_local_detection[state],
    qua_ai=2, #qua_ai is impact of importations before April 15th 
    forecast_date=forecast_date,
    cases_file_date=case_file_date,
    VoC_flag = VoC_flag, scenario=scenario
    )
elif state in ['ACT','NT','SA','WA','QLD']:
    forecast_object = Forecast(current[state],
    state,start_date,people,
    alpha_i= 0.1,
    qs=local_detection[state],qi=qi_d[state],qa=a_local_detection[state],
    qua_ai=1, forecast_date=forecast_date,
    cases_file_date=case_file_date,
    VoC_flag = VoC_flag, scenario=scenario
    )
else:
    forecast_object = Forecast(current[state],state,
    start_date,people,
    alpha_i= 0.5,
    qs=local_detection[state],qi=qi_d[state],qa=a_local_detection[state],
    qua_ai=1,  forecast_date=forecast_date,
    cases_file_date=case_file_date,
    VoC_flag = VoC_flag, scenario=scenario
    )


############ Run Simulations in parallel and return ############

def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj,methname)(*arg[2:])

if __name__ =="__main__":
    ##initialise arrays

    import_sims = np.zeros(shape=(end_time, n_sims), dtype=float)
    import_sims_obs = np.zeros_like(import_sims)
    

    import_inci = np.zeros_like(import_sims)
    import_inci_obs = np.zeros_like(import_sims)

    asymp_inci = np.zeros_like(import_sims)
    asymp_inci_obs = np.zeros_like(import_sims)

    symp_inci = np.zeros_like(import_sims)
    symp_inci_obs = np.zeros_like(import_sims)

    bad_sim = np.zeros(shape=(n_sims),dtype=int)

    travel_seeds = np.zeros(shape=(end_time,n_sims),dtype=int)
    travel_induced_cases = np.zeros_like(travel_seeds)

    #ABC parameters
    metrics = np.zeros(shape=(n_sims),dtype=float)
    qs = np.zeros(shape=(n_sims),dtype=float)
    qa = np.zeros_like(qs)
    qi = np.zeros_like(qs)
    alpha_a = np.zeros_like(qs)
    alpha_s = np.zeros_like(qs)
    accept = np.zeros_like(qs)
    ps = np.zeros_like(qs)
    cases_after = np.zeros_like(bad_sim)

    forecast_object.end_time = end_time
    forecast_object.read_in_cases()

    forecast_object.num_bad_sims = 0
    forecast_object.num_too_many = 0

    pool = mp.Pool(12)
    with tqdm(total=n_sims, leave=False, smoothing=0, miniters=500) as pbar:
        for cases, obs_cases, param_dict in pool.imap_unordered(worker,
        [(forecast_object,'simulate',end_time,n,n) 
        for n in range(n_sims)] #n is the seed
                        ):
            #cycle through all results and record into arrays 
            n = param_dict['num_of_sim']
            if param_dict['bad_sim']:
                #bad_sim True
                bad_sim[n] = 1
            else:
                #good sims
                ## record all parameters and metric
                metrics[n] = param_dict['metric']
                qs[n] = param_dict['qs']
                qa[n] = param_dict['qa']
                qi[n] = param_dict['qi']
                alpha_a[n] = param_dict['alpha_a']
                alpha_s[n] = param_dict['alpha_s']
                accept[n] = param_dict['metric']>=0.8
                cases_after[n] = param_dict['cases_after']
                ps[n] =param_dict['ps']

            
            #record cases appropriately
            import_inci[:,n] = cases[:,0]
            asymp_inci[:,n] = cases[:,1]
            symp_inci[:,n] = cases[:,2]

            import_inci_obs[:,n] = obs_cases[:,0]
            asymp_inci_obs[:,n] = obs_cases[:,1]
            symp_inci_obs[:,n] = obs_cases[:,2]
            pbar.update()
        
    pool.close()
    pool.join()

    


    

    #convert arrays into df
    results = {
        'imports_inci': import_inci,
        'imports_inci_obs': import_inci_obs,
        'asymp_inci': asymp_inci,
        'asymp_inci_obs': asymp_inci_obs,
        'symp_inci': symp_inci,
        'symp_inci_obs': symp_inci_obs,
        'total_inci_obs': symp_inci_obs + asymp_inci_obs,
        'total_inci': symp_inci + asymp_inci,
        'all_inci': symp_inci + asymp_inci + import_inci,
        'bad_sim': bad_sim,
        'metrics': metrics,
        'accept': accept,
        'qs':qs,
        'qa':qa,
        'qi':qi,
        'alpha_a':alpha_a,
        'alpha_s':alpha_s,
        'cases_after':cases_after,
        # 'travel_seeds': travel_seeds,
        # 'travel_induced_cases'+str(item.cross_border_state): travel_induced_cases,
        'ps':ps,
    }
    print("Number of bad sims is %i" % sum(bad_sim))
    #print("Number of sims in "+state\
    #        +" exceeding "+\
    #            "max cases is "+str(sum()) )
    #results recorded into parquet as dataframe
    df = forecast_object.to_df(results)


