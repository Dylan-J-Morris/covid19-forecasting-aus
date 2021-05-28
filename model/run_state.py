from sim_class import *
import pandas as pd
from sys import argv
from numpy.random import beta, gamma
from tqdm import tqdm

#from joblib import Parallel, delayed
import multiprocessing as mp

def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj,methname)(*arg[2:])

n_sims=int(argv[1]) #number of sims
start_date = argv[5] 
Reff_file_date = argv[3]#'2020-08-25'
forecast_date = argv[3]#'2020-08-25'
case_file_date = pd.to_datetime(argv[3]).strftime("%d%b%Y")#None #'24Jul'
XBstate = None
test_campaign_date = '2020-06-01'
test_campaign_factor = 1.5

# Get total number of simulation days
num_forecast_days = argv[2]
end_date = pd.to_datetime(forecast_date,format="%Y-%m-%d") + pd.Timedelta(days=num_forecast_days)
end_time = (end_date - pd.to_datetime(start_date,format="%Y-%m-%d")).days # end_time is record as a number of days

progress = True # Used to be argv[5] but was always None
forecast_type = 'R_L'# used to be argv[3]
states = [argv[4]] 
# states =['NSW','QLD','SA','TAS','VIC','WA','ACT','NT'] # old code ran all states but parallel HPC code run separate
print("Simulating state " +states[0])



if pd.to_datetime(argv[3]) < pd.to_datetime('2020-06-02'):
    if pd.to_datetime(argv[3]).day <10:
        #no leading zero on early dates
        case_file_date=case_file_date[1:]

R_I='R_I'
abc =False

# If no VoC specified, code will run without alterations.
variant_of_concern_start_date = None
if len(argv)>6:
    if argv[6] == 'UK':
        # The date from which to increase Reff due to VoC. This date is expressed as the number of days from the start of simulation.
        variant_of_concern_start_date = (pd.to_datetime(forecast_date,format='%Y-%m-%d') - pd.to_datetime(start_date,format='%Y-%m-%d')).days
            
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

forecast_dict = {}
for state in states:
    initial_people = ['I']*current[state][0] + \
            ['A']*current[state][1] + \
            ['S']*current[state][2]
    people = {}
    if abc:
        qs_prior = beta(2,2,size=10000)
        qi_prior = beta(2, 2, size=10000)
        qa_prior = beta(2,2, size=10000)
        #qi_prior = [qi_d[state]]
        #qs_prior = [local_detection[state]]
        #qa_prior = [a_local_detection[state]]
        gam =0.1 + beta(2,2,size=10000) *0.9 #np.minimum(3,gamma(4,0.25, size=1000))
        ps_prior = 0.1+beta(2,2,size=10000)*0.9

    else:
        qi_prior = [qi_d[state]]
        qs_prior = [local_detection[state]]
        qa_prior = [a_local_detection[state]]
        gam =[1/2]
        ps_prior = 0.7
        ps_prior= [ps_prior]

    for i,cat in enumerate(initial_people):
        people[i] = Person(0,0,0,0,cat)
    
    if state in ['VIC']:
        #XBstate = 'SA'
        forecast_dict[state] = Forecast(current[state],
        state,start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam, #alpha_i is impact of importations after April 15th
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
        cross_border_state=XBstate,cases_file_date=case_file_date,
        ps_list = ps_prior, test_campaign_date=test_campaign_date, 
        test_campaign_factor=test_campaign_factor,Reff_file_date=Reff_file_date,
        variant_of_concern_start_date = variant_of_concern_start_date
        )
    elif state in ['NSW']:
        forecast_dict[state] = Forecast(current[state],
        state,start_date,people,
        alpha_i= 1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=2,qua_qi_factor=1,qua_qs_factor=1, #qua_ai is impact of importations before April 15th
        forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
        cross_border_state=None,cases_file_date=case_file_date,
        ps_list = ps_prior,Reff_file_date=Reff_file_date,
        variant_of_concern_start_date = variant_of_concern_start_date
        )
    elif state in ['ACT','NT','SA','WA','QLD']:
        forecast_dict[state] = Forecast(current[state],
        state,start_date,people,
        alpha_i= 0.1, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1,
        forecast_R =forecast_type, R_I = R_I,forecast_date=forecast_date,
        cross_border_state=None,cases_file_date=case_file_date,
        ps_list = ps_prior,Reff_file_date=Reff_file_date,
        variant_of_concern_start_date = variant_of_concern_start_date
        )
    else:
        forecast_dict[state] = Forecast(current[state],state,
        start_date,people,
        alpha_i= 0.5, k =0.1,gam_list=gam,
        qs_list=qs_prior,qi_list=qi_prior,qa_list=qa_prior,
        qua_ai=1,qua_qi_factor=1,qua_qs_factor=1, 
        forecast_R = forecast_type , R_I = R_I,forecast_date=forecast_date,
        cases_file_date=case_file_date,
        ps_list = ps_prior,Reff_file_date=Reff_file_date,
        variant_of_concern_start_date = variant_of_concern_start_date
        )



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


    for key,item in forecast_dict.items():
        #item.read_in_Reff() #now in simulate method
        item.end_time = end_time
        item.read_in_cases()
        item.cross_border_seeds = np.zeros(shape=(end_time,n_sims),dtype=int)
        item.cross_border_state_cases = np.zeros_like(item.cross_border_seeds)

        item.num_bad_sims = 0
        item.num_too_many = 0
    pool = mp.Pool(12)
    with tqdm(total=n_sims) as pbar:
        for cases, obs_cases, param_dict in pool.imap_unordered(worker,
        [(forecast_dict[states[0]],'simulate',end_time,n,n) 
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
                # travel_seeds[:,n] = param_dict['travel_seeds']
                # travel_induced_cases[:,n] = param_dict['travel_induced_cases'+str(XBstate)]


            
            #record cases appropriately
            import_inci[:,n] = cases[:,0]
            asymp_inci[:,n] = cases[:,1]
            symp_inci[:,n] = cases[:,2]

            import_inci_obs[:,n] = obs_cases[:,0]
            asymp_inci_obs[:,n] = obs_cases[:,1]
            symp_inci_obs[:,n] = obs_cases[:,2]
            if progress:
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
    df = item.to_df(results)


