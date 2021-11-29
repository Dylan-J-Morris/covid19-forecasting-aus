import numpy as np
import pandas as pd
from scipy.stats import erlang, beta, gamma, poisson, norm, nbinom, binom
import math
import matplotlib.pyplot as plt
import os
from helper_functions import read_in_NNDSS, read_in_Reff_file
from params import case_insertion_threshold

from collections import deque
import gc
from itertools import cycle

from params import scale_gen, shape_gen, offset_gen, scale_inc, shape_inc, scale_rd, shape_rd, offset_rd, offset_inc

# import line_profiler
# from numba import njit
class Person:
    """
    Individuals in the forecast
    """

    def __init__(
        self, 
        parent, 
        infection_time, 
        detection_time, 
        recovery_time, 
        category
    ):
        """
        Category is one of:
        - 'I' Imported
        - 'A' Asymptomatic
        - 'S' Symptomatic
        - 'TA' Asymptomatic interstate traveller 
        - 'TS' Symptomatic interstate traveller 
        """
        self.parent = parent
        self.infection_time = infection_time
        self.detection_time = detection_time
        self.recovery_time = recovery_time
        self.category = category

class Forecast:
    """
    Forecast object that contains methods to simulate a forcast forward, given Reff and current state.
    """

    def __init__(
        self, 
        current, 
        state, 
        start_date,
        forecast_date, 
        cases_file_date, 
        end_time = None,
        VoC_flag='', 
        scenario='',
        n_sims=None, 
        from_jurisdiction_cases={}, 
        apply_interstate_seeding=None, 
        adjust_TP_forecast=False
    ):
        """Create forecast object with parameters in preperation for running simulation.

        Args:
            current (list of ints): A list of the infected people at the start of the simulation.
            state (str): The state to simulate.
            start_date (str): The %Y-%m-%d string of the start date of the forecast.
            forecast_date ([type], optional): Date to forecast from. Usually the same as cases_file_date.
            cases_file_date (str): Date of cases file to use. Format "2021-01-01".
            VoC_flag (str, optional): Which VoC to increase Reff to. Can be empty str.
            scenario (str, optional): Filename suffix for scenario run.  Can be empty str.
        """
        from params import local_detection, a_local_detection, qi_d, alpha_i, k
        
        # if applying interstate 
        self.print_at_iterations = False 
        self.apply_interstate_seeding = apply_interstate_seeding
        self.old_approach = True
        self.adjust_TP_forecast = adjust_TP_forecast

        self.n_sims = n_sims
        self.state = state
        self.end_time = end_time
        self.start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        
        # start date sets day 0 in script to start_date
        self.initial_state = current.copy()  # Observed cases on start day
        # Create an object list of Persons based on observed cases on start day/
        people = ['I']*current[0] + ['A']*current[1] + ['S']*current[2]
        self.initial_people = {i: Person(0, 0, 0, 0, cat) for i, cat in enumerate(people)}
        self.alpha_i = alpha_i[state]
        # Probability of *unobserved* imported infectious individuals
        self.qi = qi_d[state]
        self.symptomatic_detection_prob = local_detection[state]
        self.asymptomatic_detection_prob = a_local_detection[state]
        self.k = k
        self.relative_infectiousness_interstate_traveller = 0.5
        # self.qua_ai = 2 if state=='NSW' else 1 # Pre-march-quarantine version of alpha_i.
        self.qua_ai = 1
        self.gam = 1/2
        self.ps = 0.7  # Probability of being symptomatic
        # Increase Reff to due this VoC
        self.VoC_flag = VoC_flag
        # Add an optional scenario flag to load in specific Reff scenarios and save results. 
        # This does not change the run behaviour of the simulations.
        self.scenario = scenario
        # forecast_date and cases_file_date are usually the same.
        # Total number of days in simulation
        self.forecast_date = (pd.to_datetime(forecast_date, format='%Y-%m-%d') - self.start_date).days
        self.cases_file_date = cases_file_date
        # Load in Rff data before running all sims
        self.read_in_all_Reffs()
        # Assumption dates.
        # Date from which quarantine was started
        self.quarantine_change_date = pd.to_datetime('2020-04-15', format='%Y-%m-%d').dayofyear - self.start_date.dayofyear
        # Day from whih to reduce imported overseas cases escapes due to quarantine worker vaccination
        self.hotel_quarantine_vaccine_start = (pd.to_datetime("2021-05-01", format='%Y-%m-%d') - self.start_date).days
        # Day from which to start treating imported cases as delta cases
        self.VoC_on_imported_effect_start = (pd.to_datetime("2021-05-01", format='%Y-%m-%d') - self.start_date).days
        # This is a paameter which decreases the detection probability before the date where VIC started testing properly. 
        # Could be removed in future.
        if state == "VIC":
            self.test_campaign_date = (pd.to_datetime('2020-06-01', format='%Y-%m-%d') - self.start_date).days
            self.test_campaign_factor = 1.5
        else:
            self.test_campaign_date = None

        # If we need to apply seeding. Should only really be necessary for interstate travelling
        self.from_jurisdiction_cases = from_jurisdiction_cases

        self.num_bad_sims = 0
        self.num_too_many = 0

    def interstate_seeding(self, from_jurisdiction_cases):
        """
        Seed cases into the jurisdiction. This will run a binomial 
        """
        to_jurisdiction = self.state
            
        # get the from jurisdictions out of the keys
        from_jurisdictions = from_jurisdiction_cases.keys()
        
        # prob of travel between jurisdictions
        prob_travel = {
            "ACT-SA": 0.00066, 
            "VIC-SA": 0.00192, 
            "NSW-SA": 0.00074, 
            "QLD-SA": 0.00063, 
            "TAS-SA": 0.00053,      # placholder using WA-SA
            "WA-SA": 0.00053, 
            "NT-SA": 0.00053        # placeholder using WA-SA
        }
        
        # this adjusts the probabilities of travelling between jurisdictions based on early estimates
        # WILL NEED TO BE REVISED (Most likley upwards as interstate travel increases)
        scaling_factor = 1.0
        prob_travel = {k: p * scaling_factor for k, p in prob_travel.items()}
        # estimated probability that the import is vaccinated
        prob_vaccinated = 0.5

        types = ['asymp_inci', 'symp_inci']
        # vector to store the number imported from asymp and symp
        try:
            num_interstate_import = {type: np.zeros(shape=from_jurisdiction_cases['NSW']['symp_inci'].
                                                    shape[0]) 
                                    for type in types}
        except: 
            return None

        # loop over the import jurisdictions and sample number of interstate imports each day 
        for jur in from_jurisdictions:
            # prob of person travelling and being vaccinated
            to_from = jur + str('-') + to_jurisdiction
            p = prob_vaccinated*prob_travel[to_from]
            for type in types:
                n = from_jurisdiction_cases[jur][type]
                # calculate the number of interstate imports each day
                num_interstate_import[type] = np.random.binomial(n = n, p = p)
            
        return num_interstate_import
    
    def initialise_sim(self, curr_time=0.0):
        """
        Given some number of cases in self.initial_state (copied),
        simulate undetected cases in each category and their
        infectious times. Updates self.current for each person.
        """
        if curr_time == 0:
            self.alpha_s = 1.0/(self.ps + self.gam*(1.0-self.ps))
            self.alpha_a = self.gam * self.alpha_s
            self.current = self.initial_state.copy()
            self.people = self.initial_people.copy()
            # N samples for each of infection and detection times
            # Grab now and iterate through samples to save simulation
            self.generate_times(size=200000)
            self.get_detect_rv = self.iter_detect_rv()
            self.get_inf_time = self.iter_inf_time()
            self.get_detect_time = self.iter_detect_time()
            # counters for terminating early
            self.inf_forecast_counter = 0
            # this inits the number of detections (onset events) in each window to match size of max_cases_in_windows
            self.sim_cases_in_window = np.zeros_like(self.cases_in_windows)

            # assign infection time to those discovered
            # obs time is day = 0
            for person in self.people.keys():
                self.people[person].infection_time = -1*next(self.get_detect_time)
        else:
            # reinitialising, so actual people need times assume all symptomatic
            prob_symp_given_detect = self.symptomatic_detection_prob*self.ps/(
                self.symptomatic_detection_prob*self.ps + self.asymptomatic_detection_prob*(1-self.ps))
            num_symp = binom_sample(n=int(self.current[2]), p=prob_symp_given_detect)
            for person in range(int(self.current[2])):
                self.infected_queue.append(len(self.people))
                # inf_time = next(self.get_inf_time) #remove?
                detection_time = next(self.get_detect_time)
                if person <= num_symp:
                    new_person = Person(-1,curr_time-1*detection_time,curr_time, 0, 'S')
                else:
                    new_person = Person(-1,curr_time-1*detection_time,curr_time, 0, 'A')

                self.people[len(self.people)] = new_person

                #self.cases[max(0,ceil(new_person.infection_time)), 2] +=1

        # num undetected is nbinom (num failures given num detected)
        if self.current[2] == 0:
            num_undetected_s = neg_binom_sample(1, self.symptomatic_detection_prob)
        else:
            num_undetected_s = neg_binom_sample(self.current[2], self.symptomatic_detection_prob)

        total_s = num_undetected_s + self.current[2]

        # infer some non detected asymp at initialisation
        if total_s == 0:
            num_undetected_a = neg_binom_sample(1, self.ps)
            # num_undetected_a = nbinom.rvs(1,self.symptomatic_detection_prob)
        else:
            num_undetected_a = neg_binom_sample(total_s, self.ps)

        # simulate cases that will be detected within the next week
        if curr_time == 0:
            # Add each undetected case into people
            for n in range(num_undetected_a):
                self.people[len(self.people)] = Person(0, curr_time-1*next(self.get_inf_time), 0, 0, 'A')
                self.current[1] += 1
            for n in range(num_undetected_s):
                self.people[len(self.people)] = Person(0, curr_time-1*next(self.get_inf_time), 0, 0, 'S')
                self.current[2] += 1
        else:
            # reinitialised, so add these cases back onto cases
            # Add each undetected case into people
            for n in range(num_undetected_a):
                new_person = Person(-1, curr_time-1*next(self.get_inf_time), 0, 0, 'A')
                self.infected_queue.append(len(self.people))
                self.people[len(self.people)] = new_person
                self.cases[max(0, math.ceil(new_person.infection_time)), 1] += 1
            for n in range(num_undetected_s):
                new_person = Person(-1, curr_time-1*next(self.get_inf_time), 0, 0, 'S')
                self.infected_queue.append(len(self.people))
                self.people[len(self.people)] = new_person
                self.cases[max(0, math.ceil(new_person.infection_time)), 2] += 1
    
    def read_in_all_Reffs(self):
        """
        Read in all the forecasted TP's and then process them into local and imported Reffs 
        indexed in a dictionary of dictionaries where the first dictionary is indexed by the sim number. 
        """
        # use the helper functions to read in the file from forecast_TPs
        Reff_all = read_in_Reff_file(self.cases_file_date, self.adjust_TP_forecast)
    
        # use local dictionaries to store the TP paths for sims 
        import_Reffs = {}
        local_Reffs = {}
        for i in range(2000):
            import_Reffs[i], local_Reffs[i] = self.read_in_Reff(Reff_all, i)
            
        # store the results in self
        self.import_Reffs = import_Reffs 
        self.local_Reffs = local_Reffs

    def set_Reff(self):
        """
        Set the TP to use for a given simulation. 
        """
        self.R_I = self.import_Reffs[self.num_of_sim % 2000]
        self.Reff = self.local_Reffs[self.num_of_sim % 2000]

    def read_in_Reff(self, Reff_all, i):
        """
        Read in Reff CSV that was produced by the generate_R_L_forecasts.py script.
        """
        import pandas as pd

        df_forecast = Reff_all
        
        # Get R_I values and store in object.
        R_I = df_forecast.loc[(df_forecast.type == 'R_I') & 
                              (df_forecast.state == self.state), 
                              i % 2000].values[0]

        # Get only R_L forecasts
        df_forecast = df_forecast.loc[df_forecast.type == 'R_L']
        df_forecast = df_forecast.set_index(['state', 'date'])
        
        Reff_lookupstate = {}
        
        # initialise a temporary df that is only for state of interest and 
        # corresponds to the appropriate sim number
        df_forecast_tmp = df_forecast.loc[self.state, i % 2000]
        
        # print(df_forecast_tmp)
        # loop over the key-value pairs in the series df_forecast_tmp and 
        # readjust based on the start date when storing in Reff_lookupstate
        for (key, value) in df_forecast_tmp.items():
            # instead of mean and std, take all columns as samples of Reff
            # convert key to days since start date for easier indexing
            newkey = (key - self.start_date).days
            Reff_lookupstate[newkey] = value
            
        return R_I, Reff_lookupstate
    
    # @profile
    def generate_new_cases(self, parent_key, Reff, k, travel=False):
        """
        Generate offspring for each parent, check if they travel. 
        The parent_key parameter lets us find the parent from the array self.people 
        containing the objects from the branching process.
        """

        # Check parent category
        if self.people[parent_key].category == 'S':  # Symptomatic
            num_offspring = neg_binom_sample(k, 1.0 - self.alpha_s*Reff/(self.alpha_s*Reff + k))
            
        elif self.people[parent_key].category == 'A':  # Asymptomatic
            num_offspring = neg_binom_sample(k, 1.0 - self.alpha_a*Reff/(self.alpha_a*Reff + k))
            
        elif self.people[parent_key].category == 'TS':  # Symptomatic interstate traveller
            # half the infectiousness of an interstate traveller
            Reff_tmp = Reff*self.relative_infectiousness_interstate_traveller
            num_offspring = neg_binom_sample(k, 1.0 - self.alpha_s*Reff_tmp/(self.alpha_s*Reff_tmp + k))
            
        elif self.people[parent_key].category == 'TA':  # Asymptomatic interstate traveller
            # half the infectiousness of an interstate traveller
            Reff_tmp = Reff*self.relative_infectiousness_interstate_traveller
            num_offspring = neg_binom_sample(k, 1.0 - self.alpha_a*Reff_tmp/(self.alpha_a*Reff_tmp + k))
            
        else:  # International import
            Reff = self.R_I
            # Apply vaccine reduction for hotel quarantine workers
            if self.people[parent_key].infection_time >= self.hotel_quarantine_vaccine_start:
                # p_{v,h} is the proportion of hotel quarantine workers vaccinated
                p_vh = 0.9+np.random.beta(2, 4)*9/100
                # v_{e,h} is the overall vaccine effectiveness
                v_eh = 0.83+np.random.beta(2, 2)*14/100
                Reff *= (1-p_vh*v_eh)

            # Apply increase escape rate due to Delta variant.
            if self.people[parent_key].infection_time >= self.VoC_on_imported_effect_start:
                Reff = Reff*1.39*1.3

            if self.people[parent_key].infection_time < self.quarantine_change_date:
                # factor of 3 times infectiousness prequarantine changes
                num_offspring = neg_binom_sample(k, 1.0 - self.qua_ai*Reff/(self.qua_ai*Reff + k))
            else:
                num_offspring = neg_binom_sample(k, 1.0 - self.alpha_i*Reff/(self.alpha_i*Reff + k))

        if num_offspring > 0:
            # generate number of symptomatic cases
            num_sympcases = binom_sample(n=num_offspring, p=self.ps)
            
            # if self.people[parent_key].category == 'A':
            #     child_times = []
                
            for new_case in range(num_offspring):
                # define each offspring
                inf_time = self.people[parent_key].infection_time + next(self.get_inf_time)
                if inf_time > self.forecast_date:
                    self.inf_forecast_counter += 1

                # normal case within state
             #   if self.people[parent_key].category == 'A':
             #       child_times.append(ceil(inf_time))
             
                if math.ceil(inf_time) <= self.cases.shape[0]:
                    # within forecast time
                    detection_rv = next(self.get_detect_rv)
                    detect_time = inf_time + next(self.get_detect_time)
                    # recovery_time = 0  # for now not tracking recoveries

                    if new_case <= num_sympcases-1:  # minus 1 as new_case ranges from 0 to num_offspring-1
                        # first num_sympcases are symnptomatic, rest are asymptomatic
                        category = 'S'
                        self.cases[max(0, math.ceil(inf_time)-1), 2] += 1

                        # if self.test_campaign_date is not None:
                        #     # see if case is during a testing campaign
                        #     if inf_time < self.test_campaign_date:
                        #         detect_prob = self.symptomatic_detection_prob
                        #     else:
                        #         detect_prob = min(0.95, self.symptomatic_detection_prob*self.test_campaign_factor)
                        # else:
                        #     detect_prob = self.symptomatic_detection_prob
                        
                        detect_prob = self.symptomatic_detection_prob
                        if detection_rv < detect_prob:
                            # case detected
                            # only care about detected cases
                            self.increment_counters(detect_time, category)

                    else:
                        category = 'A'
                        self.cases[max(0, math.ceil(inf_time)-1), 1] += 1
                        #detect_time = 0
                        # if self.test_campaign_date is not None:
                        #     # see if case is during a testing campaign
                        #     if inf_time < self.test_campaign_date:
                        #         detect_prob = self.asymptomatic_detection_prob
                        #     else:
                        #        detect_prob = min(0.95, self.asymptomatic_detection_prob*self.test_campaign_factor)
                        # else:
                        detect_prob = self.asymptomatic_detection_prob
                            
                        if detection_rv < detect_prob:
                            # case detected
                            #detect_time = inf_time + next(self.get_detect_time)
                            self.increment_counters(detect_time, category)

                    # add new infected to queue
                    self.infected_queue.append(len(self.people))

                    # add person to tracked people
                    self.people[len(self.people)] = Person(parent_key, inf_time, detect_time, 0, category)
                    
                else:
                    # new infection exceeds the simulation time, not recorded
                    self.cases_after = self.cases_after + 1
        
    def simulate(self, end_time, sim, seed):
        """
        Simulate the branching process until end_time.
        """
        # set seed in the generator, this will get passed to the class methods and is much more efficient 
        # compared to the scipy method. Yes it was checked that numpy and scipy produce the same RVs.
        np.random.seed(seed)
        self.num_of_sim = sim
        
        # self.read_in_Reff()
        self.set_Reff()
        # generate storage for cases
        self.cases = np.zeros(shape=(self.end_time, 3), dtype=float)
        self.observed_cases = np.zeros_like(self.cases)
        self.observed_cases[0, :] = self.initial_state.copy()
        
        # Initalise undetected cases and add them to current
        self.initialise_sim()
        # number of cases after end time
        self.cases_after = 0  # gets incremented in generate new cases
        # Record day 0 cases
        self.cases[0, :] = self.current.copy()
        
        # Generate imported cases
        new_imports = []
        unobs_imports = []
        for day in range(self.end_time):
            # Values for a and b are initialised in import_cases_model() which is called by read_in_cases() during setup.
            a = self.a_dict[day]
            b = self.b_dict[day]
            # Dij = number of observed imported infectious individuals
            Dij = neg_binom_sample(a, 1-1/(b+1))
            # Uij = number of *unobserved* imported infectious individuals
            unobserved_a = 1 if Dij == 0 else Dij
            Uij = neg_binom_sample(unobserved_a, self.qi)
            unobs_imports.append(Uij)
            new_imports.append(Dij + Uij)

        for day, imports in enumerate(new_imports):
            self.cases[day, 0] = imports
            for n in range(imports):
                # Generate people
                if n - unobs_imports[day] >= 0:
                    # number of observed people
                    new_person = Person(0, day, day + next(self.get_detect_time), 0, 'I')
                    self.people[len(self.people)] = new_person
                    if new_person.detection_time <= self.end_time:
                        self.observed_cases[max(0, math.ceil(new_person.detection_time)-1), 0] += 1
                else:
                    # unobserved people
                    new_person = Person(0, day, 0, 0, 'I')
                    self.people[len(self.people)] = new_person
                    if day <= self.end_time:
                        self.cases[max(0, day-1), 0] += 1

        # now add in interstate imports
        if self.apply_interstate_seeding:
            try:
                # simulate travel volumes
                num_interstate_import = self.interstate_seeding(
                    from_jurisdiction_cases=self.from_jurisdiction_cases)
                
                # some scaffolding for times till infectious in community
                # sample times for the interstate imports to be infected and then get tested
                interstate_import_symp_inf_time = np.random.gamma(
                    3, 0.5, size=num_interstate_import['symp_inci'].sum())
                interstate_import_asymp_inf_time = np.random.gamma(
                    3, 0.5, size=num_interstate_import['asymp_inci'].sum())
                # take the ceiling as the days need to be integers
                interstate_import_symp_inf_time = np.ceil(interstate_import_symp_inf_time)
                interstate_import_asymp_inf_time = np.ceil(interstate_import_asymp_inf_time)
                
                # loop over the symptomatic travellers
                for day, cases in enumerate(num_interstate_import['symp_inci']):
                    # note: if no cases on a given day, this will not run the loop
                    for n in range(cases):
                        # new person is symptomatic and assume their infection and day of onset are 
                        # the same
                        # days_till_test = day+interstate_import_symp_inf_time[n]
                        days_till_test = day+1
                        new_person = Person(0, days_till_test, days_till_test, 0, 'TS')
                        self.people[len(self.people)] = new_person
                # loop over the asymptomatic travellers
                for day, cases in enumerate(num_interstate_import['asymp_inci']):
                    for n in range(cases):
                        # new person is asymptomatic and we assume their infection and day of onset
                        # are the same
                        # days_till_test = day+interstate_import_asymp_inf_time[n]
                        days_till_test = day+1
                        new_person = Person(0, days_till_test, days_till_test, 0, 'TA')
                        self.people[len(self.people)] = new_person
            except: 
                None

        # Create queue for infected people
        self.infected_queue = deque()
        # Assign people to infected queue
        for key, person in self.people.items():
            # add to the queue
            self.infected_queue.append(key)
            # Record their times
            if person.infection_time > self.end_time:
                # initial undetected cases have slim chance to be infected
                # after end_time
                if person.category != 'I':
                    # imports shouldn't count for extinction counts
                    self.cases_after += 1
                    # print("cases after at initialisation")
        
        
        # Record initial inferred obs including importations.
        self.inferred_initial_obs = self.observed_cases[0, :].copy()
        # print(self.inferred_initial_obs, self.current)

        # General simulation through time by proceeding through queue
        # of infecteds
        n_resim = 0
        self.bad_sim = False
        reinitialising_window = 3
        self.daycount = 0
        while len(self.infected_queue) > 0:
            day_end = self.people[self.infected_queue[0]].detection_time
            over_check = self.check_over_cases()
            if (day_end < self.forecast_date) and over_check: 
                self.num_too_many += 1
                self.bad_sim = True
                break
            else:
                # check max cases for after forecast date
                if self.inf_forecast_counter > self.max_cases:
                    # hold value forever
                    if day_end < self.cases.shape[0]-1:
                        self.cases[math.ceil(day_end):,2] = self.cases[math.ceil(day_end)-2, 2]
                        self.observed_cases[math.ceil(day_end):, 2] = self.observed_cases[math.ceil(day_end)-2, 2]
                    else:
                        self.cases_after += 1
                        
                    self.num_too_many += 1
                    break

            # stop if parent infection time greater than end time
            if self.people[self.infected_queue[0]].infection_time > self.end_time:
                self.infected_queue.popleft()
                # print("queue had someone exceed end_time!!")
            else:

                # take approproate Reff based on parent's infection time
                curr_time = self.people[self.infected_queue[0]].infection_time
                while True:
                    # sometimes initial cases infection time is pre
                    # Reff data, so take the earliest one
                    try:
                        Reff = self.Reff[math.ceil(curr_time)-1]
                    except KeyError:
                        if curr_time > 0:
                            print("Unable to find Reff for this parent at time: %.2f" % curr_time)
                            raise KeyError
                        curr_time += 1
                        continue
                    break
                # generate new cases with times
                parent_key = self.infected_queue.popleft()
                # recorded within generate new cases
                self.generate_new_cases(parent_key=parent_key, Reff=Reff, k=self.k)
                
        # self.people.clear()
        if self.bad_sim == False:
            # Check simulation for discrepancies
            for day in range(7, self.end_time):
                # each day runs through self.infected_queue
                missed_outbreak = self.data_check(day)  # True or False
                if missed_outbreak:
                    # print("missing an outbreak")
                    self.daycount += 1
                    if self.daycount >= reinitialising_window:

                        if self.old_approach:

                            n_resim += 1
                            # print("Local outbreak in "+self.state+" not simulated on day %i" % day)
                            # cases to add
                            # treat current like empty list
                            self.current[2] = max(0, self.actual[day] - sum(self.observed_cases[day, 1:]))
                            self.current[2] += max(0, self.actual[day-1] - sum(self.observed_cases[day-1, 1:]))
                            self.current[2] += max(0, self.actual[day-2] - sum(self.observed_cases[day-2, 1:]))

                            # how many cases are symp to asymp
                            prob_symp_given_detect = self.symptomatic_detection_prob*self.ps/(
                                self.symptomatic_detection_prob*self.ps +
                                self.asymptomatic_detection_prob*(1-self.ps)
                            )
                            num_symp = binom_sample(n=int(self.current[2]), p=prob_symp_given_detect)
                            # distribute observed cases over 3 days Triangularly
                            self.observed_cases[max(0, day), 2] += num_symp//2
                            self.cases[max(0, day), 2] += num_symp//2

                            self.observed_cases[max(0, day-1), 2] += num_symp//3
                            self.cases[max(0, day-1), 2] += num_symp//3

                            self.observed_cases[max(0, day-2), 2] += num_symp//6
                            self.cases[max(0, day-2), 2] += num_symp//6

                            # add asymptomatic
                            num_asymp = self.current[2] - num_symp
                            self.observed_cases[max(0, day), 2] += num_asymp//2
                            self.cases[max(0, day), 2] += num_asymp//2

                            self.observed_cases[max(0, day-1), 2] += num_asymp//3
                            self.cases[max(0, day-1), 2] += num_asymp//3

                            self.observed_cases[max(0, day-2), 2] += num_asymp//6
                            self.cases[max(0, day-2), 2] += num_asymp//6

                            self.initialise_sim(curr_time=day)
                            # print("Reinitialising with %i new cases "  % self.current[2] )

                            # reset days to zero
                            self.daycount = 0

                        else:
                    
                            n_resim += 1
                            # print("Local outbreak in "+self.state+" not simulated on day %i" % day)
                            # cases to add
                            # treat current like empty list
                            # seed only a single case into the system and see if they are symptomatic or not
                            self.current[2] = 1

                            # how many cases are symp to asymp
                            prob_symp_given_detect = self.symptomatic_detection_prob*self.ps/(
                                self.symptomatic_detection_prob*self.ps +
                                self.asymptomatic_detection_prob*(1-self.ps)
                            )
                            num_symp = binom_sample(n=int(self.current[2]), p=prob_symp_given_detect)
                            # distribute observed cases over 3 days Triangularly
                            self.observed_cases[max(0, day), 2] += num_symp
                            self.cases[max(0, day), 2] += num_symp

                            # add asymptomatic
                            num_asymp = self.current[2] - num_symp
                            self.observed_cases[max(0, day), 2] += num_asymp
                            self.cases[max(0, day), 2] += num_asymp

                            self.initialise_sim(curr_time=day)
                            # print("Reinitialising with %i new cases "  % self.current[2] )

                            # reset days to zero
                            self.daycount = 0

                if n_resim > 50:
                    # print("This sim reinitilaised %i times" % n_resim)
                    self.bad_sim = True
                    n_resim = 0
                    break
                
                # Each check of day needs to simulate the cases before moving
                # to next check, otherwise will be doubling up on undetecteds
                while len(self.infected_queue) > 0:
                    day_end = self.people[self.infected_queue[0]].detection_time
                    over_check = self.check_over_cases()
                    if (day_end < self.forecast_date) and over_check: 
                    # check for exceeding max_cases
                        self.num_too_many += 1
                        self.bad_sim = True
                        break
                    else:
                        if self.inf_forecast_counter > self.max_cases:
                            day_inf = self.people[self.infected_queue[0]].infection_time
                            self.cases[math.ceil(day_inf):,2] = self.cases[math.ceil(day_inf)-2, 2]

                            self.observed_cases[math.ceil(day_inf):, 2] = self.observed_cases[math.ceil(day_inf)-2, 2]
                            self.num_too_many += 1
                            
                            break
                        
                    # stop if parent infection time greater than end time
                    if self.people[self.infected_queue[0]].infection_time > self.end_time:
                        personkey = self.infected_queue.popleft()
                        # print("queue had someone exceed end_time!!")
                    else:
                        # take approproate Reff based on parent's infection time
                        curr_time = self.people[self.infected_queue[0]].infection_time
                        while True:
                            # sometimes initial cases infection time is pre
                            # Reff data, so take the earliest one
                            try:
                                Reff = self.Reff[math.ceil(curr_time)-1]
                            except KeyError:
                                if curr_time > 0:
                                    # print("Unable to find Reff for this parent at time: %.2f" % curr_time)
                                    raise KeyError
                                curr_time += 1
                                continue
                            break
                        # generate new cases with times
                        parent_key = self.infected_queue.popleft()
                        self.generate_new_cases(parent_key=parent_key, Reff=Reff, k=self.k)
                        #missed_outbreak = max(1,missed_outbreak*0.9)
                else:
                    # this is using a while-else loop and we only pass into here if 
                    # the loop completes. Following this we c
                    continue
                    
                # only reach here if while loop breaks, so break the data check
                break
        
        # need to empty people dict as we never remove anything from the dictionary 
        # during the simulation — possible improvement? 
        self.people.clear()
        gc.collect()
        
        # now we run a check to cutoff any sims that were too few cases — probably want to implement this better but this 
        # works for now 
        if not self.bad_sim:
            self.final_check()
        
        if self.bad_sim:
            # return NaN arrays for all bad_sims
            if self.print_at_iterations:
                print("Bad sim...")
                print(np.sum(self.observed_cases, axis=0))
            self.cumulative_cases = np.empty_like(self.cases)
            self.cumulative_cases[:] = np.nan
            return (self.cumulative_cases, self.cumulative_cases,
                    {'num_of_sim': self.num_of_sim,
                     'bad_sim': self.bad_sim}) 
        else:
            # good sim
            if self.print_at_iterations:
                print("Good sim!")
                print(np.sum(self.observed_cases, axis=0))
            return (self.cases.copy(), self.observed_cases.copy(),
                    {'num_of_sim': self.num_of_sim,
                     'bad_sim': self.bad_sim})

    def check_over_cases(self):
        """
        Performs a check to see whether we have exceeded the maximum number of allowable 
        cases in a given period.
        """
        
        # loop over windows and check for whether we have exceeded the cases in any window 
        # don't check the last window corresponding to nowcast
        if (self.sim_cases_in_window > self.max_cases_in_windows).any(): 
            if self.print_at_iterations:
                print("too many in window:", np.where(self.sim_cases_in_window > self.max_cases_in_windows))
                print("cases: ", self.sim_cases_in_window[np.where(self.sim_cases_in_window > self.max_cases_in_windows)], 
                        "max cases: ", self.max_cases_in_windows[np.where(self.sim_cases_in_window > self.max_cases_in_windows)])
            return True 
        else: 
            return False
        
    def final_check(self): 
        """
        Performs a final check for undershoot in the simulation compared to the data.
        """
        
        # loop over the windows and check to see whether we are below the windows
        if (self.sim_cases_in_window < self.min_cases_in_windows).any():
            if self.print_at_iterations:
                print("too few in window: ", np.where(self.sim_cases_in_window < self.min_cases_in_windows))
            self.bad_sim = True 
        
    def increment_counters(self, detect_time, category):
        """
        Given a detect_time and category, we increment the counters in a window and add the 
        person to the observed cases (iff they have a detection time within the observation window)
        """

        # check to see if case in forecast window
        if detect_time < self.cases.shape[0]:
            # check to see if case before the forecast date 
            if detect_time < self.forecast_date:
                for (i, x) in enumerate(self.window_sizes):
                    # window sizes is cumulative so find which one we're in and increment
                    if detect_time < x:
                        self.sim_cases_in_window[i] += 1
                        break
                    
            # add case to observed
            if (category == "S") | (category == 'TS'):
                self.observed_cases[max(0, math.ceil(detect_time)-1), 2] += 1
            elif (category == "A") | (category == 'TA'):
                self.observed_cases[max(0, math.ceil(detect_time)-1), 1] += 1

    def to_df(self, results):
        """
        Put results from the simulation into a pandas dataframe and record as h5 format. This is called externally by the run_state.py script.
        """
        import pandas as pd

        df_results = pd.DataFrame()
        n_sims = results['symp_inci'].shape[1]
        days = results['symp_inci'].shape[0]

        sim_vars = ['bad_sim']
        for key, item in results.items():
            if key not in sim_vars:
                df_results = df_results.append(
                    pd.DataFrame(item.T, index=pd.MultiIndex.from_product([[key], range(n_sims)], names=['Category', 'sim'])))
        df_results.columns = pd.date_range(start=self.start_date, periods=days)
        df_results.columns = [col.strftime('%Y-%m-%d') for col in df_results.columns]
        # Record simulation variables
        for var in sim_vars:
            df_results[var] = [results[var][sim] for cat, sim in df_results.index]

        print('VoC_flag is', self.VoC_flag)
        print("Saving results for state "+self.state)
        # save separate versions depending on whether the model is being used for seeding of not
        if not self.apply_interstate_seeding:
            df_results.to_parquet("./results/"+self.state+self.start_date.strftime(format='%Y-%m-%d')+
                                  "sim_R_L"+str(n_sims)+"_seed_"+"days_"+str(days)+".parquet")
        else:
            df_results.to_parquet("./results/"+self.state+self.start_date.strftime(format='%Y-%m-%d')+
                                  "sim_R_L"+str(n_sims)+"days_"+str(days)+".parquet")

        return df_results

    def data_check(self, day):
        """
        A metric to calculate how far the simulation is from the actual data. 
        Need to refactor this to ensure 
        """
        try:
            # calculate the actual number of 3 day cases over day-2 to day
            actual_3_day_total = 0
            for i in range(3):
                actual_3_day_total += self.actual[max(0, day-i)]
                
            # calculate the number of cases in the simulation 
            threshold = case_insertion_threshold*max(
                1, sum(
                    self.observed_cases[max(0, day-2):day+1, 1] + 
                    self.observed_cases[max(0, day-2):day+1, 2]
                )
            )
            
            if actual_3_day_total > threshold:
                if self.print_at_iterations:
                    print("actual cases: ", actual_3_day_total)
                    print("threshold: ", threshold)
                return min(3, actual_3_day_total/threshold)
            else:
                # long absence, then a case, reintroduce
                week_in_sim = sum(
                    self.observed_cases[max(0, day-7):day+1, 1] + 
                    self.observed_cases[max(0, day-7):day+1, 2]  
                )
                if week_in_sim == 0 and actual_3_day_total > 0:
                    if self.print_at_iterations:
                        print("week in sim: ", week_in_sim)
                    return actual_3_day_total
                else: 
                    # no outbreak missed
                    return False

        except KeyError:
            #print("No cases on day %i" % day)
            return False

    def read_in_cases(self):
        """
        Read in NNDSS case data to measure incidence against simulation. Nothing is returned as results are saved in object.
        This also calculates the lower and upper case limits in each of the observation windows.
        """
        import pandas as pd
        from datetime import timedelta
        import glob

        df = read_in_NNDSS(self.cases_file_date, apply_delay_at_read=True)  # Call helper_function

        self.import_cases_model(df)
		
        df = df.loc[df.STATE == self.state]

        if self.state == 'VIC':
            # data quality issue
            df.loc[df.date_inferred <= '2019-01-01', 'date_inferred'] = df.loc[df.date_inferred <= '2019-01-01', 'date_inferred'] + pd.offsets.DateOffset(year=2020)
            df.loc[df.date_inferred == '2002-07-03','date_inferred'] = pd.to_datetime('2020-07-03')
            df.loc[df.date_inferred == '2002-07-17','date_inferred'] = pd.to_datetime('2020-07-17')
            
        df = df.groupby(['date_inferred'])[['imported', 'local']].sum()
        
        df.reset_index(inplace=True)
        # make date integer from start of year
        timedelta_from_start = df.date_inferred - self.start_date
        df['date'] = timedelta_from_start.apply(lambda x: x.days)
        df = df.sort_values(by='date')
        df = df.set_index('date')
        # needed to be more careful with the reindexing as using the date inferred value
        df = df.reindex(range(self.end_time), columns={'imported', 'local'}, fill_value=0)

        # first we calculate the length of the sliding windows and the number of cases in each window
        self.calculate_counts_in_windows(df)
        # using the calcualted case counts, we determine maximum limits in each window
        self.calculate_limits(df)
        forecast_days = self.end_time-self.forecast_date
        
        print("Local cases in last 14 days is %i" % sum(df.local.values[-1*(14+forecast_days):]))
        self.actual = df.local.to_dict()
        
    def calculate_counts_in_windows(self, df):
        """
        Determines the counts in each of the windows over the forecast horizon. 
        This also saves the window sizes for incrementing counters in the code as
        well. 
        """
        
        # number of days to unrestrict the simulation
        nowcast_days = 14
        # length of comparison windows 
        window_length = 20
        # number of days we are forecasting for
        forecast_days = self.end_time-self.forecast_date
        # get the index of the last date in the data
        cases_in_window = np.array([])
        # sum the nowcast cases
        cases_in_window = np.append(cases_in_window, sum(df.local.values[-1*(nowcast_days+forecast_days):]))
        # get the number of days the simulation is run for (with data) where we subtract nowcast_days 
        # as this is the nowcast 
        n_days = self.forecast_date - nowcast_days
        # number of windows is integer division of n_days by 30
        n_windows = n_days // window_length
        # get the number of days in the first window 
        n_days_first_window = n_days - window_length*n_windows
        # the last window is for the nowcast and consists of the last two weeks before the forecast date
        window_sizes = np.array([nowcast_days])
        
        for n in range(n_windows):
            # we initially end at the beginning of the nowcast (so -(nowcast_days+forecast_days)) and 
            # add on a month per window. 
            start_index = -(nowcast_days+forecast_days+window_length*(n+1))
            # we initially end at the beginning of the nowcast (so -(nowcast_days+forecast_days)) and 
            # add on a month-1 per window. 
            end_index = -(nowcast_days+forecast_days+window_length*n)-1
            cases_in_window = np.append(cases_in_window, 
                                        sum(df.local.values[start_index:end_index]))
            window_sizes = np.append(window_sizes, window_length)
        
        if n_days_first_window != 0:
            # the last index is the number of windows+1
            end_index = -(nowcast_days+forecast_days+window_length*(n_windows+1))-1
            # now add in the cases in the last window 
            cases_in_window[-1] += sum(df.local.values[:end_index])
            window_sizes[-1] += n_days_first_window

        # we take the cumulative sum of the window sizes 
        self.window_sizes = np.cumsum(np.flip(window_sizes))
        # set these in the overall forecast object
        self.cases_in_windows = np.flip(cases_in_window)
        self.n_windows = self.window_sizes.shape[0]
        
        print("Observation windows cumulative lengths: ", self.window_sizes)
        print("Number of cases in each window: ", self.cases_in_windows)
        
    def calculate_limits(self, df):
        """
        Calculates upper on lower bounds for each of the windows during the simulation 
        period considered. Saves the results in self.min_cases_in_windows and self.max_cases_in_windows.
        """
    
        self.min_cases_in_windows = np.zeros_like(self.cases_in_windows)
        self.max_cases_in_windows = np.zeros_like(self.cases_in_windows)
        # max cases factors
        limit_factor_backcasts = 2.5
        limit_factor_nowcast = 1.5
        # backcasts all have same limit
        self.max_cases_in_windows[:-1] = np.maximum(100, np.ceil(limit_factor_backcasts * self.cases_in_windows[:-1]))
        self.max_cases_in_windows[-1] = np.maximum(100, np.ceil(limit_factor_nowcast * self.cases_in_windows[-1]))
        
        # now we calculate the lower limit, this is used to exclude forecasts following simulation 
        low_limit_backcast = 0.3
        low_limit_nowcast = 0.5
        
        # alter the minimum number of cases in the window based on those exceeding a threshold of 10 cases
        for i in range(len(self.min_cases_in_windows)-1):
            # this is the same approach used for the max_cases_in_windows but ensures the minimum number of cases is 0 
            tmp = np.maximum(0, np.floor(low_limit_backcast*self.cases_in_windows[i]))
            self.min_cases_in_windows[i] = tmp if tmp >= 10 else 0 
        
        # self.min_cases_in_windows[:-1] = np.maximum(0, np.floor(low_limit_backcast*self.cases_in_windows[:-1]))
        self.min_cases_in_windows[-1] = np.maximum(0, np.floor(low_limit_nowcast*self.cases_in_windows[-1]))
        self.max_cases = max(500000, sum(df.local.values) + sum(df.imported.values))

    def import_cases_model(self, df):
        """
        This function takes the NNDSS/linelist data and creates a set of parameters to generate imported (overseas acquired) cases over time.

        Resulting parameter dict is saved in self.a_dict and self.a_dict rather than being returned.
        """
        from datetime import timedelta

        def get_date_index(date):
            # subtract 4 from date to infer period of entry when infected
            date = date-timedelta(days=4)
            n_days_into_sim = (date - self.start_date).days
            return n_days_into_sim
        
        prior_alpha = 0.5  # Changed from 1 to lower prior (26/03/2021)
        prior_beta = 1/5

        df['date_index'] = df.date_inferred.apply(get_date_index)
        df_state = df[df['STATE'] == self.state]
        
        counts_by_date = df_state.groupby('date_index').imported.sum()

        # Replace our value for $a$ with an exponential moving average
        moving_average_a = {}
        smoothing_factor = 0.1
        # exponential moving average start
        current_ema = counts_by_date.get(-11, default=0)
        
        # Loop through each day up to forecast - 4 (as recent imports are not discovered yet)
        for j in range(-10, self.forecast_date-4):
            count_on_day = counts_by_date.get(j, default=0)
            current_ema = smoothing_factor*count_on_day + (1-smoothing_factor)*current_ema
            moving_average_a[j] = prior_alpha+current_ema

        # Set the imports moving forward to match last window
        for j in range(self.forecast_date-4, self.end_time):
            moving_average_a[j] = prior_alpha+current_ema

        self.a_dict = moving_average_a

        # Set all betas to prior plus effective period size of 1
        self.b_dict = {i: prior_beta+1 for i in range(self.end_time)}

    def generate_times(self, size=50000):
        """
        Helper function. Generate large amount of gamma draws to save on simulation time later
        """
        self.detect_rv = np.random.random(size=size)  # shape and scale
        self.inf_times = offset_gen + np.random.gamma(shape_gen, scale_gen, size=size)  # shape and scale
        self.detect_times = np.random.gamma(shape_inc, scale_inc, size=size)

    def iter_detect_rv(self):
        """
        Helper function. Access Next detection_rv.
        """
        for rv in cycle(self.detect_rv):
            yield rv

    def iter_inf_time(self):
        """
        Helper function. Access Next inf_time.
        """
        for time in cycle(self.inf_times):
            yield time

    def iter_detect_time(self):
        """
        Helper function. Access Next detect_time.
        """
        for time in cycle(self.detect_times):
            yield time
        
def binom_sample(n, p):
    """
    Helper function. Wrapper for binomial sampler.
    """
    return np.random.binomial(n, p)
    # return binom.rvs(n, p)
    
def neg_binom_sample(k, p):
    """
    Helper function. Wrapper for negative binomial sampler. 
    """
    return np.random.negative_binomial(k, p)
    # return nbinom.rvs(k, p)
    
def init_from_jurisdiction_arrays(apply_intestate_seeding, start_date, n_sims, end_time):
    """
    Read in jurisdiction local cases into this instance of the model in order to apply 
    interstate imports. 
    """
    
    # these are the jurisdictions that interstate travel is allowed from
    from_jurisdictions = ['NSW', 'VIC']
    from_jurisdiction_cases = {}
    # first 9 element of start date are the parts of interest
    end_of_file_name = (str(start_date)[:10] + 'sim_R_L' + str(n_sims) + 
                        '_seed_' + 'days_' + str(end_time) + '.parquet') 
    
    for jur in from_jurisdictions: 
        # add try-except to see if we're able to read the file in or not
        try:
            tmp_cases = pd.read_parquet('results/' + jur + end_of_file_name)
        except: 
            break
        # tmp_cases = pd.read_parquet('results/' + jur + '2021-06-10sim_R_L100000days_193.parquet')
        # get the total cases 
        tmp_cases_asymp = tmp_cases.loc['asymp_inci']
        tmp_cases_symp = tmp_cases.loc['symp_inci']
        # get the good sims and get the median
        tmp_cases_asymp_median = tmp_cases_asymp.loc[tmp_cases_asymp.bad_sim == 0].median(axis=0)
        tmp_cases_asymp_median = tmp_cases_asymp_median[:-1].to_numpy(dtype=int)
        tmp_cases_symp_median = tmp_cases_symp.loc[tmp_cases_symp.bad_sim == 0].median(axis=0)
        tmp_cases_symp_median = tmp_cases_symp_median[:-1].to_numpy(dtype=int)
        # set the cases to 0 before the forecast date - this is a hacky fix for now and we likely
        tmp_cases_asymp_median[:-30] = 0
        tmp_cases_symp_median[:-30] = 0
        # store them 
        from_jurisdiction_cases[jur] = {'symp_inci': tmp_cases_symp_median, 
                                        'asymp_inci': tmp_cases_asymp_median}
        
    return from_jurisdiction_cases