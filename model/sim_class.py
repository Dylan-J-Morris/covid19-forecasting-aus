import numpy as np
import pandas as pd
from scipy.stats import nbinom, erlang, beta, binom, gamma, poisson
from math import floor
import matplotlib.pyplot as plt
import os
from helper_functions import read_in_NNDSS, read_in_Reff_file
from params import case_insertion_threshold

from collections import deque
from math import ceil
import gc
from numpy.random import random
from itertools import cycle

from timeit import default_timer as timer

class Person:
    """
    Individuals in the forecast
    """

    def __init__(self, parent, infection_time, detection_time, recovery_time, category: str):
        """
        Category is one of 'I','A','S' for Imported, Asymptomatic and Symptomatic
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

    def __init__(self, current, state, start_date,
                 forecast_date, cases_file_date,
                 VoC_flag='', scenario='', end_time = None
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
        import numpy as np
        from params import local_detection, a_local_detection, qi_d, alpha_i, k

        self.initial_state = current.copy()  # Observed cases on start day
        # Create an object list of Persons based on observed cases on start day/
        people = ['I']*current[0] + ['A']*current[1] + ['S']*current[2]
        self.initial_people = {i: Person(0, 0, 0, 0, cat) for i, cat in enumerate(people)}

        self.state = state
        # start date sets day 0 in script to start_date
        self.start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        self.alpha_i = alpha_i[state]
        # Probability of *unobserved* imported infectious individuals
        self.qi = qi_d[state]
        self.symptomatic_detection_prob = local_detection[state]
        self.asymptomatic_detection_prob = a_local_detection[state]
        self.k = k  # Hard coded
        # self.qua_ai = 2 if state=='NSW' else 1 # Pre-march-quarantine version of alpha_i.
        self.qua_ai = 1
        self.gam = 1/2
        self.ps = 0.7  # Probability of being symptomatic
        # Increase Reff to due this VoC
        self.VoC_flag = VoC_flag
        # Add an optional scenario flag to load in specific Reff scenarios and save results. This does not change the run behaviour of the simulations.
        self.scenario = scenario
        # forecast_date and cases_file_date are usually the same.
        # Total number of days in simulation
        self.forecast_date = (pd.to_datetime(forecast_date, format='%Y-%m-%d') - self.start_date).days
        self.cases_file_date = cases_file_date
        # Load in Rff data before running all sims
        self.Reff_all = read_in_Reff_file(self.cases_file_date,  self.VoC_flag, scenario=self.scenario)
        # Assumption dates.
        # Date from which quarantine was started
        self.quarantine_change_date = pd.to_datetime('2020-04-15', format='%Y-%m-%d').dayofyear - self.start_date.dayofyear
        # Day from whih to reduce imported overseas cases escapes due to quarantine worker vaccination
        self.hotel_quarantine_vaccine_start = (pd.to_datetime("2021-05-01", format='%Y-%m-%d') - self.start_date).days
        # Day from which to start treating imported cases as delta cases
        self.VoC_on_imported_effect_start = (pd.to_datetime("2021-05-01", format='%Y-%m-%d') - self.start_date).days
        # This is a paameter which decreases the detection probability before the date where VIC started testing properly. Could be removed in future.
        if state == "VIC":
            self.test_campaign_date = (pd.to_datetime('2020-06-01', format='%Y-%m-%d') - self.start_date).days
            self.test_campaign_factor = 1.5
        else:
            self.test_campaign_date = None
            
        self.end_time = end_time
        self.num_too_many = 0
        self.num_bad_sims = 0

        assert len(people) == sum(
            current), "Number of people entered does not equal sum of counts in current status"

    def initialise_sim(self, curr_time=0):
        """
        Given some number of cases in self.initial_state (copied),
        simulate undetected cases in each category and their
        infectious times. Updates self.current for each person.
        """
        if curr_time == 0:
            self.alpha_s = 1/(self.ps + self.gam*(1-self.ps))
            self.alpha_a = self.gam * self.alpha_s
            self.current = self.initial_state.copy()
            self.people = self.initial_people.copy()

            # N samples for each of infection and detection times
            # Grab now and iterate through samples to save simulation
            self.generate_times(size=50000)
            self.get_inf_time = self.iter_inf_time()
            self.get_detect_time = self.iter_detect_time()

            # counters for terminating early
            self.inf_backcast_counter = 0
            self.inf_nowcast_counter = 0
            self.inf_forecast_counter = 0

            # assign infection time to those discovered
            # obs time is day =0
            for person in self.people.keys():
                self.people[person].infection_time = -1*next(self.get_detect_time)
        else:
            # reinitialising, so actual people need times
            # assume all symptomatic
            prob_symp_given_detect = self.symptomatic_detection_prob*self.ps/(
                self.symptomatic_detection_prob*self.ps +
                self.asymptomatic_detection_prob*(1-self.ps)
            )
            num_symp = binom.rvs(
                n=int(self.current[2]), p=prob_symp_given_detect)
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
            num_undetected_s = nbinom.rvs(1, self.symptomatic_detection_prob)
        else:
            num_undetected_s = nbinom.rvs(
                self.current[2], self.symptomatic_detection_prob)

        total_s = num_undetected_s + self.current[2]

        # infer some non detected asymp at initialisation
        if total_s == 0:
            num_undetected_a = nbinom.rvs(1, self.ps)
        else:
            num_undetected_a = nbinom.rvs(total_s, self.ps)

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
                self.cases[max(0, ceil(new_person.infection_time)), 1] += 1
            for n in range(num_undetected_s):
                new_person = Person(-1, curr_time-1*next(self.get_inf_time), 0, 0, 'S')
                self.infected_queue.append(len(self.people))
                self.people[len(self.people)] = new_person
                self.cases[max(0, ceil(new_person.infection_time)), 2] += 1

    def read_in_Reff(self):
        """
        Read in Reff CSV that was produced by the generate_R_L_forecasts.py script.
        """
        import pandas as pd

        df_forecast = self.Reff_all

        # Get R_I values and store in object.
        self.R_I = df_forecast.loc[(df_forecast.type == 'R_I') & (
            df_forecast.state == self.state), self.num_of_sim % 2000].values[0]

        # Get only R_L forecasts
        df_forecast = df_forecast.loc[df_forecast.type == 'R_L']
        df_forecast = df_forecast.set_index(['state', 'date'])

        dfReff_dict = df_forecast.loc[self.state,[0, 1]].to_dict(orient='index')

        Reff_lookupstate = {}
        for key, stats in dfReff_dict.items():
            # instead of mean and std, take all columns as samples of Reff
            # convert key to days since start date for easier indexing
            newkey = (key - self.start_date).days
            Reff_lookupstate[newkey] = df_forecast.loc[(self.state, key), self.num_of_sim % 2000]

        self.Reff = Reff_lookupstate

    def generate_new_cases(self, parent_key, Reff, k, travel=False):
        """
        Generate offspring for each parent, check if they travel. 
        The parent_key parameter lets us find the parent from the array self.people 
        containing the objects from the branching process.
        """
        import sys
        # Check parent category
        if self.people[parent_key].category == 'S':  # Symptomatic
            num_offspring = nbinom.rvs(n=k, p=1 - self.alpha_s*Reff/(self.alpha_s*Reff + k))
        elif self.people[parent_key].category == 'A':  # Asymptomatic
            num_offspring = nbinom.rvs(n=k, p=1 - self.alpha_a*Reff/(self.alpha_a*Reff + k))
        else:  # Imported
            Reff = self.R_I
            # Apply vaccine reduction for hotel quarantine workers
            if self.people[parent_key].infection_time >= self.hotel_quarantine_vaccine_start:
                # p_{v,h} is the proportion of hotel quarantine workers vaccinated
                p_vh = 0.9+beta.rvs(2, 4)*9/100
                # v_{e,h} is the overall vaccine effectiveness
                v_eh = 0.83+beta.rvs(2, 2)*14/100
                Reff *= (1-p_vh*v_eh)

            # Apply increase escape rate due to Delta variant.
            if self.people[parent_key].infection_time >= self.VoC_on_imported_effect_start:
                Reff = Reff*1.39*1.3

            if self.people[parent_key].infection_time < self.quarantine_change_date:
                # factor of 3 times infectiousness prequarantine changes
                num_offspring = nbinom.rvs(n=k, p=1 - self.qua_ai*Reff/(self.qua_ai*Reff + k))
            else:
                num_offspring = nbinom.rvs(n=k, p=1 - self.alpha_i*Reff/(self.alpha_i*Reff + k))

        if num_offspring > 0:

            num_sympcases = self.new_symp_cases(num_offspring)
            if self.people[parent_key].category == 'A':
                child_times = []
            for new_case in range(num_offspring):
                # define each offspring

                inf_time = self.people[parent_key].infection_time + next(self.get_inf_time)
                if inf_time > self.forecast_date:
                    self.inf_forecast_counter += 1

                # normal case within state
                if self.people[parent_key].category == 'A':
                    child_times.append(ceil(inf_time))
                if ceil(inf_time) > self.cases.shape[0]:
                    # new infection exceeds the simulation time, not recorded
                    self.cases_after = self.cases_after + 1
                else:
                    # within forecast time
                    detection_rv = random()
                    detect_time = inf_time + next(self.get_detect_time)

                    recovery_time = 0  # for now not tracking recoveries

                    if new_case <= num_sympcases-1:  # minus 1 as new_case ranges from 0 to num_offspring-1
                        # first num_sympcases are symnptomatic, rest are asymptomatic
                        category = 'S'
                        self.cases[max(0, ceil(inf_time)-1), 2] += 1

                        if self.test_campaign_date is not None:
                            # see if case is during a testing campaign
                            if inf_time < self.test_campaign_date:
                                detect_prob = self.symptomatic_detection_prob
                            else:
                                detect_prob = min(0.95, self.symptomatic_detection_prob*self.test_campaign_factor)
                        else:
                            detect_prob = self.symptomatic_detection_prob
                        if detection_rv < detect_prob:
                            # case detected
                            # only care about detected cases
                            if detect_time < self.cases.shape[0]:
                                if detect_time < self.forecast_date:
                                    if detect_time > self.forecast_date - 14:
                                        self.inf_nowcast_counter += 1
                                    elif detect_time > self.forecast_date - 60:
                                        self.inf_backcast_counter += 1
                                self.observed_cases[max(0, ceil(detect_time)-1), 2] += 1

                    else:
                        category = 'A'
                        self.cases[max(0, ceil(inf_time)-1), 1] += 1
                        #detect_time = 0
                        if self.test_campaign_date is not None:
                            # see if case is during a testing campaign
                            if inf_time < self.test_campaign_date:
                                detect_prob = self.asymptomatic_detection_prob
                            else:
                                detect_prob = min(0.95, self.asymptomatic_detection_prob*self.test_campaign_factor)
                        else:
                            detect_prob = self.asymptomatic_detection_prob
                        if detection_rv < detect_prob:
                            # case detected
                            #detect_time = inf_time + next(self.get_detect_time)
                            if detect_time < self.cases.shape[0]:
                                # counters increment before data date
                                if detect_time < self.forecast_date:
                                    if detect_time > self.forecast_date - 14:
                                        self.inf_nowcast_counter += 1
                                    elif detect_time > self.forecast_date - 60:
                                        self.inf_backcast_counter += 1
                                self.observed_cases[max(0, ceil(detect_time)-1), 1] += 1

                    # add new infected to queue
                    self.infected_queue.append(len(self.people))

                    # add person to tracked people
                    self.people[len(self.people)] = Person(parent_key, inf_time, detect_time, recovery_time, category)

    def simulate(self, end_time, sim, seed):
        """
        Simulate forward until end_time
        """
        np.random.seed(seed)
        self.num_of_sim = sim
        self.read_in_Reff()
        # generate storage for cases
        self.cases = np.zeros(shape=(end_time, 3), dtype=float)
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
        for day in range(end_time):
            # Values for a and b are initialised in import_cases_model() which is called by read_in_cases() during setup.
            a = self.a_dict[day]
            b = self.b_dict[day]
            # Dij = number of observed imported infectious individuals
            Dij = nbinom.rvs(a, 1-1/(b+1))
            # Uij = number of *unobserved* imported infectious individuals
            unobserved_a = 1 if Dij == 0 else Dij
            Uij = nbinom.rvs(unobserved_a, p=self.qi)

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
                    if new_person.detection_time <= end_time:
                        self.observed_cases[max(0, ceil(new_person.detection_time)-1), 0] += 1
                else:
                    # unobserved people
                    new_person = Person(0, day, 0, 0, 'I')
                    self.people[len(self.people)] = new_person
                    if day <= end_time:
                        self.cases[max(0, day-1), 0] += 1

        # Create queue for infected people
        self.infected_queue = deque()
        # Assign people to infected queue
        for key, person in self.people.items():
            # add to the queue
            self.infected_queue.append(key)
            # Record their times
            if person.infection_time > end_time:
                # initial undetected cases have slim chance to be infected
                # after end_time
                if person.category != 'I':
                    # imports shouldn't count for extinction counts
                    self.cases_after += 1
                    print("cases after at initialisation")

            # Cases already recorded at initialise_sim() by addding to
            # self.current

        # Record initial inferred obs including importations.
        self.inferred_initial_obs = self.observed_cases[0, :].copy()
        #print(self.inferred_initial_obs, self.current)

        # General simulation through time by proceeding through queue
        # of infecteds
        n_resim = 0
        self.bad_sim = False
        reinitialising_window = 3
        self.daycount = 0
        while len(self.infected_queue) > 0:
            day_end = self.people[self.infected_queue[0]].detection_time
            over_check = (self.inf_backcast_counter > self.max_backcast_cases or self.inf_nowcast_counter > self.max_nowcast_cases)
            # under_check = (self.inf_backcast_counter < self.min_backcast_cases or self.inf_nowcast_counter < self.min_nowcast_cases)
            if day_end < self.forecast_date and (over_check): 
                self.num_too_many += 1
                self.bad_sim = True
                break
            else:
                # check max cases for after forecast date
                if self.inf_forecast_counter > self.max_cases:
                    # hold value forever
                    if day_end < self.cases.shape[0]-1:
                        self.cases[ceil(day_end):,2] = self.cases[ceil(day_end)-2, 2]

                        self.observed_cases[ceil(day_end):, 2] = self.observed_cases[ceil(day_end)-2, 2]
                    else:
                        self.cases_after += 1
                        
                    self.num_too_many += 1
                    break

            # stop if parent infection time greater than end time
            if self.people[self.infected_queue[0]].infection_time > end_time:
                self.infected_queue.popleft()
                # print("queue had someone exceed end_time!!")
            else:

                # take approproate Reff based on parent's infection time
                curr_time = self.people[self.infected_queue[0]].infection_time
                while True:
                    # sometimes initial cases infection time is pre
                    # Reff data, so take the earliest one
                    try:
                        Reff = self.Reff[ceil(curr_time)-1]
                    except KeyError:
                        if curr_time > 0:
                            print(
                                "Unable to find Reff for this parent at time: %.2f" % curr_time)
                            raise KeyError
                        curr_time += 1
                        continue
                    break
                # generate new cases with times
                parent_key = self.infected_queue.popleft()
                # recorded within generate new cases
                self.generate_new_cases(parent_key, Reff=Reff, k=self.k)
                
        # self.people.clear()
        if self.bad_sim == False:
            # Check simulation for discrepancies
            for day in range(7, end_time):
                # each day runs through self.infected_queue

                missed_outbreak = self.data_check(day)  # True or False
                if missed_outbreak:
                    # print("missing an outbreak")
                    self.daycount += 1
                    if self.daycount >= reinitialising_window:
                        n_resim += 1
                        #print("Local outbreak in "+self.state+" not simulated on day %i" % day)
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
                        num_symp = binom.rvs(n=int(self.current[2]),p=prob_symp_given_detect)
                        # distribute observed cases over 3 days
                        # Triangularly
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
                        #print("Reinitialising with %i new cases "  % self.current[2] )

                        # reset days to zero
                        self.daycount = 0

                if n_resim > 10:
                    # print("This sim reinitilaised %i times" % n_resim)
                    self.bad_sim = True
                    n_resim = 0
                    break
                
                # Each check of day needs to simulate the cases before moving
                # to next check, otherwise will be doubling up on undetecteds
                while len(self.infected_queue) > 0:
                    day_end = self.people[self.infected_queue[0]].detection_time
                    over_check = (self.inf_backcast_counter > self.max_backcast_cases or self.inf_nowcast_counter > self.max_nowcast_cases)
                    # under_check = (self.inf_backcast_counter < self.min_backcast_cases or self.inf_nowcast_counter < self.min_nowcast_cases)
                    if day_end < self.forecast_date and (over_check): 
                    # check for exceeding max_cases
                            self.num_too_many += 1
                            self.bad_sim = True
                            break
                    else:
                        if self.inf_forecast_counter > self.max_cases:
                            day_inf = self.people[self.infected_queue[0]].infection_time
                            self.cases[ceil(day_inf):,2] = self.cases[ceil(day_inf)-2, 2]

                            self.observed_cases[ceil(day_inf):, 2] = self.observed_cases[ceil(day_inf)-2, 2]
                            self.num_too_many += 1
                            break
                        
                    # stop if parent infection time greater than end time
                    if self.people[self.infected_queue[0]].infection_time > end_time:
                        personkey = self.infected_queue.popleft()
                        print("queue had someone exceed end_time!!")
                    else:
                        # take approproate Reff based on parent's infection time
                        curr_time = self.people[self.infected_queue[0]
                                                ].infection_time
                        while True:
                            # sometimes initial cases infection time is pre
                            # Reff data, so take the earliest one
                            try:
                                Reff = self.Reff[ceil(curr_time)-1]
                            except KeyError:
                                if curr_time > 0:
                                    print(
                                        "Unable to find Reff for this parent at time: %.2f" % curr_time)
                                    raise KeyError
                                curr_time += 1
                                continue
                            break
                        # generate new cases with times
                        parent_key = self.infected_queue.popleft()
                        self.generate_new_cases(
                            parent_key, Reff=Reff, k=self.k)
                        #missed_outbreak = max(1,missed_outbreak*0.9)
                else:
                    # pass in here if while queue loop completes
                    continue
                # only reach here if while loop breaks, so break the data check
                break
        
        self.people.clear()
        gc.collect()
        if self.bad_sim:
            # return NaN arrays for all bad_sims
            self.cumulative_cases = np.empty_like(self.cases)
            self.cumulative_cases[:] = np.nan
            return (self.cumulative_cases, self.cumulative_cases, {
                'qs': self.symptomatic_detection_prob,
                'metric': np.nan,
                'qa': self.asymptomatic_detection_prob,
                'qi': self.qi,
                'alpha_a': self.alpha_a,
                'alpha_s': self.alpha_s,
                # 'accept':self.accept,
                'ps': self.ps,
                'bad_sim': self.bad_sim,
                'cases_after': self.cases_after,
                'num_of_sim': self.num_of_sim,
            }
            )
        else:
            # good sim

            # Perform metric for ABC
            # self.get_metric(end_time)

            return (
                self.cases.copy(),
                self.observed_cases.copy(), {
                    'qs': self.symptomatic_detection_prob,
                    'metric': np.nan,
                    'qa': self.asymptomatic_detection_prob,
                    'qi': self.qi,
                    'alpha_a': self.alpha_a,
                    'alpha_s': self.alpha_s,
                    # 'accept':self.metric>=0.8,
                    'ps': self.ps,
                    'bad_sim': self.bad_sim,
                    'cases_after': self.cases_after,
                    'num_of_sim': self.num_of_sim,
                }
            )

    def to_df(self, results):
        """
        Put results from the simulation into a pandas dataframe and record as h5 format. This is called externally by the run_state.py script.
        """
        import pandas as pd

        df_results = pd.DataFrame()
        n_sims = results['symp_inci'].shape[1]
        days = results['symp_inci'].shape[0]

        sim_vars = ['bad_sim', 'metrics', 'qs', 'qa', 'qi',
                    'accept', 'cases_after', 'alpha_a', 'alpha_s', 'ps']

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
        df_results.to_parquet(
            "./results/"+self.state+self.start_date.strftime(
                format='%Y-%m-%d')+"sim_R_L"+str(n_sims)+"days_"+str(days)+self.VoC_flag+self.scenario+".parquet",
        )

        return df_results

    def data_check(self, day):
        """
        A metric to calculate how far the simulation is from the actual data
        """
        try:
            actual_3_day_total = 0
            for i in range(3):
                actual_3_day_total += self.actual[max(0, day-i)]
            threshold = case_insertion_threshold*max(1, sum(self.observed_cases[max(0, day-2):day+1, 2] + self.observed_cases[max(0, day-2):day+1, 1]))
            if actual_3_day_total > threshold:
                return min(3, actual_3_day_total/threshold)
            else:
                # long absence, then a case, reintroduce
                week_in_sim = sum(self.observed_cases[max(0, day-7):day+1, 2] + self.observed_cases[max(0, day-7):day+1, 1])
                if week_in_sim == 0:
                    if actual_3_day_total > 0:
                        return actual_3_day_total

                # no outbreak missed
                return False

        except KeyError:
            #print("No cases on day %i" % day)
            return False

    def read_in_cases(self):
        """
        Read in NNDSS case data to measure incidence against simulation. Nothing is returned as results are saved in object.
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

        # calculate window of cases to measure against
        if df.index.values[-1] > 60:
            # if final day of data is later than day 90, then remove first 90 days
            forecast_days = self.end_time-self.forecast_date
            self.cases_to_subtract = sum(df.local.values[:-1*(60+forecast_days)])
            self.cases_to_subtract_now = sum(df.local.values[:-1*(14+forecast_days)])
        else:
            self.cases_to_subtract = 0
            self.cases_to_subtract_now = 0
            
        #self.imported_total = sum(df.imported.values)
        self.max_cases = max(500000, sum(df.local.values) + sum(df.imported.values))
        
        # +/- factors for number of cases to use in the current period to determine proximity to data
        backcast_factor = 3
        nowcast_factor = 1.5
        
        backcast_cases = (sum(df.local.values) - self.cases_to_subtract)
        nowcast_cases = (sum(df.local.values) - self.cases_to_subtract_now)
        
        # max limits are just 1+factor * number of cases over a time horizon
        self.max_backcast_cases = max(100, backcast_cases * backcast_factor)
        self.max_nowcast_cases = max(10, nowcast_cases * nowcast_factor)
        # min limits are 1-factor * number of cases over a time horizon. we take the maximum of 
        # 0 and the estimated matching interval as there's the possibility for a negative number of
        # cases if the factor >= 1. 
        
        print("Local cases in last 14 days is %i" % nowcast_cases)

        print('Max limits: ', self.max_cases, self.max_backcast_cases, self.max_nowcast_cases)

        self.actual = df.local.to_dict()

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

    def generate_times(self,  i=3.64, j=3.07, m=5.505, n=0.948, size=50000):
        """
        Helper function. Generate large amount of gamma draws to save on simulation time later
        """
        self.inf_times = np.random.gamma(i/j, j, size=size)  # shape and scale
        # self.detect_times = np.random.gamma(m/n, n, size=size)
        self.detect_times = 1 + np.random.gamma(2, 1, size=size)

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

    def new_symp_cases(self, num_new_cases: int):
        """
        Given number of new cases generated, assign them to symptomatic (S) with probability ps
        """
        # repeated Bernoulli trials is a Binomial (assuming independence of development of symptoms)

        symp_cases = binom.rvs(n=num_new_cases, p=self.ps)

        return symp_cases
