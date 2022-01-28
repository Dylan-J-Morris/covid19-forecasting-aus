"""
Main code for running the branching process. This invokes other Julia scripts in order 
to run: 
    - read_in_TP.jl:
        Code for reading the TP csv files saved from the fitting and
        forecasting parts of the model.
    - helper_functions.jl: 
        Code which is commonly reused in the model. This involves non-specialised functions
        for sampling onset and infection times, and for sampling from a NegativeBinomial 
        and Binomial distribution using asymptotic limiting properties. 
    - forecast_types.jl 
        Code for some user defined types. This lets us group variables of importance. There 
        is a struct for the dynamical constants, properties of the data and a 
        simple mapping for moving between types and indices. 
    - simulation_consistency.jl 
        Code for checking consistency of simulations against the data. The code in this file 
        is also responsible for injecting cases into the simulation if we are below some 
        threshold. 
    - assumptions.jl 
        One of the most important files. Keeps track of all assumptions in a function that 
        is used to initialise the system constants.
        
The key logic of this simulation is featured in the function simulate_branching_process(). 
This is written in such a way that it is relatively straightforward to follow the logic of 
the processes and functions are called in a systematic way. simulate_branching_process() 
will return reshaped arrays that are the "good" simulated realisations based off 
the consistency checks. 
"""

using Distributions
using Random 
using Statistics 
using LinearAlgebra
using ProgressBars
using StaticArrays

include("helper_functions.jl")
include("forecast_types.jl")
include("simulation_consistency.jl")
include("assumptions.jl")

##### Methods

function initialise_state_arrays(
    sim_duration, 
    observation_period, 
    nsims,
)
    """
    Initialising a state array object with arrays of zeros. 
    We pad the infection array Z with 35 days to account for 
    infections occuring prior to the simulation period. We do this
    separately (and not in a struct) as the arrays are large. 
    """
    
    Z = zeros(Int, sim_duration+35, 3, nsims)
    D = zeros(Int, sim_duration, 3, nsims)
    U = zeros(Int, sim_duration, 3, nsims)
    
    sim_realisations = Realisations(Z, D, U)
    
    return sim_realisations
    
end


function initialise_population!(
    forecast::Forecast, 
    start_day,
    D0, 
)
    """
    Fill the infection array for each simulation, inititialising the states of
    all the simulations. 
    """
    Z = forecast.sim_realisations.Z
    D = forecast.sim_realisations.D
    U = forecast.sim_realisations.U
    
    # extract prob of being symptomatic and prob of symptoms given detection
    p_symp = forecast.sim_constants.p_symp
    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp
    T_end = forecast.sim_features.T_end
    individual_type_map = forecast.individual_type_map
    
    total_initial_cases = sum(D0)
    
    for sim in 1:size(Z, 3)
        
        # infer some initial undetected symptomatic
        num_symptomatic_undetected = 0
        if D0.S == 0
            num_symptomatic_undetected = sample_negative_binomial_limit(
                1, 
                p_detect_given_symp,
            )
        else
            num_symptomatic_undetected = sample_negative_binomial_limit(
                D0.S, 
                p_detect_given_symp,
            )
        end

        # infer some initial undetected asymptomatic
        total_symptomatic = D0.S + num_symptomatic_undetected
        num_asymptomatic_undetected = 0
        if total_symptomatic == 0
            num_asymptomatic_undetected = sample_negative_binomial_limit(1, p_symp)
        else
            num_asymptomatic_undetected = sample_negative_binomial_limit(
                total_symptomatic, 
                p_symp, 
            )
        end
        
        # total initial cases include the undetected
        total_initial_cases_sim = total_initial_cases + 
            num_symptomatic_undetected + 
            num_asymptomatic_undetected
        
        if total_initial_cases_sim > 0  
            
            infection_times = MVector{total_initial_cases_sim}(
                zeros(Int, total_initial_cases_sim)
            )
            
            # sample negative infection times as these individuals are assumed to occur 
            # before the simulation 
            for i in 1:total_initial_cases_sim
                infection_times[i] = -sample_onset_time()
            end
            
            start_day[sim] = minimum(infection_times)
            idx = 1
            
            for _ in 1:D0.S
                Z[infection_times[idx]+36,individual_type_map.S,sim] += 1
                idx += 1
                D[1,individual_type_map.S,sim] += 1
            end
            
            for _ in 1:D0.A
                Z[infection_times[idx]+36,individual_type_map.A,sim] += 1
                idx += 1
                D[1,individual_type_map.A,sim] += 1
            end
            
            for _ in 1:num_symptomatic_undetected
                Z[infection_times[idx]+36,individual_type_map.S,sim] += 1
                idx += 1
                U[1,individual_type_map.S,sim] += 1
            end
            
            for _ in 1:num_asymptomatic_undetected
                Z[infection_times[idx]+36,individual_type_map.A,sim] += 1
                idx += 1
                U[1,individual_type_map.A,sim] += 1
            end
        end
    end
    
    return nothing
    
end


function import_cases_model!(
	forecast::Forecast,
    import_cases, 
	forecast_start_date, 
)
	"""
	A model to handle international imports in the forecasting of cases. This can be run 
    prior to the standard forecasting model and will pre-fill D, U and Z for each 
    simulation.

	We assume that cases 
	arise from 
	D[t] ∼ NegBin(a[t], 1/(b+1)) 
	where 
	a[t] = α + f(I[t]) 
	b = β + 1
	are the parameters. The function f() is an exponential weighted moving average such 
    that 
	f(I[t]) = ϕ I[t] + (1-ϕ) f(I[t-1])
	and α, β are hyperparameters fixed at 0.5 and 0.2, respectively. We take b = β + 1 
    as we assume a fixed period of 1 day for estimating the posterior. 
	"""
	
    Z = forecast.sim_realisations.Z
    D = forecast.sim_realisations.D
    U = forecast.sim_realisations.U
    
    T_observed = forecast.sim_features.T_observed
    T_end = forecast.sim_features.T_end
    
	# number of days after t = 0 (2020/03/01)
	days_into_forecast = Day(Date(forecast_start_date) - Date("2020-03-01")).value
	
	# the number of days the simulation is run for 
	forecast_days_plus_padding = -10:T_end-5
	# posterior a, b for sampling number of imports on day t 
	a_post = MVector{length(forecast_days_plus_padding)}(
        zeros(Float64, length(forecast_days_plus_padding))
    )
	
	# current ema is taken 11 days before the forecast start date
	current_ema = zero(Float64)
    current_ema += import_cases[days_into_forecast - 11]
    
	for (i, t) in enumerate(forecast_days_plus_padding)	
		# if within observation period, update the posterior. Otherwise use the last 
		# available measurement 
		if t <= T_observed
			count_on_day = import_cases[days_into_forecast + t]
			current_ema = forecast.sim_constants.ϕ*count_on_day + 
                (1-forecast.sim_constants.ϕ)*current_ema
			a_post[i] = forecast.sim_constants.prior_alpha + current_ema
		else
			a_post[i] = forecast.sim_constants.prior_alpha + current_ema
		end
	end
	
    # assuming a fixed period of 1 day (so this is the same for the whole period)
    b_post = forecast.sim_constants.prior_beta + 1
	
    # preallocate daily import vectors 
    D_I = MVector{length(a_post)}(zeros(Int, length(a_post)))
    U_I = MVector{length(a_post)}(zeros(Int, length(a_post)))
    unobserved_I = MVector{length(a_post)}(zeros(Int, length(a_post)))
     
	for sim in 1:size(D, 3)
    
		# sample the number of imported cases detected and undetected 
        # (broadcasting to allocate in place)
		D_I .= rand.(NegativeBinomial.(a_post, b_post / (1 + b_post)))
        for (i, d) in enumerate(D_I)
    		unobserved_I[i] = d == 0 ? 1 : d
        end
		U_I .= rand.(NegativeBinomial.(unobserved_I, forecast.sim_constants.qi))
	
		for (i, t) in enumerate(forecast_days_plus_padding)
            # day of infection is time t which is why the infection time in 
			if D_I[i] + U_I[i] > 0
				# add sampled imports to observation arrays
                assign_to_arrays_and_times!(
                    forecast, 
                    t, 
                    sim, 
                    num_imports_detected=D_I[i], 
					num_imports_undetected=U_I[i], 
                )
			end
		end
	end
	
	return nothing
	
end


function get_proportion_infected(
    day, 
    cases_pre_forecast, 
    D_sim, 
    U_sim, 
    N
)
    """
    Calculate the proportion of cases that have been infected up to current day.
    """
    total_cumulative_cases = cases_pre_forecast
    
    for j in 1:3
        for i in 1:day
            total_cumulative_cases += D_sim[i,j] + U_sim[i,j]
        end
    end
    
    proportion_infected = total_cumulative_cases / N
    
    return proportion_infected
    
end


function sample_offspring!(
    forecast::Forecast, 
    t,  
    TP_local_sim, 
    TP_import_sim, 
    TP_indices, 
    susceptible_depletion_sim,
    sim, 
)
    """
    Sample offspring for all adults at time t. This uses the fact that the sum of 
    N NB(s, p) is NB(N*s, p) and hence we can sample ALL offspring for a particular type 
    of adults. 
    """
    Z = forecast.sim_realisations.Z
    D = forecast.sim_realisations.D
    U = forecast.sim_realisations.U
    
    α_s = forecast.sim_constants.α_s
    α_a = forecast.sim_constants.α_a
    p_symp = forecast.sim_constants.p_symp
    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp
    p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp
    # determine overdispersion parameter based on whether Omicron is dominant or not 
    k = forecast.sim_constants.k_delta * (t < forecast.sim_features.omicron_dominant_day) + 
        forecast.sim_constants.k_omicron * (t >= forecast.sim_features.omicron_dominant_day)
    
    # extract fixed simulation values
    cases_pre_forecast = forecast.sim_features.cases_pre_forecast
    N = forecast.sim_features.N
    
    # initialise number of offspring outside of the loop 
    num_offspring = zero(Int)
    
    # pointer to the current day's infections 
    Z_tmp = @view Z[t+36,:,sim]
    
    # if the total number of infected individuals at time t is 0, return immediately
    if sum(Z_tmp) > 0 
        # extract the number of each parent who are infected on day t 
        (S_parents, A_parents, I_parents) = Z_tmp

        # reset infections on day t (so we can inject new cases)
        # find day based on the indices representing time from forecast origin
        TP_ind = findfirst(t == ind for ind in TP_indices)
        
        # create views and use a function barrier to decrease allocations 
        D_sim = @views D[:,:,sim]
        U_sim = @views U[:,:,sim]
        
        # take max of 0 and TP to deal with cases of depletion factor giving TP = 0
        proportion_infected = get_proportion_infected(
            t, cases_pre_forecast, D_sim, U_sim, N
        )
        TP_local_parent = max(
            0, 
            TP_local_sim[TP_ind] * (1 - susceptible_depletion_sim*proportion_infected)
        )
        TP_import_parent = max(
            0, 
            TP_import_sim[TP_ind] * (1 - susceptible_depletion_sim*proportion_infected)
        )
        
        # total number of infections arising from all parents infected at time t 
        # (sum of M NB(r,p) is NB(r*M,p))
        if S_parents > 0 && TP_local_parent > 0
            s = k*S_parents
            p = 1 - α_s*TP_local_parent/(k+α_s*TP_local_parent)
            num_offspring += sample_negative_binomial_limit(s, p)
        end
        
        if A_parents > 0 && TP_local_parent > 0
            s = k*A_parents
            p = 1 - α_a*TP_local_parent/(k+α_a*TP_local_parent)            
            num_offspring += sample_negative_binomial_limit(s, p)
        end
        
        if I_parents > 0 && TP_import_parent > 0
            s = k*I_parents
            # p_{v,h} is the proportion of hotel quarantine workers vaccinated
            p_vh = 0.9+rand(Beta(2, 4))*9/100
            # v_{e,h} is the overall vaccine effectiveness
            v_eh = 0.83+rand(Beta(2, 2))*14/100
            TP_import_parent *= (1-p_vh*v_eh)*1.39*1.3
            p = 1 - TP_import_parent/(k+TP_import_parent)
            num_offspring += sample_negative_binomial_limit(s, p)
        end
    
        # we dont ever get import offspring 
        if num_offspring > 0
            # sample those symptomatic
            num_symptomatic_offspring = sample_binomial_limit(num_offspring, p_symp)
            num_asymptomatic_offspring = num_offspring-num_symptomatic_offspring
            # sample number detected of each type  
            num_symptomatic_offspring_detected = sample_binomial_limit(
                num_symptomatic_offspring, p_detect_given_symp
            )
            num_asymptomatic_offspring_detected = sample_binomial_limit(
                num_asymptomatic_offspring, p_detect_given_asymp
            )
            # calc number undetected 
            num_symptomatic_offspring_undetected = num_symptomatic_offspring - 
                num_symptomatic_offspring_detected
            num_asymptomatic_offspring_undetected = num_asymptomatic_offspring - 
                num_asymptomatic_offspring_detected
            
            # add the onsets to observation arrays
            assign_to_arrays_and_times!(
                forecast, 
                t, 
                sim, 
                num_symptomatic_detected=num_symptomatic_offspring_detected, 
                num_symptomatic_undetected=num_symptomatic_offspring_undetected, 
                num_asymptomatic_detected=num_asymptomatic_offspring_detected, 
                num_asymptomatic_undetected=num_asymptomatic_offspring_undetected, 
            )
            
            # zero out infections on day t so we can restart sim 
            # and keep the progress 
            Z_tmp .= 0    
        end
    end

    return nothing 
    
end


function assign_to_arrays_and_times!(
    forecast::Forecast, 
    day, 
    sim;
    num_symptomatic_detected=0,
    num_symptomatic_undetected=0,
    num_asymptomatic_detected=0,
    num_asymptomatic_undetected=0,
    num_imports_detected=0, 
    num_imports_undetected=0,       
)
    """
    Increment the arrays D and U based on sampled offsrping counts. The number of 
    each offspring are keyword parameters which are default 0. This allows us to call 
    this function from multiple spots. 
    """
    
    # TODO: refactor this using multiple dispatch. 
    
    Z = forecast.sim_realisations.Z
    D = forecast.sim_realisations.D
    U = forecast.sim_realisations.U
    individual_type_map = forecast.individual_type_map
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day
    T_end = forecast.sim_features.T_end
    
    # sampling detected cases 
    for _ in 1:num_symptomatic_detected
        infection_time = day + sample_infection_time(omicron = day >= omicron_dominant_day)
        if infection_time <= T_end 
            Z[infection_time+36,individual_type_map.S,sim] += 1
        end
        
        onset_time = infection_time + 
            sample_onset_time(omicron = infection_time >= omicron_dominant_day)
        if onset_time < 1 
            D[1,individual_type_map.S,sim] += 1
        elseif onset_time <= T_end
            # time+1 as first index is day 0
            D[onset_time+1,individual_type_map.S,sim] += 1
        end
    end
    
    for _ in 1:num_asymptomatic_detected
        infection_time = day + sample_infection_time(omicron = day >= omicron_dominant_day)
        if infection_time <= T_end 
            Z[infection_time+36,individual_type_map.A,sim] += 1
        end
        
        onset_time = infection_time + 
            sample_onset_time(omicron = infection_time >= omicron_dominant_day)
        if onset_time < 1 
            D[1,individual_type_map.A,sim] += 1
        elseif onset_time <= T_end
            # time+1 as first index is day 0
            D[onset_time+1,individual_type_map.A,sim] += 1
        end
    end
    
    # sampling undetected cases 
    for _ in 1:num_symptomatic_undetected
        infection_time = day + sample_infection_time(omicron = day >= omicron_dominant_day)
        if infection_time <= T_end 
            Z[infection_time+36,individual_type_map.S,sim] += 1
        end
        
        onset_time = infection_time + 
            sample_onset_time(omicron = infection_time >= omicron_dominant_day)
        if onset_time < 1 
            U[1,individual_type_map.S,sim] += 1
        elseif onset_time <= T_end
            # time+1 as first index is day 0
            U[onset_time+1,individual_type_map.S,sim] += 1
        end
    end
    
    for _ in 1:num_asymptomatic_undetected
        infection_time = day + 
            sample_infection_time(omicron = day >= omicron_dominant_day)
        if infection_time <= T_end 
            Z[infection_time+36,individual_type_map.A,sim] += 1
        end
        
        onset_time = infection_time + 
            sample_onset_time(omicron = infection_time >= omicron_dominant_day)
        if onset_time < 1 
            U[1,individual_type_map.A,sim] += 1
        elseif onset_time <= T_end
            # time+1 as first index is day 0
            U[onset_time+1,individual_type_map.A,sim] += 1
        end
    end
    
    # imports are treated slightly differently as the infection time is inferred 
    # through the simpler model
    for _ in 1:num_imports_detected
        infection_time = day
        if infection_time <= T_end 
            Z[infection_time+36,individual_type_map.I,sim] += 1
        end
        
        onset_time = infection_time + 
            sample_onset_time(omicron = infection_time >= omicron_dominant_day)
        if onset_time < 1 
            D[1,individual_type_map.I,sim] += 1
        elseif onset_time <= T_end
            # time+1 as first index is day 0
            D[onset_time+1,individual_type_map.I,sim] += 1
        end
    end
    
    for _ in 1:num_imports_undetected
        infection_time = day
        if infection_time <= T_end 
            Z[infection_time+35,individual_type_map.I,sim] += 1
        end
        
        onset_time = infection_time + 
            sample_onset_time(omicron = infection_time >= omicron_dominant_day)
        if onset_time < 1 
            U[1,individual_type_map.I,sim] += 1
        elseif onset_time <= T_end
            # time+1 as first index is day 0
            U[onset_time+1,individual_type_map.I,sim] += 1
        end
    end
    
    return nothing
    
end


function simulate_branching_process(
    D0, 
    N, 
    nsims, 
    local_cases, 
    import_cases, 
    cases_pre_forecast,
    forecast_start_date, 
    date, 
    omicron_dominant_date,
    state,
)
    """
    Simulate branching process nsims times conditional on the cumulative observed cases, with 
    initial state X0 = (S, A, I) (a named tuple). T is the duration of the simulation and N is the 
    population size which is currently unused. 
    """
    # read in the TP for a given state and date
    (TP_indices, TP_local, TP_import) = create_state_TP_matrices(
        forecast_start_date, date, state
    )
    # read in susceptible_depletion parameters 
    susceptible_depletion = read_in_susceptible_depletion(date)
    
    # length of observed time series 
    T_observed = length(local_cases)
    # duration of forecast simulation
    T_end = sum(ind > 0 for ind in TP_indices)
     
    cumulative_local_cases = cumsum(local_cases)
    max_restarts = 10
    good_sims = ones(Bool, nsims)
    good_TPs = zeros(Bool, size(TP_local, 2))
    
    # get simulation constants 
    (sim_constants, individual_type_map) = set_simulation_constants(state)
    # calculate the upper and lower limits for checking sims against data 
    (sim_features, window_lengths, min_cases, max_cases) = get_simulation_limits(
        local_cases, 
        forecast_start_date,
        omicron_dominant_date,
        cases_pre_forecast, 
        TP_indices, 
        N, 
        T_observed, 
        T_end,
    )
    
    # the range of days for the simulation
    day_range = -35:T_end
        
    # initialise state arrays (T_end + 1 as we include 0 day)
    sim_realisations = initialise_state_arrays(T_end+1, T_observed+1, nsims)
    
    # put everything into a forecast object
    forecast = Forecast(
        sim_features,
        sim_realisations,
        sim_constants, 
        individual_type_map,
    )
    
    # vector to hold the first day of infections for each simulation. 
    start_day = zeros(Int, nsims)
    
    # generate initial infected population 
    initialise_population!(
        forecast,
        start_day, 
        D0, 
    )
    
    import_cases_model!(
        forecast,
        import_cases, 
        forecast_start_date, 
    )
    
    good_TPs_inds = zeros(Int, nsims)
    
    Threads.@threads for sim in ProgressBar(1:nsims)
        # counts in each window 
        case_counts = zeros(Int, length(max_cases))
        # sample the TP/susceptible_depletion for this sim
        TP_ind = sim % 2000 == 0 ? 2000 : sim % 2000
        TP_local_sim = @view TP_local[:,TP_ind]
        TP_import_sim = @view TP_import[:,TP_ind]
        susceptible_depletion_sim = susceptible_depletion[TP_ind]
    
        bad_sim = false
        injected_cases = false
        
        # initialise counter to first day, when last injection occurred
        day = start_day[sim]
        last_injection = 0
        n_restarts = 0

        while day <= day_range[end]
            
            sample_offspring!(
                forecast,
                day, 
                TP_local_sim, 
                TP_import_sim, 
                TP_indices,
                susceptible_depletion_sim,
                sim, 
            )
            
            if day > 0
                reinitialise_allowed =  day >= 7 && 
                    day <= T_observed && 
                    n_restarts < max_restarts && 
                    day - last_injection >= 3
                    
                (bad_sim, injected_cases) = check_sim!(
                    forecast,
                    forecast_start_date,
                    omicron_dominant_date,
                    case_counts, 
                    sim,
                    local_cases, 
                    window_lengths,
                    min_cases, 
                    max_cases,
                    reinitialise_allowed;
                    day=day,
                )
            end
            
            # reset day counter if we injected some cases
            if !injected_cases 
                day += 1
            else 
                day = start_day[sim]
                n_restarts += 1
                injected_cases = false
                last_injection = day
                if n_restarts > max_restarts 
                    bad_sim = true
                end
            end
            
            bad_sim && break
        end
        
        if !bad_sim
            reinitialise_allowed = false
            (bad_sim, injected_cases) = check_sim!(
                forecast,
                forecast_start_date,
                omicron_dominant_date,
                case_counts, 
                sim,
                local_cases, 
                window_lengths,
                min_cases, 
                max_cases,
                reinitialise_allowed,
            )
        end
        
        if bad_sim 
            good_sims[sim] = false
            good_TPs_inds[sim] = -1
        else
            good_TPs[TP_ind] = true
            good_TPs_inds[sim] = TP_ind
        end
    end
    
    println("======================")
    println("Summary for ", state, ":")
    println("- ", sum(good_sims), " good simulations.")
    println("- ", sum(good_TPs), " unique TPs sampled.") 
    println("======================")
    
    # keep only good sim results and truncate the TP to the same period as per D and U
    D_good = forecast.sim_realisations.D[:,:,good_sims]
    U_good = forecast.sim_realisations.U[:,:,good_sims]
    good_TPs_inds = good_TPs_inds[good_sims]
    
    TP_local_truncated = TP_local[(end-T_end):end,:]
    # calculate the total number of infections by day t
    Z_good = sum(forecast.sim_realisations.Z[:,:,good_sims], dims = 2)
    Z_good_cumulative = cumsum(Z_good, dims = 1)[(end-T_end):end,1,:]
    prop_infected = zeros(Int, size(TP_local_truncated, 1))
    
    # new array to hold the correctly sampled values
    TP_local_sims = zeros(size(TP_local_truncated, 1), length(good_TPs_inds))
    
    # adjust the local TP by susceptible depletion model 
    for (i, TP_ind) in enumerate(good_TPs_inds)
        prop_infected = Z_good_cumulative[:,i] ./ N
        TP_local_sims[:,i] = TP_local_truncated[:,TP_ind] .* 
            (1 .- susceptible_depletion[TP_ind]*prop_infected)
    end
    
    return (D_good, U_good, TP_local_sims)
    
end