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
will return reshaped arrays that are the good simulated realisations based off 
the consistency checks. 
"""

using Distributed 
using Distributions
using Random 
using Statistics 
using LinearAlgebra
using ProgressBars
using StaticArrays
using SharedArrays
using ProgressMeter

include("helper_functions.jl")
include("forecast_types.jl")
include("simulation_consistency.jl")

##### Methods
function initialise_population!(
    forecast::Forecast, 
    D0, 
)
    """
    Fill the infection array for each simulation, inititialising the states of
    all the simulations. 
    """
    Z = forecast.sim_realisation.Z
    Z_historical = forecast.sim_realisation.Z_historical
    D = forecast.sim_realisation.D
    U = forecast.sim_realisation.U
    
    # extract prob of being symptomatic and prob of symptoms given detection 
    # NOTE: this currently assumes we are starting in the Delta wave (which is true as
    # of 31/1/2022) but we will need to update this if we shift the starting date into 
    # an only Omicron period. 
    p_symp = forecast.sim_constants.p_symp.delta 
    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.delta 
    T_end = forecast.sim_features.T_end
    individual_type_map = forecast.individual_type_map
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 

    # infer some initial undetected symptomatic
    num_S_undetected = 0
    
    if D0.S == 0
        num_S_undetected = rand(NegativeBinomial(1, p_detect_given_symp))
    else
        num_S_undetected = rand(NegativeBinomial(D0.S, p_detect_given_symp))
    end

    # infer some initial undetected asymptomatic
    total_S = D0.S + num_S_undetected
    num_A_undetected = 0
    
    if total_S == 0
        num_A_undetected = rand(NegativeBinomial(1, p_symp))
    else
        num_A_undetected = rand(NegativeBinomial(total_S, p_symp))
    end
    
    # total initial cases include the undetected
    total_initial_cases = sum(D0)
    total_initial_cases += num_S_undetected + num_A_undetected
    
    start_day = 0
    
    # reset arrays 
    D .= 0
    U .= 0
    Z .= 0
    Z_historical .= 0
        
    if total_initial_cases > 0  
        inf_times = MVector{total_initial_cases}(
            zeros(Int, total_initial_cases)
        )
        
        # sample negative infection times as these individuals are assumed to occur 
        # before the simulation 
        for i in 1:total_initial_cases
            inf_times[i] = -sample_onset_time(false)
        end
        
        # the starting day is the earliest infection time deemed to contribute to the sim. 
        start_day = minimum(inf_times)
        idx = 1
        
        for _ in 1:D0.S
            Z[map_day_to_index_Z(inf_times[idx]), individual_type_map.S] += 1
            idx += 1
            D[1, individual_type_map.S] += 1
        end
        
        for _ in 1:D0.A
            Z[map_day_to_index_Z(inf_times[idx]), individual_type_map.A] += 1
            idx += 1
            D[1, individual_type_map.A] += 1
        end
        
        for _ in 1:num_S_undetected
            Z[map_day_to_index_Z(inf_times[idx]), individual_type_map.S] += 1
            idx += 1
            U[1, individual_type_map.S] += 1
        end
        
        for _ in 1:num_A_undetected
            Z[map_day_to_index_Z(inf_times[idx]), individual_type_map.A] += 1
            idx += 1
            U[1, individual_type_map.A] += 1
        end
    end
    
    return start_day
    
end


function import_cases_model!(
	forecast::Forecast,
    import_cases, 
	forecast_start_date, 
)
	"""
	A model to handle international imports in the forecasting of cases. This is run 
    prior to the standard forecasting model and will pre-fill D, U and Z for each 
    simulation.
	We assume that cases arise from 
	D[t] ~ NegBin(a[t], 1/(b+1)) 
	where 
	a[t] = alpha + f(I[t]) 
	b = beta + 1
	are the parameters. The function f() is an exponential weighted moving average such 
    that
	f(I[t]) = phi I[t] + (1-phi) f(I[t-1])
	and alpha, beta are hyperparameters fixed at 0.5 and 0.2, respectively. We 
    take b = β + 1 as we assume a fixed period of 1 day for estimating the posterior. 
	"""
	
    Z = forecast.sim_realisation.Z
    D = forecast.sim_realisation.D
    U = forecast.sim_realisation.U
    
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day
    
    # grab the correct parameters for the particular dominant strain 
    qi = 0.0
    
    T_observed = forecast.sim_features.T_observed
    T_end = forecast.sim_features.T_end
    
	# number of days after t = 0 (2020/03/01)
	days_into_forecast = Day(Date(forecast_start_date) - Date("2020-03-01")).value
	
	# the number of days the simulation is run for 
	forecast_days_plus_padding = -10:T_end
	# posterior a, b for sampling number of imports on day t 
	a_post = MVector{length(forecast_days_plus_padding)}(
        zeros(Float64, length(forecast_days_plus_padding))
    )
	
    # TODO: Check this calculation makes sense. 
    # A quick comment: 
    # I think that the +7 deals with the truncation of the data at the end of the fitting 
    # period and then the +4 ensures that we are considering the infection dates of those 
    # individuals that have later onsets. I.e. individual is detected at time t, they are 
    # infected at time t - 4, and so we want the count at time t. Bit roundabout to get 
    # there but saves us writing more code. 
    
	# current ema is taken 11 days before the forecast start date
	current_ema = zero(Float64)
    
    # imports show up in the data before
    # import_infections = import_cases[5:end]
    inc_period_shift = 5
    current_ema += import_cases[inc_period_shift + days_into_forecast - 11]
    
	for (i, t) in enumerate(forecast_days_plus_padding)	
		# if within observation period, update the posterior. Otherwise use the last 
		# available measurement 
		if t <= T_observed - inc_period_shift
			count_on_day = import_cases[inc_period_shift + days_into_forecast + t]
			current_ema = forecast.sim_constants.ϕ * count_on_day + 
                (1 - forecast.sim_constants.ϕ) * current_ema
			a_post[i] = forecast.sim_constants.prior_alpha + current_ema
		else
			a_post[i] = forecast.sim_constants.prior_alpha + current_ema
		end
	end
	
    # assuming a fixed period of 1 day (so this is the same for the whole period)
    b_post = forecast.sim_constants.prior_beta + 1
     
    for (i, t) in enumerate(forecast_days_plus_padding)
        # sample detected imports 
        D_I = rand(NegativeBinomial(a_post[i], b_post / (1 + b_post)))
        # see whether they're observed 
        unobserved_I = D_I == 0 ? 1 : D_I
        # based on the switch to omicron (in terms of infections so -5 days)
        if t < omicron_dominant_day
            qi = forecast.sim_constants.qi.delta 
        else
            qi = forecast.sim_constants.qi.omicron 
        end
        U_I = rand(NegativeBinomial(unobserved_I, qi))
        
        # day of infection is time t which is why the infection time in 
        if D_I + U_I > 0
            assign_to_arrays_and_times!(
                forecast, 
                t, 
                is_omicron = false, 
                num_I_detected = D_I, 
                num_I_undetected = U_I, 
            )
        end
    end
	
	return nothing
	
end

function calculate_historical_infected(
    forecast, 
    local_cases,
    day, 
    N;
    adjust_TP = false, 
)

    Z_historical = forecast.sim_realisation.Z_historical
    T_observed = forecast.sim_features.T_observed
    cases_pre_forecast = forecast.sim_features.cases_pre_forecast
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 
    p_detect_delta = forecast.sim_constants.p_detect.delta 
    p_detect_omicron = forecast.sim_constants.p_detect.omicron 
    
    cases = 0
    cases_pre_omicron = 0
    cases_after_omicron = 0
    
    # if adjust_TP  
    #     τ = map_day_to_index_cases(day)
    #     # if adjusting the TP then use the raw daily cases in the calculation of the 
    #     # depletion factor. We need to also scale the daily cases by the assumed 
    #     # CAR.
    #     cases_pre_omicron = 1 / p_detect_delta * sum(
    #         @view local_cases[
    #             begin:min(τ, map_day_to_index_cases(omicron_dominant_day) - 1)
    #         ]
    #     )
        
    #     if day >= omicron_dominant_day
    #         cases_after_omicron = 1 / p_detect_omicron * sum(
    #             @view local_cases[
    #                 map_day_to_index_cases(omicron_dominant_day):min(τ, length(local_cases))
    #             ]
    #         )
    #     end
        
    #     cases += cases_pre_omicron + cases_after_omicron
        
    #     if τ > length(local_cases)
    #         τ2 = min(map_day_to_index_Z(day), size(Z_historical, 1))
    #         cases += sum(@view Z_historical[map_day_to_index_Z(T_observed + 1):τ2])
    #     end
    # else    
    #     τ = min(map_day_to_index_Z(day), size(Z_historical, 1))
    #     cases = sum(@view Z_historical[begin:τ])
    # end
    
    τ = min(map_day_to_index_Z(day), size(Z_historical, 1))
    cases = sum(@view Z_historical[begin:τ])
    
    return cases 

end


function sigmoid(t, prop_pars)
    (m₀, m₁, r, τ) = prop_pars
    res = m₀ + (m₁ - m₀) / (1 + exp(-r * (t - τ)))
    
    return res 
    
end


function sample_offspring!(
    forecast::Forecast, 
    day,  
    TP, 
    prop_pars, 
    susceptible_depletion_sim,
    local_cases;
    adjust_TP = false,
)
    """
    Sample offspring for all adults at time t. This uses the fact that the sum of 
    N NB(s, p) is NB(N*s, p) and hence we can sample ALL offspring for a particular type 
    of adults. 
    """
    
    Z = forecast.sim_realisation.Z
    Z_historical = forecast.sim_realisation.Z_historical
    D = forecast.sim_realisation.D
    U = forecast.sim_realisation.U
    omicron_start_day = forecast.sim_features.omicron_start_day 
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 
    T_observed = forecast.sim_features.T_observed
    state = forecast.sim_features.state
    
    # extract fixed simulation values
    cases_pre_forecast = forecast.sim_features.cases_pre_forecast
    N = forecast.sim_features.N
    
    # pointer to the current day's infections 
    Z_tmp = @view Z[map_day_to_index_Z(day), :]
    
    total_parents = sum(Z_tmp)
    
    if total_parents > 0
        # if we are adjusting the TP, we use the Reff up until the last 30 days and then we
        # need to scale the TP afterwards. The Reff inherently features the depletion of 
        # susceptibles in its calculation (of sorts) and so can be considered the "true" 
        # factor 
        total_infected = calculate_historical_infected(
            forecast, 
            local_cases, 
            day,  
            N; 
            adjust_TP = adjust_TP,
        )
        
        (S_parents, A_parents, I_parents) = Z_tmp
        
        import_multiplier = 1.0
        
        if day < omicron_dominant_day - 30
            # p_{v,h} is the proportion of hotel quarantine workers vaccinated
            p_vh = 0.9 + rand(Beta(2, 4)) * 9 / 100
            # v_{e,h} is the overall vaccine effectiveness
            v_eh = 0.83 + rand(Beta(2, 2)) * 14 / 100
            import_multiplier = (1 - p_vh * v_eh)
        end
        
        # sum up the number local infections on the current day
        Z_historical[map_day_to_index_Z(day)] += S_parents + A_parents
        # Z_tmp gets emptied at the end of sampling so we won't be double counting 
        # zero out infections on day t so we can restart sim and keep the progress 
        Z_tmp .= 0    
        
        S_parents_omicron = 0
        A_parents_omicron = 0
        I_parents_omicron = 0
        
        if day >= omicron_start_day
            # number of days into the omicron wave
            t_omicron = day - omicron_start_day 
            prop_omicron = 0.0
            # if we are in WA, TAS or NT, the outbreak began late enough to suggest omicron 
            # proportion was high, so use the inferred long term proportions. Otherwise use the
            # sigmoidal model. 
            # if state in ("WA", "TAS", "NT")
            if state == "TAS"
                prop_omicron = prop_pars[2]
            else
                prop_omicron = sigmoid(t_omicron, prop_pars)
            end
            # sample a random number of Omicron parents
            S_parents_omicron = rand(Binomial(S_parents, prop_omicron))
            A_parents_omicron = rand(Binomial(A_parents, prop_omicron))
            I_parents_omicron = rand(Binomial(I_parents, prop_omicron))
        end
        
        # put the number of parents in a tuple to iterate over 
        num_parents = (
            S_parents - S_parents_omicron, 
            A_parents - A_parents_omicron,
            I_parents - I_parents_omicron,
            S_parents_omicron, 
            A_parents_omicron,
            I_parents_omicron,
        )
        
        for (i, n_parents) in enumerate(num_parents)  
            # a guard clause for skipping the current iteration
            n_parents == 0 && continue 
            
            # reset counters
            num_offspring = 0
            TP_parent = 0.0 
            k = 0.0 
            
            # essentially a (gross) switch statement 
            if i == 1 
                α_s = forecast.sim_constants.α_s.delta
                TP_parent = TP.TP_local_delta_parent * α_s
                k = forecast.sim_constants.k.delta
            elseif i == 2
                α_a = forecast.sim_constants.α_a.delta
                TP_parent = TP.TP_local_delta_parent * α_a
                k = forecast.sim_constants.k.delta
            elseif i == 3
                TP_parent = TP.TP_import_delta_parent * import_multiplier
                k = forecast.sim_constants.k.delta
            elseif i == 4
                α_s = forecast.sim_constants.α_s.omicron
                TP_parent = TP.TP_local_omicron_parent * α_s
                k = forecast.sim_constants.k.omicron
            elseif i == 5
                α_a = forecast.sim_constants.α_a.omicron
                TP_parent = TP.TP_local_omicron_parent * α_a
                k = forecast.sim_constants.k.omicron
            elseif i == 6
                TP_parent = TP.TP_import_omicron_parent
                k = forecast.sim_constants.k.omicron
            end
                
            # if (adjust_TP && day >= T_observed - 29) || (!adjust_TP)
            # apply truncation for data, then apply the adjustment based in the forecasting
            if (adjust_TP && day >= T_observed + 7 - 14 - 30 - 1) || !adjust_TP
                # when using the Reff, we do not adjust the TP through susceptible 
                # depletion as this is already implicit in the calculation. 
                proportion_infected = min(1, total_infected / N)
                scale_factor = max(0, 1 - susceptible_depletion_sim * proportion_infected)
                TP_parent *= scale_factor
            end
            
            num_offspring += sample_negative_binomial_limit(TP_parent * n_parents, k * n_parents)
            
            if num_offspring > 0            
                p_symp = 0.0 
                p_detect_given_symp = 0.0 
                p_detect_given_asymp = 0.0 
                is_omicron = false
                
                # set the appropriate detection probabilities based on the point in the forecast
                if i in 1:3
                    p_symp = forecast.sim_constants.p_symp.delta 
                    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.delta 
                    p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp.delta  
                    is_omicron = false
                elseif i in 4:6
                    p_symp = forecast.sim_constants.p_symp.omicron 
                    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.omicron 
                    p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp.omicron  
                    is_omicron = true
                end
                
                num_S_offspring = sample_binomial_limit(num_offspring, p_symp)
                num_A_offspring = num_offspring - num_S_offspring
    
                num_S_offspring_detected = sample_binomial_limit(
                    num_S_offspring, p_detect_given_symp
                )
                num_A_offspring_detected = sample_binomial_limit(
                    num_A_offspring, p_detect_given_asymp
                )
                
                num_S_offspring_undetected = num_S_offspring - num_S_offspring_detected
                num_A_offspring_undetected = num_A_offspring - num_A_offspring_detected
                
                assign_to_arrays_and_times!(
                    forecast, 
                    day, 
                    is_omicron = is_omicron,
                    num_S_detected = num_S_offspring_detected, 
                    num_S_undetected = num_S_offspring_undetected, 
                    num_A_detected = num_A_offspring_detected, 
                    num_A_undetected = num_A_offspring_undetected, 
                )
            end
        end
    end

    return nothing 
    
end


function assign_to_arrays_and_times!(
    forecast::Forecast, 
    day;
    is_omicron = false, 
    num_S_detected = 0,
    num_S_undetected = 0,
    num_A_detected = 0,
    num_A_undetected = 0,
    num_I_detected = 0, 
    num_I_undetected = 0,       
)
    """
    Increment the arrays D and U based on sampled offsrping counts. The number of 
    each offspring are keyword parameters which are default 0. This allows us to call 
    this function from multiple spots. 
    """
    
    Z = forecast.sim_realisation.Z
    D = forecast.sim_realisation.D
    U = forecast.sim_realisation.U
    individual_type_map = forecast.individual_type_map
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day
    T_end = forecast.sim_features.T_end
    
    # sampling detected cases 
    for _ in 1:num_S_detected
        (inf_time, onset_time) = sample_times(day, is_omicron)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.S] += 1
        end
    
        if onset_time < T_end
            D[map_day_to_index_UD(onset_time), individual_type_map.S] += 1
        end
    end
    
    for _ in 1:num_A_detected
        (inf_time, onset_time) = sample_times(day, is_omicron)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.A] += 1
        end
        
        if onset_time < T_end
            D[map_day_to_index_UD(onset_time), individual_type_map.A] += 1
        end
    end
    
    # sampling undetected cases 
    for _ in 1:num_S_undetected
        (inf_time, onset_time) = sample_times(day, is_omicron)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.S] += 1
        end
        
        if onset_time < T_end
            U[map_day_to_index_UD(onset_time), individual_type_map.S] += 1
        end
    end
    
    for _ in 1:num_A_undetected
        (inf_time, onset_time) = sample_times(day, is_omicron)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.A] += 1
        end
        
        if onset_time < T_end
            U[map_day_to_index_UD(onset_time), individual_type_map.A] += 1
        end
    end
    
    # Sampling imports. These are treated slightly differently as the infection 
    # time is inferred through the import model during initialisation 
    for _ in 1:num_I_detected
        inf_time = day
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.I] += 1
        end
        
        onset_time = inf_time + sample_onset_time(is_omicron)
        if onset_time < T_end
            # time+1 as first index is day 0
            D[map_day_to_index_UD(onset_time), individual_type_map.I] += 1
        end
    end
    
    for _ in 1:num_I_undetected
        inf_time = day
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.I] += 1
        end
        
        onset_time = inf_time + sample_onset_time(is_omicron)
        if onset_time < T_end
            # time+1 as first index is day 0
            U[map_day_to_index_UD(onset_time), individual_type_map.I] += 1
        end
    end
    
    return nothing
    
end


function clean_up_sim_results(
    D_results,
    U_results, 
    good_sims, 
)
    """
    A simple function that takes in the simulation output, cleans it up and spits out the 
    processed format. We also account for the effect of susceptible depletion in the TP 
    here so that the output includes the adjusted TP. 
    """
    # TODO: Need to refactor the layout of this. Don't need to pass in so many parameters
    
    return (
        D_results[:, :, 1:good_sims - 1], 
        U_results[:, :, 1:good_sims - 1], 
    )
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
    omicron_start_date,
    omicron_dominant_date,
    state; 
    p_detect_omicron = 0.5,
    adjust_TP = false,
)
    """
    Simulate branching process nsims times conditional on the cumulative observed cases, 
    with initial state X0 = (S, A, I) (a named tuple). T is the duration of the simulation 
    and N is the population size which is currently unused. 
    """
    # read in the TPs for the state and strains 
    (TP_indices, TP_local_delta, TP_import_delta) = get_single_state_TP(
        forecast_start_date, 
        date, 
        state,
        p_detect_omicron = p_detect_omicron, 
        strain = "Delta",
        adjust_TP = adjust_TP,
    )
    
    (_, TP_local_omicron, TP_import_omicron) = get_single_state_TP(
        forecast_start_date, 
        date, 
        state,
        p_detect_omicron = p_detect_omicron, 
        strain = "Omicron",
        adjust_TP = adjust_TP,
    )
    
    susceptible_depletion = read_in_susceptible_depletion(
        date, 
        p_detect_omicron = p_detect_omicron,
    )
    
    omicron_prop_pars = read_in_prop_omicron(date, state, p_detect_omicron = p_detect_omicron)
     
    max_restarts = 10
    good_sims = ones(Bool, nsims)
    
    # get simulation constants 
    (sim_constants, individual_type_map) = set_simulation_constants(
        state, 
        p_detect_omicron = p_detect_omicron, 
    )
    
    # calculate the upper and lower limits for checking sims against data 
    sim_features = get_simulation_limits(
        local_cases, 
        forecast_start_date,
        omicron_start_date,
        omicron_dominant_date,
        cases_pre_forecast, 
        TP_indices, 
        N, 
        state, 
    )
    
    # the range of days for the simulation -1 as the "days", day ↦ day + 1
    day_range = -35:sim_features.T_end - 1
        
    # initialise state arrays (T_end + 1 as we include 0 day)
    sim_realisation = Realisation(sim_features.T_end)
    # always targeting 2000 sims 
    sim_results = Results(sim_features.T_end, 2000)
    
    # put everything into a forecast object
    forecast = Forecast(
        sim_features,
        sim_realisation,
        sim_results,
        sim_constants, 
        individual_type_map,
    )

    good_TPs_inds = zeros(Int, 2000)
    
    # Applying a reduction in NSW TP based off observations of the model fit. This issue is
    # at the beginning of the third wave so doesn't really influence the results for 
    # Omicron. I DON'T LIKE THIS BUT FOR NOW IT'S CORRECTING SOME ISSUE IN THE TP/REFF 
    # affecting NSW only. 
    if state == "NSW" || state == "VIC"
        TP_ind = findfirst(40 == ind for ind in TP_indices)
        TP_local_delta[1:TP_ind, :] *= 0.5
        TP_local_omicron[1:TP_ind, :] *= 0.5
    end
    
    # number of TP samples saved 
    num_TP = size(TP_local_delta, 2)
    
    # counts in each window are the same size as the limit arrays
    max_cases = forecast.sim_features.max_cases
    case_counts = zero(similar(max_cases))
    
    # a counter for the good sims, will terminate if good_sims > 2000
    good_sims = 1
    
    D_results = forecast.sim_results.D
    U_results = forecast.sim_results.U
    Z_historical_results = forecast.sim_results.Z_historical
    
    @showprogress for sim in 1:nsims
        # sample the TP/susceptible_depletion for this sim
        TP_ind = sim % num_TP == 0 ? num_TP : sim % num_TP
        susceptible_depletion_sim = susceptible_depletion[TP_ind]
    
        # reset boolean flags 
        bad_sim = false
        injected_cases = false
        
        start_day = initialise_population!(
            forecast,
            D0, 
        )
        
        import_cases_model!(
            forecast,
            import_cases, 
            forecast_start_date, 
        )
        
        # initialise counter to first day, when earliest infection occured based on 
        # initialisation
        day = start_day
        # find index of the first TP to use based on the indices representing time 
        # from forecast origin
        TP_start_day_ind = findfirst(start_day == d for d in TP_indices)
        TP_day_ind = TP_start_day_ind
        
        last_injection = 0
        n_restarts = 0

        while day <= day_range[end]
            # get TP's for the current day under the two strains and put in a named tuple
            TP_day = (
                TP_local_delta_parent = TP_local_delta[TP_day_ind, TP_ind],
                TP_import_delta_parent = TP_import_delta[TP_day_ind, TP_ind],
                TP_local_omicron_parent = TP_local_omicron[TP_day_ind, TP_ind],
                TP_import_omicron_parent = TP_import_omicron[TP_day_ind, TP_ind],
            )
            # get the sampled parameters on a given day
            prop_pars_day = (
                m0 = omicron_prop_pars["m0"][TP_ind], 
                m1 = omicron_prop_pars["m1"][TP_ind], 
                r = omicron_prop_pars["r"][TP_ind], 
                tau = omicron_prop_pars["tau"][TP_ind], 
            )
            
            sample_offspring!(
                forecast,
                day, 
                TP_day, 
                prop_pars_day,
                susceptible_depletion_sim,
                local_cases; 
                adjust_TP = adjust_TP
            )
            
            reinitialise_allowed = day >= 3 && 
                day < forecast.sim_features.T_observed - 3 && 
                n_restarts < max_restarts && 
                day - last_injection >= 3
            # reinitialise_allowed = day >= 7 && 
            #     day < forecast.sim_features.T_observed - 7 && 
            #     n_restarts < max_restarts 
                
            (bad_sim, injected_cases) = check_sim!(
                forecast,
                case_counts, 
                local_cases, 
                reinitialise_allowed;
                day = day,
                sim = sim,
            )
            
            if !injected_cases 
                day += 1
                TP_day_ind += 1
            else 
                n_restarts += 1
                if n_restarts > max_restarts 
                    bad_sim = true
                end
                last_injection = day
                day = start_day
                TP_day_ind = TP_start_day_ind
                injected_cases = false
            end
            
            bad_sim && break
            
        end
        
        if !bad_sim
            D_results[:, :, good_sims] .= forecast.sim_realisation.D
            U_results[:, :, good_sims] .= forecast.sim_realisation.U
            good_sims += 1
        end
        
        if good_sims > 2000 
            break
        end
        
    end
    
    println("##########")    
    println(state, " finished with ", good_sims - 1, " good simulations.")
    println("##########")    
    
    results = clean_up_sim_results(
        D_results,
        U_results,
        good_sims, 
    )
    
    return results
    
end