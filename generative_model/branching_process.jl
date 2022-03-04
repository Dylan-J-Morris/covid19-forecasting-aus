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
    # of 31/1/2022) but we will need to update this if we shift the starting date. 
    p_symp = forecast.sim_constants.p_symp.delta 
    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.delta 
    T_end = forecast.sim_features.T_end
    individual_type_map = forecast.individual_type_map
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 
    
    total_initial_cases = sum(D0)

    # infer some initial undetected symptomatic
    num_symptomatic_undetected = 0
    
    if D0.S == 0
        num_symptomatic_undetected = rand(NegativeBinomial(1, p_detect_given_symp))
    else
        num_symptomatic_undetected = rand(NegativeBinomial(D0.S, p_detect_given_symp))
    end

    # infer some initial undetected asymptomatic
    total_symptomatic = D0.S + num_symptomatic_undetected
    num_asymptomatic_undetected = 0
    
    if total_symptomatic == 0
        num_asymptomatic_undetected = rand(NegativeBinomial(1, p_symp))
    else
        num_asymptomatic_undetected = rand(NegativeBinomial(total_symptomatic, p_symp))
    end
    
    # total initial cases include the undetected
    total_initial_cases_sim = total_initial_cases + 
        num_symptomatic_undetected + 
        num_asymptomatic_undetected
    
    start_day = 0
    
    # reset arrays 
    D .= 0
    U .= 0
    Z .= 0
    Z_historical .= 0
        
    if total_initial_cases_sim > 0  
        inf_times = MVector{total_initial_cases_sim}(
            zeros(Int, total_initial_cases_sim)
        )
        
        # sample negative infection times as these individuals are assumed to occur 
        # before the simulation 
        for i in 1:total_initial_cases_sim
            inf_times[i] = -sample_onset_time()
        end
        
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
        
        for _ in 1:num_symptomatic_undetected
            Z[map_day_to_index_Z(inf_times[idx]), individual_type_map.S] += 1
            idx += 1
            U[1, individual_type_map.S] += 1
        end
        
        for _ in 1:num_asymptomatic_undetected
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
        if t < omicron_dominant_day - inc_period_shift
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
                num_imports_detected = D_I, 
                num_imports_undetected = U_I, 
            )
        end
    end
	
	return nothing
	
end

function calculate_proportion_infected(
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
    
    if adjust_TP  
        τ = map_day_to_index_cases(day)
        # if adjusting the TP then use the raw daily cases in the calculation of the 
        # depletion factor. We need to also scale the daily cases by the assumed 
        # CAR.
        cases_pre_omicron = 1 / p_detect_delta * sum(
            @view local_cases[
                begin:min(τ, map_day_to_index_cases(omicron_dominant_day) - 1)
            ]
        )
        
        if day >= omicron_dominant_day
            cases_after_omicron = 1 / p_detect_omicron * sum(
                @view local_cases[
                    map_day_to_index_cases(omicron_dominant_day):min(τ, length(local_cases))
                ]
            )
        end
        
        cases += cases_pre_omicron + cases_after_omicron
        
        if τ > length(local_cases)
            τ2 = min(map_day_to_index_Z(day), size(Z_historical, 1))
            cases += sum(@view Z_historical[map_day_to_index_Z(T_observed + 1):τ2])
        end
    else    
        τ = min(map_day_to_index_Z(day), size(Z_historical, 1))
        cases = sum(@view Z_historical[begin:τ])
    end
    
    return min(1, cases / N)

end


function sample_offspring!(
    forecast::Forecast, 
    t,  
    TP_local_sim, 
    TP_import_sim, 
    TP_indices, 
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
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 
    T_observed = forecast.sim_features.T_observed
    
    # initialise the parameters 
    α_s = 0.0 
    α_a = 0.0
    p_symp = 0.0 
    p_detect_given_symp = 0.0 
    p_detect_given_asymp = 0.0 
    k = 0.0 
    
    # grab the correct parameters for the particular dominant strain. The -5 is as we use
    # the infection dates in the simulation and onsets are considered when discussing 
    # case ascertainment
    if t < omicron_dominant_day
        α_s = forecast.sim_constants.α_s.delta
        α_a = forecast.sim_constants.α_a.delta
        p_symp = forecast.sim_constants.p_symp.delta 
        p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.delta 
        p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp.delta 
        k = forecast.sim_constants.k.delta 
    else
        α_s = forecast.sim_constants.α_s.omicron
        α_a = forecast.sim_constants.α_a.omicron
        p_symp = forecast.sim_constants.p_symp.omicron 
        p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.omicron 
        p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp.omicron 
        k = forecast.sim_constants.k.omicron 
    end
    
    # extract fixed simulation values
    cases_pre_forecast = forecast.sim_features.cases_pre_forecast
    N = forecast.sim_features.N
    
    # initialise number of offspring outside of the loop 
    num_offspring = zero(Int)
    
    # pointer to the current day's infections 
    Z_tmp = @view Z[map_day_to_index_Z(t), :]
    
    # if the total number of infected individuals is 0, return immediately
    if sum(Z_tmp) > 0 
        
        (S_parents, A_parents, I_parents) = Z_tmp
        
        # sum up the number local infections on the current day
        # Z_tmp gets emptied at the end of sampling so we won't be double counting 
        Z_historical[map_day_to_index_Z(t)] += S_parents + A_parents
        
        proportion_infected = calculate_proportion_infected(
            forecast, 
            local_cases, 
            t,  
            N; 
            adjust_TP = adjust_TP,
        )

        # find day based on the indices representing time from forecast origin
        TP_ind = findfirst(ind == t for ind in TP_indices)
            
        TP_local_parent = TP_local_sim[TP_ind]
        TP_import_parent = TP_import_sim[TP_ind]
        
        # if we are adjusting the TP, we use the Reff up until the last 30 days and then we
        # need to scale the TP afterwards. The Reff inherently features the depletion of 
        # susceptibles in its calculation (of sorts) and so can be considered the "true" 
        # factor 
        if (adjust_TP && t >= T_observed - 29) || (!adjust_TP)
            scale_factor = max(0, 1 - susceptible_depletion_sim * proportion_infected)
            TP_local_parent *= scale_factor
            TP_import_parent *= scale_factor    
        end
        
        # Use old model's import scaling pre-Omicron wave and then 
        # just use the inferred R_I for omicron after it's introduction. This latter 
        # estimate will implicitly include a vaccination effect as well as changes to the 
        # import restrictions. The simplified model assumes this is relatively fixed across 
        # the omicron wave.
        if t < omicron_dominant_day - 30
            # p_{v,h} is the proportion of hotel quarantine workers vaccinated
            p_vh = 0.9 + rand(Beta(2, 4)) * 9 / 100
            # v_{e,h} is the overall vaccine effectiveness
            v_eh = 0.83 + rand(Beta(2, 2)) * 14 / 100
            TP_import_parent *= (1 - p_vh * v_eh) * 1.39 * 1.3
        end
        
        # total number of infections arising from all parents infected at time t 
        # (sum of M NB(μ, ϕ) is NB(μ * M, ϕ * M))
        if S_parents > 0 && TP_local_parent > 0
            ϕ = k * S_parents 
            μ = α_s * TP_local_parent * S_parents
            num_offspring += sample_negative_binomial_limit(μ, ϕ)
        end
        
        if A_parents > 0 && TP_local_parent > 0
            ϕ = k * A_parents 
            μ = α_a * TP_local_parent * A_parents
            num_offspring += sample_negative_binomial_limit(μ, ϕ)
        end
        
        if I_parents > 0 && TP_import_parent > 0
            ϕ = k * I_parents
            μ = TP_import_parent * I_parents
            num_offspring += sample_negative_binomial_limit(μ, ϕ)
        end
    
        if num_offspring > 0
            num_symptomatic_offspring = sample_binomial_limit(num_offspring, p_symp)
            num_asymptomatic_offspring = num_offspring - num_symptomatic_offspring
            
            num_symptomatic_offspring_detected = sample_binomial_limit(
                num_symptomatic_offspring, p_detect_given_symp
            )
            num_asymptomatic_offspring_detected = sample_binomial_limit(
                num_asymptomatic_offspring, p_detect_given_asymp
            )
            
            num_symptomatic_offspring_undetected = num_symptomatic_offspring - 
                num_symptomatic_offspring_detected
            num_asymptomatic_offspring_undetected = num_asymptomatic_offspring - 
                num_asymptomatic_offspring_detected
            
            assign_to_arrays_and_times!(
                forecast, 
                t, 
                num_symptomatic_detected = num_symptomatic_offspring_detected, 
                num_symptomatic_undetected = num_symptomatic_offspring_undetected, 
                num_asymptomatic_detected = num_asymptomatic_offspring_detected, 
                num_asymptomatic_undetected = num_asymptomatic_offspring_undetected, 
            )
            
            # zero out infections on day t so we can restart sim and keep the progress 
            Z_tmp .= 0    
        end
    end

    return nothing 
    
end


function assign_to_arrays_and_times!(
    forecast::Forecast, 
    day;
    num_symptomatic_detected = 0,
    num_symptomatic_undetected = 0,
    num_asymptomatic_detected = 0,
    num_asymptomatic_undetected = 0,
    num_imports_detected = 0, 
    num_imports_undetected = 0,       
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
    for _ in 1:num_symptomatic_detected
        (inf_time, onset_time) = sample_times(day, omicron = day >= omicron_dominant_day)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.S] += 1
        end
    
        if onset_time < T_end
            D[map_day_to_index_UD(onset_time), individual_type_map.S] += 1
        end
    end
    
    for _ in 1:num_asymptomatic_detected
        (inf_time, onset_time) = sample_times(day, omicron = day >= omicron_dominant_day)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.A] += 1
        end
        
        if onset_time < T_end
            D[map_day_to_index_UD(onset_time), individual_type_map.A] += 1
        end
    end
    
    # sampling undetected cases 
    for _ in 1:num_symptomatic_undetected
        (inf_time, onset_time) = sample_times(day, omicron = day >= omicron_dominant_day)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.S] += 1
        end
        
        if onset_time < T_end
            U[map_day_to_index_UD(onset_time), individual_type_map.S] += 1
        end
    end
    
    for _ in 1:num_asymptomatic_undetected
        (inf_time, onset_time) = sample_times(day, omicron = day >= omicron_dominant_day)
        
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.A] += 1
        end
        
        if onset_time < T_end
            U[map_day_to_index_UD(onset_time), individual_type_map.A] += 1
        end
    end
    
    # Sampling imports. These are treated slightly differently as the infection 
    # time is inferred through the import model during initialisation 
    for _ in 1:num_imports_detected
        inf_time = day
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.I] += 1
        end
        
        onset_time = inf_time + 
            sample_onset_time(omicron = inf_time >= omicron_dominant_day)
        if onset_time < T_end
            # time+1 as first index is day 0
            D[map_day_to_index_UD(onset_time), individual_type_map.I] += 1
        end
    end
    
    for _ in 1:num_imports_undetected
        inf_time = day
        if inf_time < T_end 
            Z[map_day_to_index_Z(inf_time), individual_type_map.I] += 1
        end
        
        onset_time = inf_time +
            sample_onset_time(omicron = inf_time >= omicron_dominant_day)
        if onset_time < T_end
            # time+1 as first index is day 0
            U[map_day_to_index_UD(onset_time), individual_type_map.I] += 1
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
    state; 
    p_detect_omicron = 0.5,
    adjust_TP = false,
)
    """
    Simulate branching process nsims times conditional on the cumulative observed cases, 
    with initial state X0 = (S, A, I) (a named tuple). T is the duration of the simulation 
    and N is the population size which is currently unused. 
    """
    # read in the TP for a given state and date
    (TP_indices, TP_local, TP_import) = create_state_TP_matrices(
        forecast_start_date, 
        date, 
        state,
        p_detect_omicron = p_detect_omicron, 
        adjust_TP = adjust_TP,
    )
    
    # read in susceptible_depletion parameters 
    susceptible_depletion = read_in_susceptible_depletion(
        date, 
        p_detect_omicron = p_detect_omicron,
    )
     
    max_restarts = 10
    good_sims = ones(Bool, nsims)
    good_TPs = zeros(Bool, size(TP_local, 2))
    
    # get simulation constants 
    (sim_constants, individual_type_map) = set_simulation_constants(
        state, 
        p_detect_omicron = p_detect_omicron, 
    )
    
    # calculate the upper and lower limits for checking sims against data 
    sim_features = get_simulation_limits(
        local_cases, 
        forecast_start_date,
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
    # Omicron. 
    if state == "NSW"
        TP_ind = findfirst(40 == ind for ind in TP_indices)
        TP_local[1:TP_ind, :] *= 0.5
        TP_import[1:TP_ind, :] *= 0.5
    end
    
    # number of TP samples saved 
    num_TP = size(TP_local, 2)
    
    # counts in each window are the same size as the comparison arrays
    max_cases = forecast.sim_features.max_cases
    case_counts = zero(similar(max_cases))
    
    # a counter for the good sims, will terminate if good_sim_counter > 2000
    good_sims = 1
    
    D_results = forecast.sim_results.D
    U_results = forecast.sim_results.U
    Z_historical_results = forecast.sim_results.Z_historical
    
    @showprogress for sim in 1:nsims
        # sample the TP/susceptible_depletion for this sim
        TP_ind = sim % num_TP == 0 ? num_TP : sim % num_TP
        TP_local_sim = @view TP_local[:, TP_ind]
        TP_import_sim = @view TP_import[:, TP_ind]
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
                local_cases; 
                adjust_TP = adjust_TP
            )
            
            if day > 0
                reinitialise_allowed =  day >= 3 && 
                    day < forecast.sim_features.T_observed && 
                    n_restarts < max_restarts && 
                    day - last_injection >= 3
                
                (bad_sim, injected_cases) = check_sim!(
                    forecast,
                    forecast_start_date,
                    omicron_dominant_date,
                    case_counts, 
                    local_cases, 
                    reinitialise_allowed;
                    day = day,
                )
            end
            
            if !injected_cases 
                day += 1
            else 
                last_injection = day
                day = start_day
                n_restarts += 1
                injected_cases = false
                
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
                local_cases, 
                reinitialise_allowed,
            )
        end
        
        if !bad_sim 
            good_TPs_inds[good_sims] = TP_ind
            D_results[:, :, good_sims] .= forecast.sim_realisation.D
            U_results[:, :, good_sims] .= forecast.sim_realisation.U
            Z_historical_results[:, good_sims] .= forecast.sim_realisation.Z_historical
            good_sims += 1
        end
        
        if good_sims > 2000 
            break
        end
        
    end
    
    println("##########")    
    println(state, " finished with ", good_sims - 1, " good simulations.")
    println("##########")    
    
    # truncate the TP to the same period as per D and U
    TP_local_truncated = TP_local[(end - forecast.sim_features.T_end):end, :]
    # calculate the cumulative number of infections by day t
    Z_historical_cumulative = cumsum(Z_historical_results, dims = 1)
    
    Z_historical_cumulative = Z_historical_cumulative[
        end - size(TP_local_truncated, 1) + 1:end, 
        :,
    ]
    
    prop_infected = Z_historical_cumulative ./ N
    
    # new array to hold the correctly sampled TP samples 
    TP_local_sims = zeros(Float64, size(TP_local_truncated, 1), length(good_TPs_inds))
    # factor for adjusting the local TP by accounting for susceptible depletion
    scale_factor = ones(Float64, size(TP_local_sims))
    
    for (i, TP_ind) in enumerate(good_TPs_inds[1:good_sims - 1])
        scale_factor[:, i] = max.(
            0, 1 .- susceptible_depletion[TP_ind] * prop_infected[:, i]
        )
        if adjust_TP
            # if we're adjusting the TP, then we should use the actual reduction factor
            # before the switch over to the TP model 
            scale_factor[1:forecast.sim_features.T_observed - 30, i] .= 1
        end
        TP_local_sims[:, i] = TP_local_truncated[:, TP_ind] .* scale_factor[:, i]
    end
    
    return (
        D_results[:, :, 1:good_sims - 1], 
        U_results[:, :, 1:good_sims - 1], 
        TP_local_sims[:, 1:good_sims - 1], 
        scale_factor[:, 1:good_sims - 1], 
        Z_historical_results[:, 1:good_sims - 1],
    )
    
end