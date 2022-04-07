include("forecast_types.jl")
include("helper_functions.jl")

function calculate_bounds(local_cases, τ, state)
    
    # observation period 
    T = length(local_cases)
    
    idxs_limits =  Vector{UnitRange{Int64}}()
    
    idxs_start = 1
    while idxs_start - 1 != length(local_cases)
        idxs_end = idxs_start + τ - 1 
        if length(local_cases) - idxs_end < τ
            idxs_end = length(local_cases)
        end
        push!(idxs_limits, idxs_start:idxs_end)
        idxs_start = idxs_end + 1
    end
    
    Cₜ = [sum(local_cases[idx]) for idx in idxs_limits]
    
    # multipliers on the n-day average
    (ℓ, u) = (0.25, 2.5)
    Lₜ = ceil.(Int, ℓ * Cₜ)
    Uₜ = ceil.(Int, u * Cₜ)
    
    # remove restrictions over last τ * 2 days 
    (ℓ, u) = (0.5, 2.0)
    Lₜ[end-1:end] = ceil.(Int, ℓ * Cₜ[end-1:end])
    Uₜ[end-1:end] = ceil.(Int, u * Cₜ[end-1:end])
    
    min_limit = τ * 50
    
    Lₜ[Lₜ .< min_limit] .= 0 
    Uₜ[Uₜ .< min_limit] .= min_limit 
    
    return (Lₜ, Uₜ, idxs_limits) 
    
end


function count_cases!(
    case_counts, 
    forecast::Forecast, 
)

    idxs_limits = forecast.sim_features.idxs_limits
    D = forecast.sim_realisation.D
    case_counts .= 0
    
    for (i, idx) in enumerate(idxs_limits)
        case_counts[i] = sum(@view D[idx, 1:2])
    end
    
    return nothing
    
end


function get_simulation_limits(
    local_cases, 
    forecast_start_date,
    cases_pre_forecast, 
    TP_indices, 
    N, 
    state; 
    τ = 5,
    data_truncation = 7, 
    nowcast_truncation = 30, 
    fitting_truncation = 14,
)
    """
    Using the observed cases, determine the limits of cases over the backcast and 
    nowcast. This assumes consistency over windows of fixed length and a nowcast period 
    of 14 days.
    """
    
    # length of observed time series 
    T_observed = length(local_cases)
    # duration of forecast simulation including initial day
    T_end = sum(ind >= 0 for ind in TP_indices)
        
    (min_cases, max_cases, idxs_limits) = calculate_bounds(local_cases, τ, state)
    
    # the maximum allowable cases over the forecast period is the population size
    max_forecast_cases = N ÷ 2
    
    reff_change_time = (
        T_observed + data_truncation - 1 - (fitting_truncation + nowcast_truncation)
    )
    
    omicron_start_date = "2021-11-15"
    omicron_start_day = Dates.value(
        Dates.Date(omicron_start_date) - Dates.Date(forecast_start_date)
    )
    
    sim_features = Features(
        max_forecast_cases,
        cases_pre_forecast, 
        N,
        T_observed, 
        T_end, 
        τ,
        min_cases, 
        max_cases, 
        idxs_limits,
        state,
        reff_change_time,
        omicron_start_day, 
    )
    
    return sim_features
    
end


function check_sim!(
    forecast::Forecast, 
    case_counts, 
    local_cases,
    reinitialise_allowed;
    day = 0,
    sim = 0, 
)
    """
    Check for consistency of the simulations against the data. This will also 
    check for instances of superspreading events and will insert cases if conditions
    are met.
    """
    
    # guard clause for if we shouldn't check the simulation
    day < 0 && return (false, false)
    
    print_status = false
    print_inject_status = false
    
    Z = forecast.sim_realisation.Z
    D = forecast.sim_realisation.D
    U = forecast.sim_realisation.U
    
    # days forecast observed for 
    T_observed = forecast.sim_features.T_observed
    T_end = forecast.sim_features.T_end
    τ = forecast.sim_features.τ
    max_forecast_cases = forecast.sim_features.max_forecast_cases
    consistency_multiplier = forecast.sim_constants.consistency_multiplier
    
    min_cases = forecast.sim_features.min_cases
    max_cases = forecast.sim_features.max_cases
    idxs_limits = forecast.sim_features.idxs_limits
    
    # initialise as bad sim
    bad_sim = false
    injected_cases = false 
    
    # if day == 0
    #     # count how many cases each day 0
    #     count_cases!(case_counts, forecast)
    #     if any(case_counts .< min_cases)
    #         bad_sim = true
    #         return (bad_sim, injected_cases)
    #     end
    # end
    
    # only check for consistency if we don't inject cases.
    ζ = ceil(map_day_to_index_cases(day) ÷ τ)
        
    if 1 <= ζ <= length(case_counts)
        # count how many cases each day 0
        count_cases!(case_counts, forecast)
        if any(case_counts[1:ζ] .> max_cases[1:ζ])
            if print_status 
                bad_id = case_counts[1:ζ] .> max_cases[1:ζ]
                println(
                    "Sim: ", sim, "\n",
                    "Too many cases: ", case_counts[1:ζ][bad_id], "\n", 
                    "Maximum: ", max_cases[1:ζ][bad_id], "\n", 
                    "Window: ", findfirst(bad_id), "\n",
                )                
            end
            bad_sim = true 
        end
        
        if !bad_sim && reinitialise_allowed
            # calculate the number of detections missed over the τ day period 
            X_day_range = min(T_observed - 1, day - τ + 1):min(T_observed - 1, day)
            actual_X_day_cases = sum(
                @view local_cases[map_day_to_index_cases.(X_day_range)]
            )
            # calculate total number of cases over week 
            sim_X_day_cases = sum(@view D[map_day_to_index_UD.(X_day_range), 1:2])
            missing_detections = 0
            threshold = ceil(Int, consistency_multiplier * max(1, sim_X_day_cases))
            
            if (actual_X_day_cases > threshold)
                if print_status && print_inject_status 
                    println(
                        "======\n", 
                        "Sim: ", sim, "\n",
                        "Injected cases...\n",
                        "Actual cases: ", actual_X_day_cases, "\n", 
                        "Sim cases: ", sim_X_day_cases, "\n",
                        "Day added: ", day, "\n",
                        "======",
                    )
                end
                
                # uniformly sample a number of missing detections to add in
                missing_detections = ceil(
                    Int, 
                    actual_X_day_cases - sim_X_day_cases,
                )
                
                injected_cases = true 
                
                inject_cases!(
                    forecast::Forecast, 
                    missing_detections, 
                    X_day_range, 
                )
            end
        end
        
        # count updated cases
        count_cases!(case_counts, forecast)
        
        if any(case_counts[1:ζ] .< min_cases[1:ζ])
            if print_status 
                bad_id = case_counts[1:ζ] .< min_cases[1:ζ]
                println(
                    "Sim: ", sim, "\n",
                    "Too few cases: ", case_counts[1:ζ][bad_id], "\n",
                    "Minimum:  ", min_cases[1:ζ][bad_id], "\n",
                    "Window: ", findfirst(bad_id), "\n",
                )                
            end
            bad_sim = true 
        end
        
    elseif ζ > length(case_counts)
        # this is just the total cases over the forecast horizon 
        D_forecast = sum(@view D[T_observed + 1:end, 1:2])
        # if we've exceeded the max cases over a given period
        if D_forecast > max_forecast_cases
            bad_sim = true
        end
    end
    
    return (bad_sim, injected_cases)
    
end


function inject_cases!(
    forecast::Forecast,
    missing_detections, 
    X_day_range, 
)
    """
    A model for injecting cases into the simulation following long periods of low 
    incidence. 
    """
    
    Z = forecast.sim_realisation.Z
    D = forecast.sim_realisation.D
    U = forecast.sim_realisation.U
    individual_type_map = forecast.individual_type_map
    
    total_injected = 0
    # assume that by mid December the adjusted GI and IP are noticeable for the case insertion. 
    # small approximation but should be Ok overall.
    omicron_dominant_day = forecast.sim_features.omicron_start_day + 30
    # initialise the parameters 
    p_symp = 0.0 
    p_detect_given_symp = 0.0 
    p_detect_given_asymp = 0.0 
    p_symp_given_detect = 0.0 
    
    ind = map_day_to_index_p(X_day_range[end])
    
    p_symp = forecast.sim_constants.p_symp[ind]
    p_detect_given_symp = forecast.sim_constants.p_detect_given_symp[ind]
    p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp[ind]
    p_symp_given_detect = forecast.sim_constants.p_symp_given_detect[ind]
    
    # sample number detected 
    num_symptomatic_detected = sample_binomial_limit(missing_detections, p_symp_given_detect)
    num_asymptomatic_detected = missing_detections - num_symptomatic_detected
    
    # infer some undetected symptomatic
    num_symptomatic_undetected = sample_negative_binomial_limit(
        missing_detections, p_detect_given_symp
    )
    
    # infer some undetected asumptomatic
    num_symptomatic_total = num_symptomatic_detected + num_symptomatic_undetected
    
    num_asymptomatic_undetected = 0
    if num_symptomatic_total == 0
        num_asymptomatic_undetected = sample_negative_binomial_limit(1, p_symp)
    else
        num_asymptomatic_undetected = sample_negative_binomial_limit(num_symptomatic_total, p_symp)
    end
    
    # infection time is then 1 generation interval earlier
    for _ in 1:num_symptomatic_detected
        total_injected += 1
        onset_time = rand(X_day_range)
        inf_time = onset_time - sample_onset_time(onset_time >= omicron_dominant_day)
        Z[map_day_to_index_Z(inf_time), individual_type_map.S] += 1
        D[map_day_to_index_UD(onset_time), individual_type_map.S] += 1 
    end
    
    # infection time is then 1 generation interval earlier
    for _ in 1:num_asymptomatic_detected
        total_injected += 1
        onset_time = rand(X_day_range)
        # -1 as the sampled onset time is ceiling and so technically the infection day 
        # would be rounded down 
        inf_time = onset_time - sample_onset_time(onset_time >= omicron_dominant_day)
        Z[map_day_to_index_Z(inf_time), individual_type_map.A] += 1
        D[map_day_to_index_UD(onset_time), individual_type_map.A] += 1 
    end
    
    # now add in the inferred undetected cases in the same way. Note that the 
    # onset times are kinda irrelevant here, but just help with consistent 
    for _ in 1:num_symptomatic_undetected
        total_injected += 1
        onset_time = rand(X_day_range)
        inf_time = onset_time - sample_onset_time(onset_time >= omicron_dominant_day)
        Z[map_day_to_index_Z(inf_time), individual_type_map.S] += 1
        U[map_day_to_index_UD(onset_time), individual_type_map.S] += 1 
    end
    
    # now add in the inferred undetected cases 
    for _ in 1:num_asymptomatic_undetected
        total_injected += 1
        onset_time = rand(X_day_range)
        inf_time = onset_time - sample_onset_time(onset_time >= omicron_dominant_day)
        Z[map_day_to_index_Z(inf_time), individual_type_map.A] += 1
        U[map_day_to_index_UD(onset_time), individual_type_map.A] += 1 
    end
    
    return nothing 
    
end