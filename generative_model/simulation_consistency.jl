include("forecast_types.jl")
include("helper_functions.jl")

function ρ(X, X_sim; metric = "absolute")
    """
    Calculate the sum of the absolute difference at each time point between observed data 
    X and simulated data X_sim noting that each row of these corresponds to a realisation. 
    that 
    """
    
    ϵ = zeros(size(X, 2))
    
    if metric == "absolute"
        ϵ .= sum(abs.(X - X_sim), dims = 1)[:]
    elseif metric == "MSE"
        ϵ .= sum((X - X_sim) .^ 2, dims = 1)[:]
    elseif metric == "RMSE"
        N = size(X, 1)
        ϵ .= sqrt.(1 / N * sum((X - X_sim) .^ 2, dims = 1)[:])
    end
    
    return ϵ
    
end

function final_check(D_good, local_cases; p = 0.1, metric = "absolute")
    """
    Determines the top p×100% of simulations based on some metric ρ(X, X_sim) and returns 
    those indices.  
    """
    # get only the observed cases over the period we have cases for
    D_observed = D_good[begin:length(local_cases), 1, :] + 
        D_good[begin:length(local_cases), 2, :]
    # tile local_cases to easily calculate the distance ρ(X, X_sim)
    local_cases_mat = repeat(local_cases, 1, size(D_observed, 2))
    
    # calculate distance between data and the simulations
    ϵ = ρ(local_cases_mat, D_observed, metric = metric)
    # determine the sim indices of the top p×100% 
    good_inds = ϵ .< quantile(ϵ, p)
    
    return good_inds 
    
end


# function calculate_bounds(local_cases, τ, state)
    
#     # tolerance for when we consider cases to be stable
#     ϵ = 0.0
    
#     # multipliers on the n-day average 
#     (ℓ, u) = (0.5, 1.75)
    
#     # observation period 
#     T = length(local_cases) - 1
#     # the slope over the n-day period 
#     m = 0.0
#     Mₜ = zeros(Float64, T)
    
#     Lₜ = zeros(Float64, T)
#     Uₜ = zeros(Float64, T) 
    
#     # consider τ = 3 and t = (0, 1, 2), clearly n = 2 - 0 = 2
#     n = τ - 1
    
#     n2 = 7
    
#     for t in 1:T
        
#         # if state == "NSW"  
#         #     if t < 30
#         #         (ℓ, u) = (0, 10)
#         #     elseif t < T - 7
#         #         (ℓ, u) = (0.5, 2.0)
#         #     else
#         #         (ℓ, u) = (0.5, 2.0)
#         #     end
#         # elseif state == "NT"
#         #     if t < 30
#         #         (ℓ, u) = (0, 4.0)
#         #     elseif t < T - 7
#         #         (ℓ, u) = (0.25, 4.0)
#         #     else
#         #         (ℓ, u) = (0.5, 3.0)
#         #     end
#         # end
        
#         # approximate the slope naively 
#         if t < T - n
#             m = 1 / n * (
#                 local_cases[map_day_to_index_UD(t) + n] - 
#                 local_cases[map_day_to_index_UD(t)]
#             )
#         else
#             m = 1 / n * (
#                 local_cases[map_day_to_index_UD(t)] - 
#                 local_cases[map_day_to_index_UD(t) - n]
#             ) 
#         end
        
#         Mₜ[t] = m
        
#         # depending on the sign of the slope, take average of future or past cases 
#         if m < -ϵ
#             Lₜ[t] = ℓ * mean(
#                 local_cases[map_day_to_index_UD(t):min(T, map_day_to_index_UD(t) + n)]
#             ) 
#             Uₜ[t] = u * mean(
#                 local_cases[max(1, map_day_to_index_UD(t) - n):map_day_to_index_UD(t)]
#             ) 
#             # Lₜ[t] = mean(local_cases[t:min(T, t + n)]) 
#             # Uₜ[t] = mean(local_cases[max(1, t - n):t]) 
#         elseif m > ϵ
#             Lₜ[t] = ℓ * mean(
#                 local_cases[max(1, map_day_to_index_UD(t) - n):map_day_to_index_UD(t)]
#             ) 
#             Uₜ[t] = u * mean(
#                 local_cases[map_day_to_index_UD(t):min(T, map_day_to_index_UD(t) + n)]
#             ) 
#             # Lₜ[t] = mean(local_cases[max(1, t - 2):t]) 
#             # Uₜ[t] = mean(local_cases[t:min(T, t + 2)]) 
#         else
#             n2 = n ÷ 2
#             Lₜ[t] = ℓ * mean(
#                 local_cases[
#                     max(1, map_day_to_index_UD(t) - n2):min(T, map_day_to_index_UD(t) + n2)
#                 ]
#             ) 
#             Uₜ[t] = u * mean(
#                 local_cases[
#                     max(1, map_day_to_index_UD(t) - n2):min(T, map_day_to_index_UD(t) + n2)
#                 ]
#             ) 
#             # Lₜ[t] = mean(local_cases[max(1, t - n2):min(T, t + n2)]) 
#             # Uₜ[t] = mean(local_cases[max(1, t - n2):min(T, t + n2)]) 
#         end
        
#         # adjust the bounds for periods with low cases
#         if Lₜ[t] < 50
#             Lₜ[t] = 0
#         end
        
#         if Uₜ[t] < 50
#             Uₜ[t] = 50
#         end
        
#         if isnan(Lₜ[t])
#             Lₜ[t] = Lₜ[t-1]
#         end
        
#         if isnan(Uₜ[t])
#             Uₜ[t] = Uₜ[t-1]
#         end
#     end
    
#     return (Lₜ, Uₜ) 
    
# end


function calculate_bounds(local_cases, τ, state)
    
    # tolerance for when we consider cases to be stable
    ϵ = 0.0
    
    # multipliers on the n-day average
    (ℓ, u) = (0.0, 2.5)
    # (ℓ, u) = (1.0, 1.0)
    
    # observation period 
    T = length(local_cases) - 1
    # the slope over the n-day period 
    m = 0.0
    
    L = (T ÷ 7) + 1
    Mₜ = zeros(Float64, L)
    Cₜ = zeros(Float64, L)
    
    Lₜ = zeros(Float64, L)
    Uₜ = zeros(Float64, L) 
    
    # consider τ = 3 and t = (0, 1, 2), clearly n = 2 - 0 = 2
    n = τ - 1
    
    n2 = 7
    
    # indexer for the bound arrays 
    s = 1
    
    for t in 1:T
        
        if mod(t - 1, τ) == 0 
            
            Cₜ[s] = sum(
                local_cases[t:min(T, t + n)]
            ) 
            
            Lₜ[s] = ℓ * Cₜ[s]
            Uₜ[s] = u * Cₜ[s]
            
            # adjust the bounds for periods with low cases
            if Lₜ[s] < 100
                Lₜ[s] = 0
            end
            
            if Uₜ[s] < 100
                Uₜ[s] = 100
            end
            
            if isnan(Lₜ[s])
                Lₜ[s] = Lₜ[s - 1]
            end
            
            if isnan(Uₜ[s])
                Uₜ[s] = Uₜ[s - 1]
            end
            
            s += 1
            
        end
        
    end
    
    # return (Lₜ, Uₜ, Cₜ) 
    return (Lₜ, Uₜ) 
    
end

# function calculate_bounds_choice(local_cases, τ, state; type = 1)
    
#     Lₜ = []
#     Uₜ = []
    
#     if type == 1
#         (Lₜ, Uₜ) = calculate_bounds_weekly(local_cases, τ, state)
#     elseif type == 2
        
#     end
    
# end


function get_simulation_limits(
    local_cases, 
    forecast_start_date,
    omicron_dominant_date,
    cases_pre_forecast, 
    TP_indices, 
    N, 
    state, 
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
    
    # this is the number of days into the forecast simulation that dominant begins 
    days_delta = (Dates.Date(omicron_dominant_date) - 
        Dates.Date(forecast_start_date)).value
        
    (min_cases, max_cases) = calculate_bounds(local_cases, 7, state)
    
    # cases_each_period = [
    #     sum(local_cases[begin:days_delta]),
    #     # sum(local_cases[max(1, T_observed - 90):end]),
    #     sum(local_cases[max(1, T_observed - 60):end]),
    #     sum(local_cases[max(1, T_observed - 14):end]),
    # ] 
    
    # min_cases = 0.25 * cases_each_period
    # max_cases = 4.0 * cases_each_period
    # min_cases[min_cases .< 50] .= 0
    # max_cases[max_cases .< 50] .= 50
    
    # the maximum allowable cases over the forecast period is the population size
    max_forecast_cases = N
    
    # get the day we want to start using omicron GI and incubation period (+1) as 0 
    # corresponds to the first element of the arrays 
    omicron_dominant_day = (omicron_dominant_date - forecast_start_date).value
    
    sim_features = Features(
        max_forecast_cases,
        cases_pre_forecast, 
        N,
        T_observed, 
        T_end, 
        omicron_dominant_day, 
    )
    
    return (
        sim_features,
        min_cases, 
        max_cases,
    )
    
end


# function count_cases!(
#     case_counts, 
#     forecast::Forecast, 
#     sim,
# )

#     D = forecast.sim_realisations.D
    
#     # case_counts_tmp = deepcopy(case_counts)
#     day = 1
#     tmp = 0
    
#     for i in 1:length(case_counts)
#         case_counts[day] = D[map_day_to_index_UD(i), 1, sim] + 
#             D[map_day_to_index_UD(i), 2, sim]
#         day += 1
#     end
    
#     # moving_average!(case_counts, case_counts_tmp, 3)
#     # case_counts .= case_counts_tmp
    
#     return nothing
    
# end


function count_cases!(
    case_counts, 
    forecast::Forecast, 
    sim,
)

    D = forecast.sim_realisations.D
    T_observed = forecast.sim_features.T_observed
    
    # case_counts_tmp = deepcopy(case_counts)
    day_index = 1
    i = 1
    tmp = 0
    case_counts .= 0
    
    for t in 1:T_observed
        tmp += D[t, 1, sim] + D[t, 2, sim]
        if mod(t, 7) == 0
            case_counts[i] = tmp
            i += 1
            tmp = 0;
        end
        i > length(case_counts) && break
    end
    
    # moving_average!(case_counts, case_counts_tmp, 3)
    # case_counts .= case_counts_tmp
    
    return nothing
    
end


function check_sim!(
    forecast::Forecast, 
    forecast_start_date,
    omicron_dominant_date,
    case_counts, 
    sim,
    local_cases,
    min_cases, 
    max_cases, 
    reinitialise_allowed;
    day = 0,
    state = nothing,
)
    """
    Check for consistency of the simulations against the data. This will also 
    check for instances of superspreading events and will insert cases if conditions
    are met.
    """
    
    print_status = false
    
    Z = forecast.sim_realisations.Z
    D = forecast.sim_realisations.D
    U = forecast.sim_realisations.U
    
    # days forecast observed for 
    T_observed = forecast.sim_features.T_observed
    max_forecast_cases = forecast.sim_features.max_forecast_cases
    consistency_multiplier = forecast.sim_constants.consistency_multiplier
    
    # initialise the bad sim
    bad_sim = false
    injected_cases = false 
    
    days_delta = (
        Dates.Date(omicron_dominant_date) - 
        Dates.Date(forecast_start_date)
    ).value
    
    # this is just the total cases over the forecast horizon 
    D_forecast = sum(@view D[T_observed + 1:end, 1:2, sim])
    # count how many cases each day 
    count_cases!(case_counts, forecast, sim)
    
    # sum all cases over observation period 
    # case_counts = [
    #     sum(D[begin:days_delta, 1:2, sim]),
    #     # sum(D[max(1, T_observed - 90):T_observed, 1:2, sim]),
    #     sum(D[max(1, T_observed - 60):T_observed, 1:2, sim]),
    #     sum(D[max(1, T_observed - 14):T_observed, 1:2, sim])
    # ]
    
    # if we've exceeded the max cases over a given period
    if D_forecast > max_forecast_cases
        bad_sim = true
        # bad_sim && println("sim: ", sim, " had too many forecast cases")
        print_status && println("too many forecast cases")
    end
    
    if !bad_sim 
        if any(case_counts .> max_cases) 
            bad_sim = true 
        end
        
        # for (i, (c, m)) in enumerate(
        #     zip(case_counts, max_cases)
        # )
        #     if c > m 
        #         bad_sim = true 
        #         # bad_sim && println("sim: ", sim, " had too many cases: ", c, " > ", m)
        #         print_status && println("sim: ", sim, " too many cases: ", c, " > ", m, " day: ", day)
        #         break
        #     end
        # end
        
        if day == 0 
            if any(case_counts .< min_cases)
                bad_sim = true
            end
            
            # for (i, (c, m)) in enumerate(
            #     zip(case_counts, min_cases)
            # )
            #     if c < m 
            #         bad_sim = true 
            #         print_status && println("sim: ", sim, " too few cases: ", c, " < ", m, " day: ", day)
            #         break
            #     end
            # end
        end
    end
    
    # see whether we need to inject any cases
    injected_cases = false
    
    if !bad_sim && reinitialise_allowed
        # calculate the number of detections missed over the 5 day period 
        # actual_3_day_cases = sum(@view local_cases[day-2:day])
        observed_3_day_cases = sum(
            @view local_cases[map_day_to_index_UD(day)]
        )
        # sim_7_day_cases = sum(@view D[day-6:day,:,sim])
        # sim__day_threshold = consistency_multiplier*max(1, sim_3_day_cases)
        # calculate total number of cases over week 
        sim_3_day_cases = sum(@view D[map_day_to_index_UD(day), 1:2, sim])
        missing_detections = 0
        threshold = max(1, ceil(Int, consistency_multiplier * sim_3_day_cases))
        
        if (observed_3_day_cases > threshold)
            # print_status && println(
            #     " sim: ", sim,  
            #     " sim cases: ", sim_3_day_cases,
            #     " actual cases: ", observed_3_day_cases, 
            #     " day added: ", day,
            # )
            # uniformly sample a number of missing detections to add in
            missing_detections = ceil(
                Int, 
                observed_3_day_cases - sim_3_day_cases,
            )
            
            # println("missing detections: ", missing_detections)
            
            # missing_detections_injected = max(1, rand(1:missing_detections))
            
            injected_cases = true 
            inject_cases!(
                forecast::Forecast, 
                missing_detections, 
                day, 
                sim, 
            )
        end
    end
    
    return (bad_sim, injected_cases)
    
end


function inject_cases!(
    forecast::Forecast,
    missing_detections, 
    day, 
    sim,
)
    """
    A model for injecting cases into the simulation following long periods of low 
    incidence. 
    """
    
    Z = forecast.sim_realisations.Z
    D = forecast.sim_realisations.D
    U = forecast.sim_realisations.U
    individual_type_map = forecast.individual_type_map
    
    total_injected = 0
    
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 
    # initialise the parameters 
    p_symp = 0.0 
    p_detect_given_symp = 0.0 
    p_detect_given_asymp = 0.0 
    p_symp_given_detect = 0.0 
    
    # grab the correct parameters for the particular dominant strain 
    if day < omicron_dominant_day 
        p_symp = forecast.sim_constants.p_symp.delta 
        p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.delta 
        p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp.delta 
        p_symp_given_detect = forecast.sim_constants.p_symp_given_detect.delta 
    else
        p_symp = forecast.sim_constants.p_symp.omicron 
        p_detect_given_symp = forecast.sim_constants.p_detect_given_symp.omicron 
        p_detect_given_asymp = forecast.sim_constants.p_detect_given_asymp.omicron 
        p_symp_given_detect = forecast.sim_constants.p_symp_given_detect.omicron 
    end
    
    # sample number detected 
    num_symptomatic_detected = sample_binomial_limit(
        missing_detections, p_symp_given_detect
    )
    num_asymptomatic_detected = missing_detections - num_symptomatic_detected
    
    # println("detections added: ", num_symptomatic_detected + num_asymptomatic_detected)
    
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
        num_asymptomatic_undetected = sample_negative_binomial_limit(
            num_symptomatic_total, p_symp
        )
    end
    
    # detection times are determined triangularly 
    num_symptomatic_each_day = (
        num_symptomatic_detected ÷ 2, 
        num_symptomatic_detected ÷ 2 + 
        num_symptomatic_detected ÷ 3,
        num_symptomatic_detected ÷ 2 + 
        num_symptomatic_detected ÷ 3 + 
        num_symptomatic_detected ÷ 6,
    )
    counter = 0
    onset_time = day
    
    # infection time is then 1 generation interval earlier
    for _ in 1:num_symptomatic_detected
        total_injected += 1
        # if counter >= num_symptomatic_each_day[2]
        #     onset_time = day - 1
        # elseif counter >= num_symptomatic_each_day[3]
        #     onset_time = day - 2
        # end
        inf_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day) - 1
        Z[map_day_to_index_Z(inf_time), individual_type_map.S, sim] += 1
        D[map_day_to_index_UD(onset_time), individual_type_map.S, sim] += 1 
        counter += 1
    end
    
    # detection times are determined triangularly 
    num_asymptomatic_each_day = (
        num_asymptomatic_detected ÷ 2, 
        num_asymptomatic_detected ÷ 2 + 
        num_asymptomatic_detected ÷ 3,
        num_asymptomatic_detected ÷ 2 + 
        num_asymptomatic_detected ÷ 3 + 
        num_asymptomatic_detected ÷ 6, 
    )
    counter = 0
    onset_time = day
    
    # infection time is then 1 generation interval earlier
    for _ in 1:num_asymptomatic_detected
        total_injected += 1
        # if counter >= num_asymptomatic_each_day[2]
        #     onset_time = day - 1
        # elseif counter >= num_asymptomatic_each_day[3]
        #     onset_time = day - 2
        # end
        
        # -1 as the sampled onset time is ceiling and so technically the infection day 
        # would be rounded down 
        inf_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day) - 1
        Z[map_day_to_index_Z(inf_time), individual_type_map.A, sim] += 1
        D[map_day_to_index_UD(onset_time), individual_type_map.A, sim] += 1 
        counter += 1
    end
    
    # now add in the inferred undetected cases in the same way. Note that the 
    # onset times are kinda irrelevant here, but just help with consistent 
    # simulation recording.
    num_symptomatic_each_day = (
        num_symptomatic_undetected ÷ 2, 
        num_symptomatic_undetected ÷ 2 + 
        num_symptomatic_undetected ÷ 3,
        num_symptomatic_undetected ÷ 2 + 
        num_symptomatic_undetected ÷ 3 + 
        num_symptomatic_undetected ÷ 6,
    )
    counter = 0
    onset_time = day
    
    for _ in 1:num_symptomatic_undetected
        total_injected += 1
        # if counter >= num_symptomatic_each_day[2]
        #     onset_time = day - 1
        # elseif counter >= num_symptomatic_each_day[3]
        #     onset_time = day - 2
        # end
        inf_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day) - 1
        Z[map_day_to_index_Z(inf_time), individual_type_map.S, sim] += 1
        U[map_day_to_index_UD(onset_time), individual_type_map.S, sim] += 1 
        counter += 1
    end
    
    # detection times are determined triangularly 
    num_asymptomatic_each_day = (
        num_asymptomatic_undetected ÷ 2, 
        num_asymptomatic_undetected ÷ 2 + 
        num_asymptomatic_undetected ÷ 3,
        num_asymptomatic_undetected ÷ 2 + 
        num_asymptomatic_undetected ÷ 3 + 
        num_asymptomatic_undetected ÷ 6,
    )
    counter = 0
    onset_time = day
    
    # now add in the inferred undetected cases 
    for _ in 1:num_asymptomatic_undetected
        total_injected += 1
        # if counter >= num_asymptomatic_each_day[2]
        #     onset_time = day - 1
        # elseif counter >= num_asymptomatic_each_day[3]
        #     onset_time = day - 2
        # end
        inf_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day) - 1
        Z[map_day_to_index_Z(inf_time), individual_type_map.A, sim] += 1
        U[map_day_to_index_UD(onset_time), individual_type_map.A, sim] += 1 
        counter += 1
    end
    
    # println("total injected is: ", total_injected, " at time: ", day)
    
    return nothing 
    
end