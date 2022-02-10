include("forecast_types.jl")
include("helper_functions.jl")

function get_simulation_limits(
    local_cases, 
    forecast_start_date,
    omicron_dominant_date,
    cases_pre_forecast, 
    TP_indices, 
    N, 
    T_observed, 
    T_end,
)
    """
    Using the observed cases, determine the limits of cases over the backcast and 
    nowcast. This assumes consistency over windows of fixed length and a nowcast period 
    of 14 days.
    """
    # this is the number of days into the forecast simulation that dominant begins 
    days_delta = (Dates.Date(omicron_dominant_date) - 
        Dates.Date(forecast_start_date)).value
        
    # calculate the cases over the various windows
    cases_pre_backcast = sum(@view local_cases[1:days_delta])
    cases_backcast = sum(@view local_cases[days_delta+1:T_observed])
    cases_60 = sum(@view local_cases[60:T_observed])
    cases_nowcast = sum(@view local_cases[14:T_observed])
    # cases_nowcast = sum(@view local_cases[7:T_observed])
    
    # calculate minimum and maximum observed cases in each period 
    min_cases = floor.(
        Int, 
        [0.3*cases_pre_backcast, 0.5*cases_backcast, 0.5*cases_60, 0.7*cases_nowcast]
    )
    # min_cases = 0*cases_in_each_window
    max_cases = ceil.(
        Int, 
        [2.5*cases_pre_backcast, 3.0*cases_backcast, 2.5*cases_60, 2.5*cases_nowcast]
    )

    # assume maximum of 250 cases if the observed is less than that
    for (i, val) in enumerate(max_cases)
        if val < 100
            max_cases[i] = 100
        end
    end

    for (i, val) in enumerate(min_cases)
        if val < 50
            min_cases[i] = 0
        end
    end
    
    # the maximum allowable cases over the forecast period is the population size
    max_forecast_cases = N
    
    # get the day we want to start using omicron GI and incubation period (+1) as 0 
    # corresponds to the first element of the arrays 
    omicron_dominant_day = (omicron_dominant_date - forecast_start_date).value + 1
    
    sim_features = Features(
        max_forecast_cases,
        cases_pre_forecast, 
        N,
        T_observed, 
        T_end, 
        omicron_dominant_day, 
    )
    
    window_lengths = 0
    
    return (
        sim_features,
        window_lengths,
        min_cases, 
        max_cases,
    )
    
end


function count_cases_in_windows!(
    forecast::Forecast, 
    case_counts,  
    window_lengths, 
    D_total, 
    sim
)
    """
    Count the cases in each observation period and store in case_counts. 
    """
    
    t_start = 1
    for (i,t) in enumerate(window_lengths)
        case_counts[i] = sum(@view D_total[t_start:t,sim])
        t_start = t
    end
    
    return nothing
    
end


function check_sim!(
    forecast::Forecast, 
    forecast_start_date,
    omicron_dominant_date,
    case_counts, 
    sim,
    local_cases,
    window_lengths, 
    min_cases, 
    max_cases, 
    reinitialise_allowed;
    day = 0,
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
    
    days_delta = (Dates.Date(omicron_dominant_date) - 
        Dates.Date(forecast_start_date)).value
    
    cases_pre_backcast = sum(@view D[1:days_delta,1:2,sim])
    cases_backcast = sum(@view D[days_delta+1:T_observed,1:2,sim])
    cases_60 = sum(@view D[60:T_observed,1:2,sim])
    cases_nowcast = sum(@view D[14:T_observed,1:2,sim])
    
    case_counts[1] = cases_pre_backcast
    case_counts[2] = cases_backcast
    case_counts[3] = cases_60
    case_counts[end] = cases_nowcast
    
    # this is just the total cases over the forecast horizon 
    D_forecast = sum(@view D[T_observed-7+1:end,1:2,sim])
    
    # if we've exceeded the max cases over a given period
    if D_forecast > max_forecast_cases
        bad_sim = true
        print_status && println("too many forecast cases")
    end
    
    if !bad_sim 
        for (i, (c,m)) in enumerate(zip(case_counts,max_cases))
            if c > m 
                bad_sim = true 
                print_status && println("too many cases", c, " ", m, " day: ", day)
                break
            end
        end
        
        if day == 0
            for (i, (c,m)) in enumerate(zip(case_counts,min_cases))
                if c < m 
                    bad_sim = true 
                    print_status && println("too few cases", c, " ", m, " day: ", day)
                    break
                end
            end
        end
    end
    
    # see whether we need to inject any cases
    injected_cases = false
    
    if !bad_sim && reinitialise_allowed
        # calculate the number of detections missed over the 5 day period 
        actual_3_day_cases = sum(@view local_cases[day-2:day])
        sim_3_day_cases = sum(@view D[day-2:day,:,sim])
        sim_3_day_threshold = consistency_multiplier*max(1, sim_3_day_cases)
        # calculate total number of cases over week 
        sim_week_cases = sum(@view D[day-6:day,:,sim])
        missing_detections = 0
        
        if (actual_3_day_cases > sim_3_day_threshold) || 
            (sim_week_cases == 0 && actual_3_day_cases > 0)
            print_status && println(
                "actual: ", actual_3_day_cases, 
                " 3day: ", sim_3_day_cases, 
                " 3day thresh: ", sim_3_day_threshold, 
                " week: ", sim_week_cases, 
                " day added: ", day,
            )
             
            if sim_week_cases == 0
                missing_detections = rand(1:actual_3_day_cases)
            else
                missing_detections = rand(1:(actual_3_day_cases - sim_3_day_cases))
            end
            
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
    
    total_injected = missing_detections
    
    omicron_dominant_day = forecast.sim_features.omicron_dominant_day 
    # initialise the parameters 
    p_symp = 0.0 
    p_detect_given_symp = 0.0 
    p_detect_given_asymp = 0.0 
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
    # infer some undetected symptomatic
    num_symptomatic_undetected = sample_negative_binomial_limit(
        missing_detections, p_detect_given_symp
    )
    
    # infer some undetected asumptomatic
    num_symptomatic_total = num_symptomatic_detected + num_symptomatic_undetected
    
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
        if counter >= num_symptomatic_each_day[2]
            onset_time = day-1
        elseif counter >= num_symptomatic_each_day[3]
            onset_time = day-2
        end
        infection_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day)
        Z[infection_time+36,individual_type_map.S,sim] += 1
        D[onset_time,individual_type_map.S,sim] += 1 
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
        if counter >= num_asymptomatic_each_day[2]
            onset_time = day-1
        elseif counter >= num_asymptomatic_each_day[3]
            onset_time = day-2
        end
        infection_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day)
        Z[infection_time+36,individual_type_map.A,sim] += 1
        D[onset_time,individual_type_map.A,sim] += 1 
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
        if counter >= num_symptomatic_each_day[2]
            onset_time = day-1
        elseif counter >= num_symptomatic_each_day[3]
            onset_time = day-2
        end
        infection_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day)
        Z[infection_time+36,individual_type_map.S,sim] += 1
        U[onset_time,individual_type_map.S,sim] += 1 
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
        if counter >= num_asymptomatic_each_day[2]
            onset_time = day-1
        elseif counter >= num_asymptomatic_each_day[3]
            onset_time = day-2
        end
        infection_time = onset_time - 
            sample_onset_time(omicron = onset_time >= omicron_dominant_day)
        Z[infection_time+36,individual_type_map.A,sim] += 1
        U[onset_time,individual_type_map.A,sim] += 1 
        counter += 1
    end
    
    # println("total injected is: ", total_injected, " at time: ", day)
    
    return nothing 
    
end