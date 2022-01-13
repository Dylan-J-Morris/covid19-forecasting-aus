include("forecast_types.jl")
include("helper_functions.jl")

function count_cases_in_windows!(
    case_counts,  
    window_lengths, 
    D_total, 
    simulation_constants::SimulationConstants, 
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
    Z, 
    D, 
    U,
    D_total, 
    D_total_cumsum,
    case_counts, 
    sim,
    local_cases,
    window_lengths, 
    min_cases, 
    max_cases, 
    reinitialise_allowed, 
    sim_features::SimulationFeatures,
    sim_constants::SimulationConstants, 
    individual_type_map::IndividualTypeMap;
    day = 0,
)
    """
    Check for consistency of the simulations against the data. This will also 
    check for instances of superspreading events and will insert cases if conditions
    are met.
    """
    
    print_status = false
    
    # days forecast observed for 
    T_observed = sim_features.T_observed
    max_forecast_cases = sim_features.max_forecast_cases
    
    consistency_multiplier = sim_constants.consistency_multiplier
    
    # initialise the bad sim
    bad_sim = false
    injected_cases = false 
    
    # take out the observed data by adding the observed compartments
    # then we take the cumulative sum as this lets us differnce the vector 
    # more easily
    if day <= T_observed
        for i in 1:day
            D_total[i,sim] = D[i,1,sim] + D[i,2,sim]
            # this is just an inplace, non-allocating cumulative sum
            if i == 1
                D_total_cumsum[i,sim] = D_total[i,sim]
            else 
                D_total_cumsum[i,sim] = D_total[i-1,sim] + D_total[i,sim]
            end
        end
        
        count_cases_in_windows!(
            case_counts, 
            window_lengths, 
            D_total, 
            sim_constants::SimulationConstants, 
            sim,
        )
    end
    
    # this is just the total cases over the forecast horizon 
    D_forecast = sum(@view D[T_observed+1:end,1:2,sim])
    
    # if we've exceeded the max cases over a given period
    if D_forecast > max_forecast_cases
        bad_sim = true
        print_status && println("too many forecast cases")
    end
    
    if !bad_sim 
        for (i, (c,m)) in enumerate(zip(case_counts,max_cases))
            if c > m 
                bad_sim = true 
                print_status && println("too many cases", c, " ", m, " window: ", i)
                break
            end
        end
        
        if day == 0
            for (i, (c,m)) in enumerate(zip(case_counts,min_cases))
                if c < m 
                    bad_sim = true 
                    print_status && println("too few cases", c, " ", m, " window: ", i)
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
        sim_3_day_cases = sum(@view D_total[day-2:day,sim])
        sim_3_day_threshold = consistency_multiplier*max(1, sim_3_day_cases)
        # calculate total number of cases over week 
        sim_week_cases = sum(@view D_total[day-6:day,sim])
        missing_detections = 0
        
        if actual_3_day_cases > sim_3_day_threshold || (sim_week_cases == 0 && actual_3_day_cases > 0)
            # println("actual: ", actual_3_day_cases, " 3day: ", sim_3_day_cases, " 3day thresh: ", sim_3_day_threshold, " week: ", sim_week_cases, " day added: ", day)
            # instead of sampling exactly the number required, just sample a random amount 
            if sim_week_cases == 0
                missing_detections = rand(1:actual_3_day_cases)
            else
                missing_detections = rand(1:(actual_3_day_cases - sim_3_day_cases))
            end
            # println("simulated: ", sum(D_total[day-2:day]))
            # println("actual: ", sum(floor.(Int, consistency_multiplier*observed_cases[day-2:day])))
            injected_cases = true 
            inject_cases!(
                Z, 
                D, 
                U,
                missing_detections, 
                day, 
                sim, 
                sim_constants, 
                sim_features, 
                individual_type_map,
            )
        end
    end
    
    return (bad_sim, injected_cases)
end

function inject_cases!(
    Z, 
    D, 
    U,
    missing_detections, 
    day, 
    sim,
    sim_constants::SimulationConstants,
    sim_features::SimulationFeatures,
    individual_type_map::IndividualTypeMap,
)
    """
    A model for injecting cases into the simulation following long periods of low 
    incidence. 
    """
    
    total_injected = missing_detections
    
    p_symp = sim_constants.p_symp
    p_detect_given_symp = sim_constants.p_detect_given_symp
    p_detect_given_asymp = sim_constants.p_detect_given_asymp
    
    # prob of being symptomatic given detection 
    p_detect = p_symp*p_detect_given_symp + (1-p_symp)*p_detect_given_asymp
    p_symp_given_detect = p_detect_given_symp*p_symp/p_detect 
    # sample number symptomatic detected
    num_symptomatic_detected = sample_binomial_limit(missing_detections, p_symp_given_detect)
    num_asymptomatic_detected = missing_detections - num_symptomatic_detected
    # infer some undetected symptomatic
    num_symptomatic_undetected = sample_negative_binomial_limit(missing_detections, p_detect_given_symp)
    # infer some undetected asumptomatic
    num_symptomatic_total = num_symptomatic_detected + num_symptomatic_undetected
    
    if num_symptomatic_total == 0
        num_asymptomatic_undetected = sample_negative_binomial_limit(1, p_symp)
    else
        num_asymptomatic_undetected = sample_negative_binomial_limit(num_symptomatic_total, p_symp)
    end
    
    # detection times are determined triangularly 
    num_symptomatic_each_day = (
        num_symptomatic_detected ÷ 2, 
        num_symptomatic_detected ÷ 2 + num_symptomatic_detected ÷ 3,
        num_symptomatic_detected ÷ 2 + num_symptomatic_detected ÷ 3 + num_symptomatic_detected ÷ 6 
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
        infection_time = onset_time - sample_onset_time()
        Z[infection_time+35,individual_type_map.S,sim] += 1
        D[onset_time,individual_type_map.S,sim] += 1 
        counter += 1
    end
    
    # detection times are determined triangularly 
    num_asymptomatic_each_day = (
        num_asymptomatic_detected ÷ 2, 
        num_asymptomatic_detected ÷ 2 + num_asymptomatic_detected ÷ 3,
        num_asymptomatic_detected ÷ 2 + num_asymptomatic_detected ÷ 3 + num_asymptomatic_detected ÷ 6 
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
        infection_time = onset_time - sample_onset_time()
        Z[infection_time+35,individual_type_map.A,sim] += 1
        D[onset_time,individual_type_map.A,sim] += 1 
        counter += 1
    end
    
    # now add in the inferred undetected cases in the same way. Note that the 
    # onset times are kinda irrelevant here, but just help with consistent 
    # simulation recording.
    num_symptomatic_each_day = (
        num_symptomatic_undetected ÷ 2, 
        num_symptomatic_undetected ÷ 2 + num_symptomatic_undetected ÷ 3,
        num_symptomatic_undetected ÷ 2 + num_symptomatic_undetected ÷ 3 + num_symptomatic_undetected ÷ 6 
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
        infection_time = onset_time - sample_onset_time()
        Z[infection_time+35,individual_type_map.S,sim] += 1
        U[onset_time,individual_type_map.S,sim] += 1 
        counter += 1
    end
    
    # detection times are determined triangularly 
    num_asymptomatic_each_day = (
        num_asymptomatic_undetected ÷ 2, 
        num_asymptomatic_undetected ÷ 2 + num_asymptomatic_undetected ÷ 3,
        num_asymptomatic_undetected ÷ 2 + num_asymptomatic_undetected ÷ 3 + num_asymptomatic_undetected ÷ 6 
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
        infection_time = onset_time - sample_onset_time()
        Z[infection_time+35,individual_type_map.A,sim] += 1
        U[onset_time,individual_type_map.A,sim] += 1 
        counter += 1
    end
    
    # println("total injected is: ", total_injected, " at time: ", day)
    
    return nothing 
end