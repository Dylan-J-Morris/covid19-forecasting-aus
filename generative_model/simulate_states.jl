using CSV
using DelimitedFiles
using DataFrames
using Dates
using Plots 
using TimerOutputs
using BenchmarkTools
using Revise

include("branching_process.jl")
include("processing_sim.jl")

function simulate_single_state(
    file_date, 
    state, 
    nsims, 
    run_simulation; 
    truncation_days = 7,
    p_detect_omicron = 0.5,
    adjust_TP = false, 
)
    """
    This runs the branching process for a single state. 
    """
    
    # set seed for consistent plots (NOTE: this is not useful when multithreading 
    # enabled as we use separate seeds but the simulation pool should handle that)
    # rng = Random.seed!(2022)
    rng = Random.Xoshiro(2022)
    
    jurisdiction_assumptions = JurisdictionAssumptions()
    
    # get the latest onset date
    latest_start_date = Dates.Date(
        maximum(v for v in values(jurisdiction_assumptions.simulation_start_dates))
    )

    (local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
    dates = local_case_dict["date"]
    last_date_in_data = dates[end]
    forecast_end_date = last_date_in_data + Dates.Day(35)
    
    # create vector for dates 
    onset_dates = latest_start_date:Dates.Day(1):forecast_end_date
    
    if run_simulation 
        forecast_start_date = Dates.Date(
            jurisdiction_assumptions.simulation_start_dates[state]
        )
        local_cases = local_case_dict[state]
        import_cases = import_case_dict[state]

        # named tuple for initial condi
        D0 = jurisdiction_assumptions.initial_conditions[state]
        N = jurisdiction_assumptions.pop_sizes[state]

        # get the observed cases 
        cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
        local_cases = local_case_dict[state][dates .>= forecast_start_date]
        import_cases = import_case_dict[state]
        # cutoff the last bit of the local cases
        local_cases = local_cases[begin:end - truncation_days + 1]
        import_cases = import_cases[begin:end - truncation_days + 1]
        
        (D, U, TP_local) = simulate_branching_process(
            D0, 
            N, 
            nsims, 
            local_cases, 
            import_cases, 
            cases_pre_forecast,
            forecast_start_date, 
            file_date, 
            jurisdiction_assumptions.omicron_dominant_date, 
            state,
            p_detect_omicron = p_detect_omicron,
            adjust_TP = adjust_TP, 
        )
        
        save_simulations(
            D,
            TP_local,
            state,
            file_date,
            onset_dates,
            rng,
        )
    end
    
    return nothing
    
end


function simulate_all_states(
    file_date, 
    states, 
    nsims, 
    run_simulation; 
    truncation_days = 7,
    p_detect_omicron = 0.5,
    adjust_TP = false, 
)
    """
    This runs the branching process for each of the states in states and then 
    merges and saves the files. If running a single state, this merging of files is 
    done in a post-processing step. 
    """
    
    # set seed for consistent plots (NOTE: this is not useful when multithreading 
    # enabled as we use separate seeds but the simulation pool should handle that)
    rng = Random.seed!(2022)

    jurisdiction_assumptions = JurisdictionAssumptions()

    # get the latest onset date
    latest_start_date = Dates.Date(
        maximum(v for v in values(jurisdiction_assumptions.simulation_start_dates))
    )

    (local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
    dates = local_case_dict["date"]
    last_date_in_data = dates[end]
    forecast_end_date = last_date_in_data + Dates.Day(35)
    
    # create vector for dates 
    onset_dates = latest_start_date:Dates.Day(1):forecast_end_date
    
    if run_simulation 
        for state in states
            
            println("======================")
            println("Simulating ", state)
            println("======================")

            forecast_start_date = Dates.Date(
                jurisdiction_assumptions.simulation_start_dates[state]
            )
            local_cases = local_case_dict[state]
            import_cases = import_case_dict[state]

            # named tuple for initial condi
            D0 = jurisdiction_assumptions.initial_conditions[state]
            N = jurisdiction_assumptions.pop_sizes[state]

            # get the observed cases 
            cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
            local_cases = local_case_dict[state][dates .>= forecast_start_date]
            # cutoff the last bit of the local cases
            import_cases = import_case_dict[state]
            local_cases = local_cases[begin:end - truncation_days + 1]
            import_cases = import_cases[begin:end - truncation_days + 1]
            
            (D, U, TP_local) = simulate_branching_process(
                D0, 
                N, 
                nsims[state], 
                local_cases, 
                import_cases, 
                cases_pre_forecast,
                forecast_start_date, 
                file_date, 
                jurisdiction_assumptions.omicron_dominant_date, 
                state,
                p_detect_omicron = p_detect_omicron, 
                adjust_TP = adjust_TP,
                parallel = true
            )
            
            save_simulations(
                D,
                TP_local,
                state,
                file_date,
                onset_dates,
                rng,
            )

        end
    end

    # merge all the simulation and TP files for states into single CSV
    merge_simulation_files(file_date)    
    merge_TP_files(file_date)
    
    return nothing
    
end