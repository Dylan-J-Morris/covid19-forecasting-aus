using CSV
using DelimitedFiles
using DataFrames
using Dates
using Plots 
using TimerOutputs
using BenchmarkTools
using Revise
using PDFmerger

include("read_in_cases.jl")
include("branching_process.jl")
include("processing_sim.jl")
include("plot_forecasts.jl")

function simulate_all_states(file_date, states_to_run, nsims, run_simulation)
    """
    This runs the branching process for each of the states in states_to_run and then 
    merges and saves the files. 
    """
    
    # set seed for consistent plots (NOTE: this is not useful when multithreading 
    # enabled as we use separate seeds but the simulation pool should handle that)
    rng = Random.Xoshiro(2022)

    simulation_start_dates = Dict{String, String}(
        "NSW" => "2021-06-23",
        "QLD" => "2021-11-01",
        "SA" => "2021-11-01",
        "TAS" => "2021-11-01",
        "WA" => "2021-12-15",
        "ACT" => "2021-08-01",
        "NT" => "2021-12-01",
        "VIC" => "2021-08-01",
    )
    
    # date we want to apply increase in cases due to Omicron 
    omicron_dominant_date = Dates.Date("2021-12-15")

    # get the latest onset date
    latest_start_date = Dates.Date(maximum(v for v in values(simulation_start_dates)))

    pop_sizes = Dict{String, Int}(
        "NSW" => 8189266,
        "QLD" => 5221170,
        "SA" => 1773243,
        "TAS" => 541479,
        "VIC" => 6649159,
        "WA" => 2681633,
        "ACT" => 432266,
        "NT" => 246338,
    )
        
    initial_conditions = Dict{
        String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}
    }(
        "NSW" => (S = 5, A = 8, I = 0),
        "QLD" => (S = 0, A = 0, I = 0),
        "SA" => (S = 0, A = 0, I = 0),
        "TAS" => (S = 0, A = 0, I = 0),
        "VIC" => (S = 20, A = 20, I = 0),
        "WA" => (S = 3, A = 2, I = 0),
        "ACT" => (S = 0, A = 0, I = 0),
        "NT" => (S = 3, A = 2, I = 0),   
    )

    (local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
    dates = local_case_dict["date"]
    last_date_in_data = dates[end]
    forecast_end_date = last_date_in_data + Dates.Day(35)
    
    # create vector for dates 
    onset_dates = latest_start_date:Dates.Day(1):forecast_end_date
    
    # add a small truncation to the simulations as we don't trust the most recent data 
    truncation_days = 5
    
    if run_simulation 
        for state in states_to_run

            forecast_start_date = Dates.Date(simulation_start_dates[state])
            local_cases = local_case_dict[state]
            import_cases = import_case_dict[state]

            # named tuple for initial condi
            D0 = initial_conditions[state]
            N = pop_sizes[state]

            # get the observed cases 
            cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
            local_cases = local_case_dict[state][dates .>= forecast_start_date]
            # cutoff the last bit of the local cases
            import_cases = import_case_dict[state]
            local_cases = local_cases[begin:end-truncation_days]
            import_cases = import_cases[begin:end-truncation_days]
            
            (D, U, TP_local) = simulate_branching_process(
                D0, 
                N, 
                nsims[state], 
                local_cases, 
                import_cases, 
                cases_pre_forecast,
                forecast_start_date, 
                file_date, 
                omicron_dominant_date, 
                state,
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
    plot_all_forecast_intervals(file_date, states_to_run, local_case_dict)
    
    return nothing
    
end


function plot_all_forecast_intervals(file_date, states, local_case_dict)
    """
    Simple wrapper function to plot the forecasts with various zoom and confidence levels. 
    """
    
    plot_all_forecasts(
        file_date, 
        states_to_run, 
        local_case_dict, 
        confidence_level = "both",
    )
    plot_all_forecasts(
        file_date, 
        states_to_run, 
        local_case_dict, 
        zoom = true, 
        confidence_level = "both",
    )
    plot_all_forecasts(
        file_date, 
        states_to_run, 
        local_case_dict, 
        confidence_level = "50",
    )
    plot_all_forecasts(
        file_date, 
        states_to_run, 
        local_case_dict, 
        confidence_level = "95",
    )
    
    dir_name = "figs/case_forecasts/" * file_date * "/"
    file_name_tmp = "UoA_forecast_"

    pdf_filenames = [
        dir_name * file_name_tmp * file_date * "_zoomed_both_intervals.pdf",
        dir_name * file_name_tmp * file_date * "_both_intervals.pdf",
        dir_name * file_name_tmp * file_date * "_50_intervals.pdf",
    ]
    # merge the pdfs and delete the files
    merge_pdfs(
        pdf_filenames, 
        dir_name * file_name_tmp * file_date * ".pdf", 
        cleanup=true,
    )

    return nothing    
    
end