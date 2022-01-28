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
include("generative_model.jl")
include("processing_sim.jl")
include("plot_forecasts.jl")

function simulate_all_states(file_date, states_to_run, nsims)
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
        "WA" => "2021-11-01",
        "ACT" => "2021-08-01",
        "NT" => "2021-11-01",
        "VIC" => "2021-08-01",
    )
    
    # date we want to apply increase in cases due to Omicron 
    omicron_dominant_date = Dates.Date("2021-12-10")

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
        
    initial_conditions = Dict{String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}}(
        "NSW" => (S = 5, A = 8, I = 0),
        "QLD" => (S = 0, A = 0, I = 0),
        "SA" => (S = 0, A = 0, I = 0),
        "TAS" => (S = 0, A = 0, I = 0),
        "VIC" => (S = 20, A = 20, I = 0),
        "WA" => (S = 0, A = 0, I = 0),
        "ACT" => (S = 0, A = 0, I = 0),
        "NT" => (S = 0, A = 0, I = 0),    
    )

    (local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
    dates = local_case_dict["date"]
    last_date_in_data = dates[end]
    forecast_end_date = last_date_in_data + Dates.Day(35)
    
    # create vector for dates 
    onset_dates = latest_start_date:Dates.Day(1):forecast_end_date

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
        import_cases = import_case_dict[state]
        
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
        
        save_simulations(D,TP_local,state,file_date,onset_dates,rng)

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
    
    plot_all_forecasts(file_date, states_to_run, local_case_dict, confidence_level="both")
    plot_all_forecasts(file_date, states_to_run, local_case_dict, zoom=true, confidence_level="both")
    plot_all_forecasts(file_date, states_to_run, local_case_dict, confidence_level="50")
    plot_all_forecasts(file_date, states_to_run, local_case_dict, confidence_level="95")
    
    dir_name = "figs/case_forecasts/"*file_date*"/"
    file_name_tmp = "UoA_forecast_"

    pdf_filenames = [
        dir_name*file_name_tmp*file_date*"_zoomed_both_intervals.pdf",
        dir_name*file_name_tmp*file_date*"_both_intervals.pdf",
        dir_name*file_name_tmp*file_date*"_50_intervals.pdf",
    ]
    # merge the pdfs and delete the files
    merge_pdfs(
        pdf_filenames, 
        dir_name*file_name_tmp*file_date*".pdf", 
        cleanup=true,
    )

    return nothing    
end

function simulate_single_state(file_date,state,nsims)
    """
    This runs the branching process for each of the states in states_to_run and then 
    merges and saves the files. 
    """

    simulation_start_dates = Dict{String, String}(
        "NSW" => "2021-06-23",
        "QLD" => "2021-11-01",
        "SA" => "2021-11-01",
        "TAS" => "2021-11-01",
        "WA" => "2021-11-01",
        "ACT" => "2021-08-01",
        "NT" => "2021-11-01",
        "VIC" => "2021-08-01",
    )

    # get the latest onset date
    latest_onset_date = Dates.Date(maximum(v for v in values(simulation_start_dates)))
    onset_dates = latest_onset_date:Dates.Day(1):latest_onset_date+Dates.Day(size(D,1)-1)

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
        
    initial_conditions = Dict{String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}}(
        "NSW" => (S = 5, A = 5, I = 0),
        "QLD" => (S = 0, A = 0, I = 0),
        "SA" => (S = 0, A = 0, I = 0),
        "TAS" => (S = 0, A = 0, I = 0),
        "VIC" => (S = 20, A = 20, I = 0),
        "WA" => (S = 0, A = 0, I = 0),
        "ACT" => (S = 0, A = 0, I = 0),
        "NT" => (S = 0, A = 0, I = 0),    
    )

    (local_case_dict, import_case_dict) = read_in_cases(file_date)
    dates = local_case_dict["date"]

    forecast_start_date = Dates.Date(simulation_start_dates[state])
    local_cases = local_case_dict[state]
    import_cases = import_case_dict[state]

    # named tuple for initial condi
    D0 = initial_conditions[state]
    N = pop_sizes[state]

    # get the observed cases 
    cases_pre_forecast = sum(local_case_dict[state][.!plot_ind])
    local_cases = local_case_dict[state][plot_ind]
    import_cases = import_case_dict[state]
    
    (D, U, TP_local) = simulate_branching_process(
        D0, 
        N, 
        nsims, 
        local_cases, 
        import_cases, 
        cases_pre_forecast,
        forecast_start_date, 
        file_date, 
        state,
    )
    
    save_simulations(D,state,file_date,onset_dates)
    
    # merge all the simulation and TP files for states into single CSV
    merge_simulation_files(file_date)    
    merge_TP_files(file_date)
    
    # now we can plot everything
    plot_all_forecasts(file_date,state,local_case_dict,confidence_level="both")
    plot_all_forecasts(file_date,state,local_case_dict,confidence_level="50")
    plot_all_forecasts(file_date,state,local_case_dict,confidence_level="95")
    
    return nothing 
    
end

function visualise_data()
    
    # choose file date 
    file_date = "2022-01-04"
    
    # states to simulate 
    states_to_run = [
        "NSW",
        "QLD",
        "SA",
        "TAS",
        "VIC",
        "WA",
        "ACT",
        "NT",
    ]

    simulation_start_dates = Dict{String, String}(
        "NSW" => "2021-06-23",
        "QLD" => "2021-11-01",
        "SA" => "2021-11-01",
        "TAS" => "2021-11-01",
        "WA" => "2021-11-01",
        "ACT" => "2021-08-01",
        "NT" => "2021-11-01",
        "VIC" => "2021-08-01",
    )

    # get the latest onset date
    latest_onset_date = Dates.Date(maximum(v for v in values(simulation_start_dates)))
    onset_dates = latest_onset_date:Dates.Day(1):latest_onset_date+Dates.Day(size(D,1)-1)

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
        
    initial_conditions = Dict{String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}}(
        "NSW" => (S = 5, A = 5, I = 0),
        "QLD" => (S = 0, A = 0, I = 0),
        "SA" => (S = 0, A = 0, I = 0),
        "TAS" => (S = 0, A = 0, I = 0),
        "VIC" => (S = 20, A = 20, I = 0),
        "WA" => (S = 0, A = 0, I = 0),
        "ACT" => (S = 0, A = 0, I = 0),
        "NT" => (S = 0, A = 0, I = 0),    
    )

    (local_case_dict, import_case_dict) = read_in_cases(file_date)
    dates = local_case_dict["date"]
    
    # indices to actually plot over (simulation horizon)
    plot_ind = dates .>= forecast_start_date   

    plot_stuff = false
    if plot_stuff
        # plot the actual cases 
        fig = plot(legend=false)
        bar!(
            fig, 
            dates[plot_ind], 
            local_cases[plot_ind]+import_cases[plot_ind], 
            lc="gray", 
            fc="gray", 
            fillalpha=0.4, 
            linealpha=0.4
        )
        bar!(
            fig, 
            dates[plot_ind], 
            import_cases[plot_ind], 
            lc="gray", 
            fc="gray", 
            fillalpha=0.4, 
            linealpha=0.4
        )
    end
    
end
