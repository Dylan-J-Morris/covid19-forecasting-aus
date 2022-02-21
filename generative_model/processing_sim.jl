using DataFrames
using CSV

function save_simulations(
    D,
    TP_local,
    state,
    file_date,
    onset_dates,
    rng,
)
    """
    Saves a CSV in the same format as required for the ensemble forecast. 
    Also saves a CSV for the TP paths used per simulation.
    """
    df_observed = DataFrame()
    df_TP_local = DataFrame()
    df_observed[!, "state"] = [state for _ in 1:length(onset_dates)]
    df_observed[!, "onset date"] = onset_dates
    df_TP_local[!, "state"] = [state for _ in 1:length(onset_dates)]
    df_TP_local[!, "onset date"] = onset_dates
    # total observed cases 
    D_observed = D[:, 1, :] + D[:, 2, :]
    
    # indexer for naming the columns 
    i = 0
    for (d, tp) in zip(eachcol(D_observed), eachcol(TP_local))
        df_observed[!, "sim" * string(i)] = d[(end - length(onset_dates) + 1):end]
        df_TP_local[!, "sim" * string(i)] = tp[(end - length(onset_dates) + 1):end]
        i += 1
        if i == 2000
            break
        end
    end
    
    if i == 0  
        for i in 0:1999
            # if nothing sampled, set observations to be missing
            df_observed[!, "sim" * string(i)] = [missing for _ in 1:length(onset_dates)]
            df_TP_local[!, "sim" * string(i)] = [missing for _ in 1:length(onset_dates)]
        end
    elseif i < 2000 
        # if we are under the required number of sims, sample randomly from the good sims 
        for j in i:1999
            # sample a random column 
            if i == 1
                ind = 1
            else
                ind = rand(rng, 1:(i - 1))
            end
            d = D_observed[:, ind]
            tp = TP_local[:, ind]
            df_observed[!, "sim" * string(j)] = d[(end - length(onset_dates) + 1):end]
            df_TP_local[!, "sim" * string(j)] = tp[(end - length(onset_dates) + 1):end]
        end
    end
    
    # this adds a column for the date we are supplied the NINDSS file 
    df_observed[!, "data date"] .= file_date
    df_TP_local[!, "data date"] .= file_date
    
    # directory path, and check to see whether it's good
    dir_name = joinpath("results", "UoA_forecast_output", file_date)
    if !ispath(dir_name)
        mkpath(dir_name)
    end
    
    # file name is just the state and file date
    sim_file_name = state * "_" * file_date * "_sim.csv"
    TP_file_name = state * "_" * file_date * "_TP.csv"
    CSV.write(dir_name * "/" * sim_file_name, df_observed)
    CSV.write(dir_name * "/" * TP_file_name, df_TP_local)
    
    return nothing
    
end

function merge_simulation_files(file_date)
    """
    Merge the simulation files into a single file. 
    """
    # set the dir name and read in the state files for merging
    dir_name = joinpath("results", "UoA_forecast_output", file_date)
    state_file_names = readdir(dir_name)
    # need to remove the non-output files
    states = []
    for f in state_file_names
        ind = findfirst("_", f)[1]
        if (f[1:ind-1] ∉ states) && (f[1:ind-1] != "UoA")
            push!(states, f[1:ind-1])
        end
    end

    # now we make sure the filenames are as we want
    for (i,f) in enumerate(states)
        g = f * "_" * file_date * "_sim.csv"
        states[i] = g 
    end
    
    df_merged = DataFrame()
    df_tmp = DataFrame()
    
    # loop over the states and read in and add to the merged df
    for s in states
        df_tmp = CSV.read(dir_name * "/" * s, DataFrame)
        df_merged = [df_merged; df_tmp]
    end
    
    CSV.write(dir_name * "/" * "UoA_samples_" * file_date * ".csv", df_merged)
    
    return nothing
    
end

function merge_TP_files(file_date)
    """
    Merge the TP files in a similar format to the overall simulation file.
    """
    # set the dir name and read in the state files for merging
    dir_name = joinpath("results", "UoA_forecast_output", file_date)
    state_file_names = readdir(dir_name)
    # need to remove the non-output files
    states = []
    for f in state_file_names
        ind = findfirst("_", f)[1]
        if (f[1:ind-1] ∉ states) && (f[1:ind-1] != "UoA")
            push!(states, f[1:ind-1])
        end
    end

    # now we make sure the filenames are as we want
    for (i,f) in enumerate(states)
        g = f * "_" * file_date * "_TP.csv"
        states[i] = g 
    end
    
    df_merged = DataFrame()
    df_tmp = DataFrame()
    
    # loop over the states and read in and add to the merged df
    for s in states
        df_tmp = CSV.read(dir_name * "/" * s, DataFrame)
        df_merged = [df_merged; df_tmp]
    end
    
    CSV.write(dir_name * "/" * "UoA_TP_" * file_date * ".csv", df_merged)
    
    return nothing
    
end

function summarise_forecast_for_plotting(D_observed)
    """
    Summarises the simulations into a dataframe.
    """
    
    df_observed_summary = DataFrame(
        "median" => median.(eachrow(D_observed)), 
        "bottom" => quantile.(eachrow(D_observed), 0.025),
        "lower" => quantile.(eachrow(D_observed), 0.25),
        "upper" => quantile.(eachrow(D_observed), 0.75),
        "top" => quantile.(eachrow(D_observed), 0.975),
    )
    
    return df_observed_summary
    
end