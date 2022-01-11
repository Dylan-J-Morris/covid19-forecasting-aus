using FileIO
using CSV
using DataFrames
using Plots
using Measures      # for better positioning of figures 
pyplot()

include("read_in_cases.jl")
include("processing_sim.jl")

function plot_all_forecasts(
    file_date, 
    local_case_dict; 
    confidence_level="both"
)
    """
    Plots all the forecasts and saves the result as a pdf. 
    """
    
    # read in the sim results and kept TP's
    sim_all_states = CSV.read(
        "results/UoA_forecast_output/2022-01-04/UoA_samples_"*file_date*".csv", 
        DataFrame
    )
    TP_all_states = CSV.read(
        "results/UoA_forecast_output/2022-01-04/UoA_TP_"*file_date*".csv", 
        DataFrame
    )
    
    # read in the case data 
    case_dates = collect(local_case_dict["date"])

    # states to simulate 
    states_to_plot = [
        "NSW",
        "QLD",
        "SA",
        "TAS",
        "VIC",
        "WA",
        "ACT",
        "NT",
    ]
    
    # these indices are for the relevant plots per state 
    case_plot_inds = [1,2,5,6,9,10,13,14]
    TP_plot_inds = [3,4,7,8,11,12,15,16]
    
    # these are heights of the relevant plots
    heights = [0.4, 0.2, 0.4 , 0.2, 0.4 , 0.2, 0.4 , 0.2]
    # normalize the heights for a good fit 
    height_norm = heights / sum(heights)
    # layer the heights so that we get the full length of the plot 
    heights_col = [height_norm; height_norm]
    
    # define a custom 8x2 layout with the chosen heights 
    l = @layout [
        grid(8, 2, heights=heights_col)
    ]
    
    # initialise the plot 
    fig = plot(
        legend=false, 
        layout = l, 
        dpi=200, 
        size=(750,1200), 
        link=:x, 
        framestyle=:box, 
        margin=2mm,
        rightmargin=3mm,
    )
    
    for (i,state) in enumerate(states_to_plot)
        
        # take indices for plots from the defined index arrays 
        c = case_plot_inds[i]
        tp = TP_plot_inds[i]
        
        # get relevant columns of the dataframe and merge to matrix
        D = Matrix(sim_all_states[sim_all_states.state .== state, 3:end-1])
        # if anything in D is missing, it means the sims didn't work so 0 out those entries
        D[ismissing.(D)] .= 0
        D = Matrix{Int}(D)
        
        TP_local = Matrix(TP_all_states[TP_all_states.state .== state, 3:end-1])
        TP_local[ismissing.(TP_local)] .= 0.0
        TP_local = Matrix{Float64}(TP_local)
        
        onset_dates = sim_all_states[sim_all_states.state .== state, "onset date"]
        df_D_summary = summarise_forecast_for_plotting(D)
        df_TP_summary = summarise_forecast_for_plotting(TP_local)

        dates = local_case_dict["date"]
        local_cases = local_case_dict[state][dates .>= onset_dates[1]]
        # boolean for correct case dates 
        case_dates_ind = [d ∈ onset_dates ? true : false for d in case_dates]

        onset_dates_lims = Dates.value.([onset_dates[end]-Dates.Day(75), onset_dates[end]])
        
        if confidence_level == "50" || confidence_level == "both"
            plot!(
                fig,
                title=state, 
                subplot=c, 
                onset_dates, 
                df_D_summary[!,"median"], 
                xaxis=nothing,
                linecolor=1, 
                linewidth=2, 
                ribbon=(
                    vec(df_D_summary[!,"median"]-df_D_summary[!,"lower"]), 
                    vec(df_D_summary[!,"upper"] - df_D_summary[!,"median"])
                ), 
                color=1, 
                fillalpha=0.3,
            )
        end 
        
        if confidence_level == "95" || confidence_level == "both"
            plot!(
                fig, 
                title=state,
                subplot=c, 
                onset_dates, 
                df_D_summary[!,"median"], 
                xaxis=nothing,
                linecolor=1, 
                linewidth=2, 
                ribbon=(
                    vec(df_D_summary[!,"median"]-df_D_summary[!,"bottom"]), 
                    vec(df_D_summary[!,"top"] - df_D_summary[!,"median"])
                ), 
                color=1, 
                fillalpha=0.3,
            )
        end
        
        bar!(
            fig, 
            subplot=c, 
            case_dates[case_dates_ind], 
            local_cases, 
            lc="gray", 
            fc="gray", 
            fillalpha=0.4, 
            linealpha=0.4,
        )
        
        if confidence_level == "50" || confidence_level == "both"
            plot!(
                fig, 
                subplot=tp, 
                onset_dates, 
                df_TP_summary[!,"median"], 
                linecolor=1, 
                linewidth=2, 
                ribbon=(
                    vec(df_TP_summary[!,"median"]-df_TP_summary[!,"lower"]), 
                    vec(df_TP_summary[!,"upper"] - df_TP_summary[!,"median"])
                ), 
                color=1, 
                fillalpha=0.3,
            )
        end
        
        if confidence_level == "95" || confidence_level == "both"
            plot!(
                fig, 
                subplot=tp, 
                onset_dates, 
                df_TP_summary[!,"median"], 
                linecolor=1, 
                linewidth=2, 
                ribbon=(
                    vec(df_TP_summary[!,"median"]-df_TP_summary[!,"bottom"]), 
                    vec(df_TP_summary[!,"top"] - df_TP_summary[!,"median"])
                ), 
                color=1, 
                fillalpha=0.3,
            )
        end
        
        hline!(
            fig, 
            subplot=tp, 
            [1], 
            lc=:black, 
            ls=:dash
        )

        xlims!(fig, subplot=c, onset_dates_lims...)
        xlims!(fig, subplot=tp, onset_dates_lims...)
        ylims!(fig, subplot=tp, 0, 1.25*maximum(df_TP_summary[!,"top"]))
        
    end

    dir_name = joinpath("figures", file_date)
    if !ispath(dir_name)
        mkpath(dir_name)
    end
    
    savefig(fig, dir_name*"/UoA_forecast_"*file_date*"_"*confidence_level*"_intervals.pdf")

    return nothing 

end

function plot_single_forecast(
    file_date, 
    state, 
    local_case_dict; 
    confidence_level="both"
)
    """
    Plots all the forecasts and saves the result as a pdf. 
    """
    
    # read in the sim results and kept TP's
    sim_all_states = CSV.read(
        "results/UoA_forecast_output/2022-01-04/UoA_samples_"*file_date*".csv", 
        DataFrame
    )
    TP_all_states = CSV.read(
        "results/UoA_forecast_output/2022-01-04/UoA_TP_"*file_date*".csv", 
        DataFrame
    )
    
    # read in the case data 
    case_dates = collect(local_case_dict["date"])
    
    # these are heights of the relevant plots
    heights = [0.4, 0.2]
    # normalize the heights for a good fit 
    height_norm = heights / sum(heights)
    
    # define a custom 8x2 layout with the chosen heights 
    l = @layout [
        grid(2, 1, heights=heights)
    ]
    
    # initialise the plot 
    fig = plot(
        legend=false, 
        layout = l, 
        dpi=144, 
        size=(700,1200), 
        link=:x, 
        framestyle=:box
    )
    
    c = 1
    tp = 2
    
    # get relevant columns of the dataframe and merge to matrix
    D = Matrix(sim_all_states[sim_all_states.state .== state, 3:end-1])
    # if anything in D is missing, it means the sims didn't work so 0 out those entries
    D[ismissing.(D)] .= 0
    D = Matrix{Int}(D)
    
    TP_local = Matrix(TP_all_states[TP_all_states.state .== state, 3:end-1])
    TP_local[ismissing.(TP_local)] .= 0.0
    TP_local = Matrix{Float64}(TP_local)
    
    onset_dates = sim_all_states[sim_all_states.state .== state, "onset date"]
    df_D_summary = summarise_forecast_for_plotting(D)
    df_TP_summary = summarise_forecast_for_plotting(TP_local)

    dates = local_case_dict["date"]
    local_cases = local_case_dict[state][dates .>= onset_dates[1]]
    # boolean for correct case dates 
    case_dates_ind = [d ∈ onset_dates ? true : false for d in case_dates]

    onset_dates_lims = Dates.value.([onset_dates[end]-Dates.Day(75), onset_dates[end]])

    annotate!(
        fig,
        subplot=c,
        (0.2,0.8),
        text(state, 8)
    )
    
    if confidence_level == "50" || confidence_level == "both"
        plot!(
            fig, 
            subplot=c, 
            onset_dates, 
            df_D_summary[!,"median"], 
            xaxis=nothing,
            linecolor=1, 
            linewidth=2, 
            ribbon=(
                vec(df_D_summary[!,"median"]-df_D_summary[!,"lower"]), 
                vec(df_D_summary[!,"upper"] - df_D_summary[!,"median"])
            ), 
            color=1, 
            fillalpha=0.2,
        )
    end 
    
    if confidence_level == "95" || confidence_level == "both"
        plot!(
            fig, 
            subplot=c, 
            onset_dates, 
            df_D_summary[!,"median"], 
            xaxis=nothing,
            linecolor=1, 
            linewidth=2, 
            ribbon=(
                vec(df_D_summary[!,"median"]-df_D_summary[!,"bottom"]), 
                vec(df_D_summary[!,"top"] - df_D_summary[!,"median"])
            ), 
            color=1, 
            fillalpha=0.2
        )
    end
    
    bar!(
        fig, 
        subplot=c, 
        case_dates[case_dates_ind], 
        local_cases, 
        lc="gray", 
        fc="gray", 
        fillalpha=0.4, 
        linealpha=0.4,
    )
    
    if confidence_level == "50" || confidence_level == "both"
        plot!(
            fig, 
            subplot=tp, 
            onset_dates, 
            df_TP_summary[!,"median"], 
            linecolor=1, 
            linewidth=2, 
            ribbon=(
                vec(df_TP_summary[!,"median"]-df_TP_summary[!,"lower"]), 
                vec(df_TP_summary[!,"upper"] - df_TP_summary[!,"median"])
            ), 
            color=1, 
            fillalpha=0.2
        )
    end
    
    if confidence_level == "95" || confidence_level == "both"
        plot!(
            fig, 
            subplot=tp, 
            onset_dates, 
            df_TP_summary[!,"median"], 
            linecolor=1, 
            linewidth=2, 
            ribbon=(
                vec(df_TP_summary[!,"median"]-df_TP_summary[!,"bottom"]), 
                vec(df_TP_summary[!,"top"] - df_TP_summary[!,"median"])
            ), 
            color=1, 
            fillalpha=0.2,
        )
    end
    
    hline!(
        fig, 
        subplot=tp, 
        [1], 
        lc=:black, 
        ls=:dash
    )

    xlims!(fig, subplot=c, onset_dates_lims...)
    xlims!(fig, subplot=tp, onset_dates_lims...)
    ylims!(fig, subplot=tp, 0, 1.25*maximum(df_TP_summary[!,"top"]))

    dir_name = joinpath("figures", file_date)
    if !ispath(dir_name)
        mkpath(dir_name)
    end
    
    savefig(fig, dir_name*"/UoA_forecast_"*file_date*"_"*state*"_"*confidence_level*"_intervals.pdf")

    return nothing 

end