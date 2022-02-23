using FileIO
using CSV
using DataFrames
using Plots
using Measures      # for better positioning of figures 
using PDFmerger

# if you have python installed, you can use pyplot for better formatting
pyplot()
# gr()

include("processing_sim.jl")

function plot_all_forecasts(
    file_date, 
    states,
    local_case_dict; 
    zoom = false,
    confidence_level = "both",
    truncation_days = 7,
)
    """
    Plots all the forecasts and saves the result as a pdf. 
    """
    
    # read in the sim results and kept TP's
    sim_all_states = CSV.read(
        "results/UoA_forecast_output/" * file_date * "/UoA_samples_" * file_date * ".csv", 
        DataFrame,
    )
    TP_all_states = CSV.read(
        "results/UoA_forecast_output/" * file_date * "/UoA_TP_" * file_date * ".csv", 
        DataFrame,
    )
    
    # read in the case data 
    case_dates = collect(local_case_dict["date"])
    
    # these indices are for the relevant plots per state 
    case_plot_inds = [1, 2, 5, 6, 9, 10, 13, 14]
    TP_plot_inds = [3, 4, 7, 8, 11, 12, 15, 16]
    
    # these are heights of the relevant plots
    heights = [0.4, 0.2, 0.4 , 0.2, 0.4 , 0.2, 0.4 , 0.2]
    # normalize the heights for a good fit 
    height_norm = heights / sum(heights)
    # layer the heights so that we get the full length of the plot 
    heights_col = [height_norm; height_norm]
    
    # define a custom 8x2 layout with the chosen heights 
    l = @layout [
        grid(8, 2, heights = heights_col)
    ]
    
    # initialise the plot 
    fig = plot( 
        layout = l, 
        dpi = 200, 
        size = (750, 1200), 
        link = :x, 
        framestyle = :box, 
        margin = 3mm,
        rightmargin = 4mm,
        legend = :outerright
    )
    
    for (i, state) in enumerate(states)
        
        # take indices for plots from the defined index arrays 
        c = case_plot_inds[i]
        tp = TP_plot_inds[i]
        
        # get relevant columns of the dataframe and merge to matrix
        D = Matrix(sim_all_states[sim_all_states.state .== state, 3:end - 1])
        # if anything in D is missing, it means the sims didn't work so 0 out those entries
        D[ismissing.(D)] .= 0
        D = Matrix{Int}(D)
        
        TP_local = Matrix(TP_all_states[TP_all_states.state .== state, 3:end - 1])
        TP_local[ismissing.(TP_local)] .= 0.0
        TP_local = Matrix{Float64}(TP_local)
        
        onset_dates = sim_all_states[sim_all_states.state .== state, "onset date"]
        df_D_summary = summarise_forecast_for_plotting(D)
        df_TP_summary = summarise_forecast_for_plotting(TP_local)

        dates = local_case_dict["date"]
        local_cases = local_case_dict[state][dates .>= onset_dates[1]]
        # boolean for correct case dates 
        case_dates_ind = [d ∈ onset_dates ? true : false for d in case_dates]

        onset_dates_lims = Dates.value.(
            [onset_dates[end] - Dates.Day(90), onset_dates[end]]
        )
        
        if confidence_level == "50" || confidence_level == "both"
            plot!(
                fig,
                title = state, 
                subplot = c, 
                onset_dates, 
                df_D_summary[!, "median"], 
                xaxis = nothing,
                linecolor = 1, 
                linewidth = 1, 
                color = 1, 
                fillalpha = 0.2,
                label = false,
            )
            plot!(
                fig, 
                subplot = c,
                onset_dates, 
                [df_D_summary[!, "median"] df_D_summary[!, "median"]], 
                fillrange = [df_D_summary[!, "lower"] df_D_summary[!, "upper"]],
                fillalpha = 0.4, 
                c = 1, 
                label = ["50%" nothing]
            )
            
        end 
        
        if confidence_level == "95" || confidence_level == "both"
            plot!(
                fig, 
                title = state,
                subplot = c, 
                onset_dates, 
                df_D_summary[!, "median"], 
                xaxis = nothing,
                linecolor = 1, 
                linewidth = 1, 
                color = 1, 
                fillalpha = 0.2,
                label = false,
            )
            
            plot!(
                fig, 
                subplot = c,
                onset_dates, 
                [df_D_summary[!, "median"] df_D_summary[!, "median"]], 
                fillrange = [df_D_summary[!, "bottom"] df_D_summary[!, "top"]],
                fillalpha = 0.2, 
                c = 1, 
                label = ["95%" nothing]
            )
        end
        
        # bar!(
        #     fig, 
        #     subplot = c, 
        #     case_dates[case_dates_ind], 
        #     local_cases, 
        #     lc = "gray", 
        #     fc = "gray", 
        #     fillalpha = 0.4, 
        #     linealpha = 0.4,
        #     label = false,
        # )
        
        plot!(
            fig, 
            subplot = c, 
            case_dates[case_dates_ind], 
            local_cases, 
            lc = "black",
            label = false,
        )
        
        if confidence_level == "50" || confidence_level == "both"
            plot!(
                fig, 
                subplot = tp, 
                onset_dates, 
                df_TP_summary[!, "median"], 
                linecolor = 1, 
                linewidth = 1, 
                color = 1, 
                fillalpha = 0.2,
                label = false,
            )
            
            plot!(
                fig, 
                subplot = tp,
                onset_dates, 
                [df_TP_summary[!, "median"] df_TP_summary[!, "median"]], 
                fillrange = [df_TP_summary[!, "lower"] df_TP_summary[!, "upper"]],
                fillalpha = 0.4, 
                c = 1, 
                label = false
            )
        end
        
        if confidence_level == "95" || confidence_level == "both"
            plot!(
                fig, 
                subplot = tp, 
                onset_dates, 
                df_TP_summary[!, "median"], 
                linecolor = 1, 
                linewidth = 1, 
                color = 1, 
                fillalpha = 0.2,
                label = false,
            )
            
            plot!(
                fig, 
                subplot = tp,
                onset_dates, 
                [df_TP_summary[!, "median"] df_TP_summary[!, "median"]], 
                fillrange = [df_TP_summary[!, "bottom"] df_TP_summary[!, "top"]],
                fillalpha = 0.2, 
                c = 1, 
                label = false
            )
        end
        
        # plot the world famous Reff = 1 (= TP)
        hline!(
            fig, 
            subplot = tp, 
            [1], 
            lc = :black, 
            ls = :dash,
            label = false,
        )
        
        # plot the last date used for conditioning the simulations 
        vline!(
            fig, 
            subplot = c,
            [Dates.Date(file_date) - Dates.Day(truncation_days)], 
            lc = :gray,
            ls = :dash,
            label = false,
        )
        vline!(
            fig, 
            subplot = tp,
            [Dates.Date(file_date) - Dates.Day(truncation_days)], 
            lc = :gray,
            ls = :dash,
            label = false,
        )
        
        # adjust limits to make plots a little nicer to look at 
        xlims!(fig, subplot = c, onset_dates_lims...)
        xlims!(fig, subplot = tp, onset_dates_lims...)
        ylims!(fig, subplot = tp, 0, 1.25 * maximum(df_TP_summary[!, "top"]))
        if zoom 
            ylims!(fig, subplot = c, 0, 1.25 * maximum(local_cases))
        end
        
    end

    dir_name = joinpath("figs", "case_forecasts", file_date)
    if !ispath(dir_name)
        mkpath(dir_name)
    end
    
    if zoom 
        savefig(
            fig, 
            dir_name *
            "/UoA_forecast_" *
            file_date *
            "_" *
            "zoomed_" *
            confidence_level *
            "_intervals.pdf"
        )
    else
        savefig(
            fig, 
            dir_name *
            "/UoA_forecast_" *
            file_date *
            "_" *
            confidence_level *
            "_intervals.pdf"
        )
    end
        
    return nothing 
    
end


function plot_all_forecast_intervals(file_date, states, local_case_dict)
    """
    Simple wrapper function to plot the forecasts with various zoom and confidence levels. 
    """
    
    plot_all_forecasts(
        file_date, 
        states, 
        local_case_dict, 
        confidence_level = "both",
    )
    plot_all_forecasts(
        file_date, 
        states, 
        local_case_dict, 
        zoom = true, 
        confidence_level = "both",
    )
    plot_all_forecasts(
        file_date, 
        states, 
        local_case_dict, 
        confidence_level = "50",
    )
    plot_all_forecasts(
        file_date, 
        states, 
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