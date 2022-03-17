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
    dates, 
    local_case_dict; 
    zoom = false,
    confidence_level = "both",
    truncation_days = 7,
)
    """
    Plots all the forecasts and saves the result as a pdf. 
    """
    
    forecast_origin = string(Dates.Date(file_date) - Dates.Day(truncation_days))
    
    # read in the sim results and kept TP's
    sim_all_states = CSV.read(
        "results/UoA_forecast_output/" * forecast_origin * "/UoA_samples_" * forecast_origin * ".csv", 
        DataFrame,
    )
    
    # read in the case data 
    case_dates = collect(dates)
    
    # initialise the plot 
    fig = plot( 
        layout = (4, 2), 
        dpi = 200, 
        size = (750, 1200), 
        link = :x, 
        framestyle = :box, 
        margin = 3mm,
        rightmargin = 4mm,
        legend = :outerright
    )
    
    for (sp, state) in enumerate(states)
        
        # get relevant columns of the dataframe and merge to matrix
        D = Matrix(sim_all_states[sim_all_states.state .== state, 3:end - 1])
        # if anything in D is missing, it means the sims didn't work so 0 out those entries
        D[ismissing.(D)] .= 0
        D = Matrix{Int}(D)
        
        onset_dates = sim_all_states[sim_all_states.state .== state, "onset date"]
        df_D_summary = summarise_forecast_for_plotting(D)

        local_cases = local_case_dict[state][dates .>= onset_dates[1]]
        # boolean for correct case dates 
        case_dates_ind = [d ∈ onset_dates ? true : false for d in case_dates]

        onset_dates_lims = Dates.value.(
            [onset_dates[1], onset_dates[end]]
        )
        
        if confidence_level == "50" || confidence_level == "both"
            plot!(
                fig,
                title = state, 
                subplot = sp, 
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
                subplot = sp,
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
                subplot = sp, 
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
                subplot = sp,
                onset_dates, 
                [df_D_summary[!, "median"] df_D_summary[!, "median"]], 
                fillrange = [df_D_summary[!, "bottom"] df_D_summary[!, "top"]],
                fillalpha = 0.2, 
                c = 1, 
                label = ["95%" nothing]
            )
        end
        
        plot!(
            fig, 
            subplot = sp, 
            case_dates[case_dates_ind], 
            local_cases, 
            lc = "black",
            label = false,
        )
        
        # plot the last date used for conditioning the simulations 
        vline!(
            fig, 
            subplot = sp,
            [Dates.Date(file_date) - Dates.Day(truncation_days)], 
            lc = :gray,
            ls = :dash,
            label = false,
        )
        
        # adjust limits to make plots a little nicer to look at 
        xlims!(fig, subplot = sp, onset_dates_lims...)
        if zoom 
            ylims!(fig, subplot = sp, 0, 1.25 * maximum(local_cases))
        end
        
    end

    dir_name = joinpath("figs", "case_forecasts", forecast_origin)
    if !ispath(dir_name)
        mkpath(dir_name)
    end
    
    if zoom 
        savefig(
            fig, 
            dir_name *
            "/UoA_forecast_" *
            forecast_origin *
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
            forecast_origin *
            "_" *
            confidence_level *
            "_intervals.pdf"
        )
    end
        
    return nothing 
    
end


function plot_all_forecast_intervals(
    file_date, 
    states, 
    dates, 
    local_case_dict; 
    truncation_days = 7,
)
    """
    Simple wrapper function to plot the forecasts with various zoom and confidence levels. 
    """
    
    plot_all_forecasts(
        file_date, 
        states, 
        dates,
        local_case_dict, 
        confidence_level = "both",
    )
    plot_all_forecasts(
        file_date, 
        states, 
        dates,
        local_case_dict, 
        zoom = true, 
        confidence_level = "both",
    )
    plot_all_forecasts(
        file_date, 
        states, 
        dates,
        local_case_dict, 
        confidence_level = "50",
    )
    plot_all_forecasts(
        file_date, 
        states, 
        dates,
        local_case_dict, 
        confidence_level = "95",
    )
    
    forecast_origin = string(Dates.Date(file_date) - Dates.Day(truncation_days))
    dir_name = "figs/case_forecasts/" * forecast_origin * "/"
    file_name_tmp = "UoA_forecast_"
    
    pdf_filenames = [
        dir_name * file_name_tmp * forecast_origin * "_zoomed_both_intervals.pdf",
        dir_name * file_name_tmp * forecast_origin * "_both_intervals.pdf",
        dir_name * file_name_tmp * forecast_origin * "_50_intervals.pdf",
    ]
    # merge the pdfs and delete the files
    merge_pdfs(
        pdf_filenames, 
        dir_name * file_name_tmp * forecast_origin * ".pdf", 
        cleanup=true,
    )

    return nothing    
    
end