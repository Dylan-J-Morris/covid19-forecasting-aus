"""
Author: Dylan Morris
Date: 10 Feb 2022 at 6:15:09 pm
Description: 
Use this script to quickly plot diagnostics of a chain. This could involve scatter-plots 
to assess correlations between parameters, traceplots to check for divergent behaviours and
also just assessing the overall reasonableness of the posterior samples. 
"""

let
    # for dataframe handling 
    using DataFrames
    using CSV
    using Chain

    # nice plotting 
    using StatsPlots
    using PlotThemes
    using Plots
    using LaTeXStrings

    # analysis/commonly used
    using Statistics
    using StatsBase 
    using Dates
    using KernelDensity
    using Random
    using Distributions
    using BenchmarkTools
    using ProgressBars
end

gr()

default(
	linewidth=1.5, 
	label=nothing, 
	framestyle=:box,
)
scalefontsizes(0.95)

##

file_date = ARGS[1]

file_name = (
    "results/"
    * file_date 
    * "/50_case_ascertainment/posterior_sample_"
    * file_date 
    * ".csv"
)

samples = CSV.read(file_name, DataFrame)

dir_name = joinpath("figs", "posterior_summary", file_date)
if !ispath(dir_name)
    mkpath(dir_name)
end

let
    
    num_samples = nrow(samples)
    # we run 4 independent chains and so this just ensures that we get the right number of samples 
    # per chain 
    num_each_chain = num_samples ÷ 4
    
    sp = 1
    page = 1

    names_to_plot = names(samples)

    fig = plot(
        layout = (8,5),
        dpi=200, 
        size=(1000,1200), 
        # link=:x, 
        framestyle=:minimal, 
        legend = false,
    )

    for name in ProgressBar(names_to_plot)
        if sp == 1
            fig = plot(
                layout = (8,5),
                dpi=200, 
                size=(1000,1200), 
                # link=:x, 
                framestyle=:minimal, 
                legend = false,
            )
        end
        # plot!(fig, subplot = sp, samples[!,name])
        for i in 1:4
            plot!(fig, subplot = sp, kde(samples[(1 + (i - 1) * num_each_chain):(i * num_each_chain),name]))
        end
        xlabel!(fig, subplot = sp, name)
        sp += 1
        if sp == 8*5 + 1 || sp == length(names_to_plot) + 1
            sp = 1 
            savefig(fig, dir_name*"/page"*string(page)*".pdf")
            page += 1
        end
    end

    using PDFmerger

    pdf_filenames = [
        dir_name*"/page"*string(p)*".pdf" for p in 1:page-1
    ]

    merge_pdfs(
        pdf_filenames, 
        dir_name*"/kde.pdf",
        cleanup=true,
    )

    sp = 1
    page = 1

    names_to_plot = names(samples)

    fig = plot(
        layout = (8,5),
        dpi=200, 
        size=(1000,1200), 
        # link=:x, 
        framestyle=:minimal, 
        legend = false,
    )

    for name in ProgressBar(names_to_plot)
        if sp == 1
            fig = plot(
                layout = (8,5),
                dpi=200, 
                size=(1000,1200), 
                # link=:x, 
                framestyle=:minimal, 
                legend = false,
            )
        end
        # plot!(fig, subplot = sp, samples[!,name])
        for i in 1:4
            plot!(fig, subplot = sp, samples[(1 + (i - 1) * num_each_chain):(i * num_each_chain),name])
        end
        xlabel!(fig, subplot = sp, name)
        sp += 1
        if sp == 8*5 + 1 || sp == length(names_to_plot) + 1
            sp = 1 
            savefig(fig, dir_name*"/page"*string(page)*".pdf")
            page += 1
        end
    end

    using PDFmerger

    pdf_filenames = [
        dir_name*"/page"*string(p)*".pdf" for p in 1:page-1
    ]

    merge_pdfs(
        pdf_filenames, 
        dir_name*"/traceplots.pdf",
        cleanup=true,
    )
    
end