"""
This file is used to test the simulation. This is just the simulate_states code unwrapped 
so will be a little less optimal than when directly run but should enable an easier 
observation of correctness. 
"""

using Revise
using Distributed
# using CairoMakie
using Plots
using Chain
using ProfileView

##### PROFILING CODE 

include("simulate_states.jl")

D = []
U = []
TP_local = []

# parameters to pass to the main function
file_date = "2022-03-29"

# set seed for consistent plots (NOTE: this is not useful when multithreading 
# enabled as we use separate seeds but the simulation pool should handle that)
rng = Random.Xoshiro(2022)

jurisdiction_assumptions = JurisdictionAssumptions()
    
# get the latest onset date
latest_start_date = Dates.Date(
    maximum(v for v in values(jurisdiction_assumptions.simulation_start_dates))
)

omicron_dominant_date = "2021-12-15"
(dates, local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
last_date_in_data = dates[end]
forecast_end_date = last_date_in_data + Dates.Day(35)
# create vector for dates
onset_dates = latest_start_date:Dates.Day(1):forecast_end_date
# add a small truncation to the simulations as we don't trust the most recent data
truncation_days = 7
# states to simulate
state = "SA"
nsims = 100
p_detect_omicron = 0.5
forecast_start_date = Dates.Date(
    jurisdiction_assumptions.simulation_start_dates[state]
)

# named tuple for initial conditions
D0 = jurisdiction_assumptions.initial_conditions[state]
N = jurisdiction_assumptions.pop_sizes[state]

# get the observed cases
cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
local_cases = local_case_dict[state][dates .>= forecast_start_date]
# cutoff the last bit of the local cases
import_cases = import_case_dict[state]
local_cases = Int.(local_cases[begin:end-truncation_days+1])
import_cases = Int.(import_cases)

ProfileView.@profview (D, U, TP_local, scale_factor, Z_historical) = simulate_branching_process(
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
    adjust_TP = false,
)

##### NORMAL RUNNNING
using Revise
using Distributed
# using CairoMakie
using Plots
using Chain
using ProfileView

include("simulate_states.jl")

# parameters to pass to the main function
file_date = "2022-03-29"

# set seed for consistent plots (NOTE: this is not useful when multithreading 
# enabled as we use separate seeds but the simulation pool should handle that)
rng = Random.Xoshiro(2022)

jurisdiction_assumptions = JurisdictionAssumptions()
    
# get the latest onset date
latest_start_date = Dates.Date(
    maximum(v for v in values(jurisdiction_assumptions.simulation_start_dates))
)

omicron_start_date = "2021-11-15"
omicron_dominant_date = "2021-12-15"
(dates, local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
last_date_in_data = dates[end]
forecast_end_date = last_date_in_data + Dates.Day(35)
# create vector for dates
onset_dates = latest_start_date:Dates.Day(1):forecast_end_date
# add a small truncation to the simulations as we don't trust the most recent data
truncation_days = 7
# states to simulate
state = "SA"
nsims = 1000
forecast_start_date = Dates.Date(
    jurisdiction_assumptions.simulation_start_dates[state]
)

# named tuple for initial conditions
D0 = jurisdiction_assumptions.initial_conditions[state]
N = jurisdiction_assumptions.pop_sizes[state]

# get the observed cases
cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
local_cases = local_case_dict[state][dates .>= forecast_start_date]
# cutoff the last bit of the local cases
import_cases = import_case_dict[state]
local_cases = Int.(local_cases[begin:end-truncation_days+1])
import_cases = Int.(import_cases)

plot(local_cases)
# ylims!(0, 100)

include("simulate_states.jl")

(D, U) = simulate_branching_process(
    D0,
    N,
    nsims,
    local_cases,
    import_cases,
    cases_pre_forecast,
    forecast_start_date,
    file_date,
    state,
    adjust_TP = true,
)

# save_simulations(
#     D,
#     state,
#     file_date,
#     onset_dates,
#     rng,
# )

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end]
    D_local = D[:, 1, :] + D[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    D_local_mean = mean(D_local, dims = 2)

    d_start = Dates.Date(forecast_start_date)
    Δd = Dates.Day(1)
    d_end = Dates.Date(forecast_start_date) + Δd * (size(D_local, 1) - 1)
    D_local_dates = d_start:Δd:d_end
    
    f = plot()
    # plot!(f, dates[dates .>= forecast_start_date], local_cases, legend = false, linealpha = 1, lc = 1)
    # plot!(f, Z_historical[36:end, :], legend = false, lc = 3, linealpha = 0.5)
    plot!(f, D_local_dates, D_local, legend = false, lc = 1, linealpha = 0.5)
    # plot!(f, sims_old[!, "onset date"], sims_old_mat, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, dates[dates .>= forecast_start_date][begin:end-truncation_days+1], local_cases[begin:end-truncation_days+1], legend = false, lc = :black, lw = 2)
    # plot!(f, local_cases2.date_onset, local_cases2.count, legend = false, linealpha = 1, lc = 1)
    plot!(f, D_local_dates, D_local_median, legend = false, lc = 2, lw = 2)
    vline!(f, [Dates.Date("2021-11-15")], ls = :dash, lc = :black)
    # xlims!(f, 0, length(local_cases) + 35)
    ylims!(f, 0, 8000)
    xlims!(f, Dates.value(Dates.Date("2021-12-15")), Dates.value(Dates.Date("2022-03-31")))
    # ylims!(f, 0, 500)
end

savefig(state * "_forecast_using_reff.pdf")

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    import_cases = import_case_dict[state][dates .>= forecast_start_date]
    import_cases = import_cases[begin:end]
    D_import = D[:, 3, :]
    D_import_median = median(D_import, dims = 2)
    D_import_mean = mean(D_import, dims = 2)

    d_start = Dates.Date(forecast_start_date)
    Δd = Dates.Day(1)
    d_end = Dates.Date(forecast_start_date) + Δd * (size(D_import, 1) - 1)
    D_import_dates = d_start:Δd:d_end
        
    f = plot()
    # plot!(f, dates[dates .>= forecast_start_date], import_cases, legend = false, linealpha = 1, lc = 1)
    # plot!(f, Z_historical[36:end, :], legend = false, lc = 3, linealpha = 0.5)
    plot!(f, D_import_dates, D_import, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, dates[dates .>= forecast_start_date][begin:end-truncation_days+1], import_cases[begin:end-truncation_days+1], legend = false, linealpha = 1, lc = 1)
    # plot!(f, import_cases2.date_onset, import_cases2.count, legend = false, linealpha = 1, lc = 1)
    plot!(f, D_import_dates, D_import_median, legend = false, lc = 3)
    vline!(f, [Dates.Date("2021-12-15")], ls = :dash, lc = :black)
    # xlims!(f, 0, length(local_cases) + 35)
    ylims!(f, 0, 20)
    # xlims!(f, Dates.value(Dates.Date("2021-12-15")), Dates.value(Dates.Date("2022-03-31")))
    # ylims!(f, 0, 500)
end

save_simulations(
    D,
    state,
    file_date,
    onset_dates,
    rng,
    truncation_days = truncation_days,
)

let 
    plot(scale_factor, lc = 1, alpha = 0.05, legend = false)
end

# plot for assessing the overall forecast fits

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end-truncation_days]
    
    D_local = D[:, 1, :] + D[:, 2, :]
    U_local = U[:, 1, :] + U[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    U_local_median = median(U_local, dims = 2)
    
    import_cases = import_case_dict[state][dates .>= forecast_start_date]
    import_cases = import_cases[begin:end-truncation_days]
    D_import = D[:, 3, :]
    D_import_median = median(D_import, dims = 2)
    U_import = U[:, 3, :]
    U_import_median = median(U_import, dims = 2)
    
    f = plot(layout = (3, 1))
    plot!(f, subplot = 1, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 1, D_local, legend = false, lc = 2, linealpha = 0.5)
    # plot!(f, subplot = 1, U_local, legend = false, lc = 4, linealpha = 0.5)
    # plot!(f, subplot = 1, scale_factor[36:end, :], legend = false, lc = 4, linealpha = 0.5)
    plot!(f, subplot = 1, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 1, D_local_median, legend = false, lc = 3)
    # plot!(f, subplot = 1, U_local_median, legend = false, lc = 5)
    vline!(f, subplot = 1, [19], lc = :black)
    xlims!(f, subplot = 1, 0, length(local_cases) + 35)
    ylims!(f, subplot = 1, 0, maximum(local_cases) * 3)
    # ylims!(f, subplot = 1, 0, 5000)
    # ylims!(f, subplot = 1, 0, 50000)
    # ylims!(0, 30000)
    # ylims!(0, 50000)
    # ylims!(0, 3000)
    # ylims!(0, 15000)
    plot!(f, subplot = 2, import_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 2, D_import, legend = false, lc = 2, linealpha = 0.5)
    # plot!(f, subplot = 2, U_import, legend = false, lc = 4, linealpha = 0.5)
    plot!(f, subplot = 2, import_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 2, D_import_median, legend = false, lc = 3)
    # plot!(f, subplot = 2, U_import_median, legend = false, lc = 5)
    # plot!(D_import_mean, legend = false, lc = 3)
    xlims!(f, subplot = 2, 0, length(import_cases) + 35)
    # xlims!(length(local_cases) - 60, length(local_cases) + 35)
    ylims!(f, subplot = 2, 0, maximum(import_cases) * 1.5)
    # ylims!(f, subplot = 2, 0, 10)
    # ylims!(0, 30000)
    # ylims!(0, 50000)
    # ylims!(0, 3000)
    # ylims!(0, 15000)
    plot!(f, subplot = 3, TP_local, legend = false, linealpha = 0.5, lc = 1)
    xlims!(f, subplot = 3, 0, length(local_cases) + 35)
    hline!(f, subplot = 3, [1.0], lc = :black, ls = :dash)
    ylims!(f, subplot = 3, 0, 3)
end

# plots for assessing the local case forecast specifically
let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end-truncation_days]
    D_local = D[:, 1, :] + D[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    D_local_mean = mean(D_local, dims = 2)
    
    f = plot()
    plot!(f, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, Z_historical[36:end, :], legend = false, lc = 3, linealpha = 0.5)
    plot!(f, median(Z_historical[36:end, :], dims = 2), lc = :black, lw = 4)
    plot!(f, D_local, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, scale_factor[36:end, :], legend = false, lc = 4, linealpha = 0.5)
    plot!(f, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, D_local_median, legend = false, lc = 3)
    xlims!(f, 0, length(local_cases) + 35)
    ylims!(f, 0, 15000)
end

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end]
    D_local = D[:, 1, :] + D[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    D_local_mean = mean(D_local, dims = 2)

    d_start = Dates.Date(forecast_start_date)
    Δd = Dates.Day(1)
    d_end = Dates.Date(forecast_start_date) + Δd * (size(D_local, 1) - 1)
    D_local_dates = d_start:Δd:d_end
        
    f = plot()
    # plot!(f, dates[dates .>= forecast_start_date], local_cases, legend = false, linealpha = 1, lc = 1)
    # plot!(f, Z_historical[36:end, :], legend = false, lc = 3, linealpha = 0.5)
    plot!(f, D_local_dates, D_local, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, dates[dates .>= forecast_start_date][begin:end-truncation_days+1], local_cases[begin:end-truncation_days+1], legend = false, linealpha = 1, lc = 1)
    # plot!(f, local_cases2.date_onset, local_cases2.count, legend = false, linealpha = 1, lc = 1)
    plot!(f, D_local_dates, D_local_median, legend = false, lc = 3)
    vline!(f, [Dates.Date("2021-12-15")], ls = :dash, lc = :black)
    # xlims!(f, 0, length(local_cases) + 35)
    ylims!(f, 0, 7000)
    # xlims!(f, Dates.value(Dates.Date("2021-12-15")), Dates.value(Dates.Date("2022-03-31")))
    # ylims!(f, 0, 500)
end

forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
local_cases = local_case_dict[state]
# local_cases = local_cases[begin:end-truncation_days]

D_local = D[:, 1, :] + D[:, 2, :]
D_local_median = median(D_local, dims = 2)
D_local_low = quantile.(eachrow(D_local), 0.25)
D_local_high = quantile.(eachrow(D_local), 0.75)
D_local_bottom = quantile.(eachrow(D_local), 0.05)
D_local_top = quantile.(eachrow(D_local), 0.95)
D_local_mean = mean(D_local, dims = 2)

let 
    fig = Figure(size = (800, 1200))  

    axs = [
        Axis(fig[i, j])
        for i in 1:4, j in 1:2
    ]
    
    for i in 1:4
        linkyaxes!(axs[i, 1], axs[i, 2])
        linkxaxes!(axs[i, 1], axs[i, 2])
    end
    
    x = 0:length(forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1)) - 1
    
    for ax in axs
        ax.xticks = (
            0:21:length(forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1)) - 1,
            string.(forecast_start_date:Dates.Day(21):forecast_start_date + Dates.Day(size(D, 1)-1))
        )
        ax.xticklabelrotation = π/4
        lines!(
            ax,
            # forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1),
            x,
            vec(D_local_median), 
            color = :blue,
        )
        band!(
            ax,
            # forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1),
            x,
            D_local_low,
            D_local_high, 
            color = (:skyblue, 0.5),
        )
        band!(
            ax,
            # forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1),
            x,
            D_local_bottom,
            D_local_top, 
            color = (:skyblue, 0.3),
        )
        lines!(
            ax,
            # Dates.Date("2020-03-01"):Dates.Day(1):Dates.Date("2020-03-01") + Dates.Day(length(local_cases)-1),
            x[1:length(local_cases)],
            local_cases, 
            color = :black,
        )
    end

    fig
end
# plot!(
#     f, 
#     D_local, 
#     forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1),
#     legend = false, 
#     lc = 3, 
#     linealpha = 0.5
# )
# plot!(
#     f, 
#     legend = false, 
#     linealpha = 1, 
#     lc = 1
# )

# x1 = forecast_start_date
# x2 = forecast_start_date + Dates.Day(size(D, 1)-1)

# # plot!(f, (1:length(local_cases)) .+ 7, local_cases, legend = false, linealpha = 1, lc = 1)
# # plot!(f, D_local_median, legend = false, lc = 3)
# vline!(f, [Dates.Date("2021-12-01")], lc = :black)
# xlims!((x1, x2))
# ylims!(f, 0, 200)
# ylims!(f, 0, 5000)
forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
local_cases = local_case_dict[state][dates .>= forecast_start_date]
local_cases = local_cases[begin:end-truncation_days]

D_local = D[:, 1, :] + D[:, 2, :]
D_observed = D_local[begin:length(local_cases), :]
local_cases_mat = repeat(local_cases, 1, size(D_observed, 2))
error = vec(sum((local_cases_mat - D_observed) .^ 2, dims = 1))
idx = error .< quantile(error, 0.05)

D_local_median = median(D_local[:, idx], dims = 2)

let
    plot(local_cases, legend = false, linealpha = 1, lc = 1)
    # plot!(Lₜ, legend = false, linealpha = 1, lc = 1, ls = :dash)
    # plot!(Uₜ, legend = false, linealpha = 1, lc = 1, ls = :dash)
    # plot!(D_med, legend = false, linealpha = 1)
    # plot!(D_local_tmp, legend = false, lc = :grey, linealpha = 0.5)
    plot!(D_local[:, idx], legend = false, lc = 2, linealpha = 0.5)
    plot!(D_local_median, legend = false, lc = 3, linealpha = 0.5)
    # vline!([length(local_cases)], lc = "black", ls = :dash)
    plot!(local_cases, legend = false, linealpha = 1, lc = 1)
    # plot!(D_local_med, legend = false, lc = 2)
    xlims!(1, length(local_cases) + 35)
    ylims!(0, 6000)
end

dir_name = joinpath("tmp_plots", "case_forecasts")
if !ispath(dir_name)
    mkpath(dir_name)
end

savefig(
    dir_name * 
    "/" * 
    states_to_run[1] * 
    ".pdf"
)

savefig(
    dir_name * 
    "/" * 
    states_to_run[1] * 
    "_zoom.pdf"
)

# # merge all the simulation and TP files for states into single CSV
# merge_simulation_files(file_date)    
# merge_TP_files(file_date)
# plot_all_forecast_intervals(file_date, states_to_run, local_case_dict)

#

x = 0:10
C = 1 .+ sin.(x)
plot(x, C)

Mt = zero(similar(C))

for t in range(1, length(Mt))
    
    if t < length(Mt) - 2
        Mt[t] = (C[t+2] - C[t]) / 2
    else
        Mt[t] = (C[t] - C[t-2]) / 2
    end
    
end

plot(x, C)
plot!(x, Mt)

Lt = zero(similar(C))
Ut = zero(similar(C))

ϵ = 0.5
u = 2
ℓ = 0.5

for t in range(1, length(Lt))

    if Mt[t] < -ϵ
        Lt[t] = ℓ / 3 * sum(C[t:min(length(Lt), t+2)])
        Ut[t] = u / 3 * sum(C[max(1, t-2):t])
    elseif Mt[t] > ϵ
        Lt[t] = ℓ / 3 * sum(C[max(1, t-2):t])
        Ut[t] = u / 3 * sum(C[t:min(length(Lt), t+2)])
    else
        Lt[t] = ℓ / 3 * sum(C[max(1, t-1):min(length(Lt), t+1)])
        Ut[t] = u / 3 * sum(C[max(1, t-1):min(length(Lt), t+1)])
    end
    
end

plot(x, C, lc = 1, legend = false)
plot!(x, Lt, lc = 1, ls = :dash)
plot!(x, Ut, lc = 1, ls = :dash)

################ 

k = 0.6
(shape_gen, scale_gen) = (2.75, 1.00)

(local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
dates = local_case_dict["date"]
last_date_in_data = dates[end]
forecast_end_date = last_date_in_data + Dates.Day(35)

# create vector for dates
onset_dates = latest_start_date:Dates.Day(1):forecast_end_date

# add a small truncation to the simulations as we don't trust the most recent data
truncation_days = 7

# states to simulate
state = "QLD"
nsims = 10000

forecast_start_date = Dates.Date(
    jurisdiction_assumptions.simulation_start_dates[state]
)

local_cases = local_case_dict[state]
import_cases = import_case_dict[state]

local_cases_file = CSV.read(
    "data/local_cases_20220222-122116.csv", 
    DataFrame
)

plot(dates, local_cases)
plot!(
    local_cases_file.date_onset[local_cases_file.state .== state], 
    local_cases_file.count[local_cases_file.state .== state]
)
xlims!((Dates.Date("2021-11-15"), Dates.Date("2022-02-22")))


df = CSV.read("results/posterior_sample_2022-02-15.csv", DataFrame)
df2 = CSV.read("results/posterior_sample_2022-02-22.csv", DataFrame)

#####################

f(t, m0, m1, r, τ) = m0 + (m1 - m0) / (1 + exp(-r * (t - τ)))

m0 = 0.075
m1 = 0.75
τ = 30

f1(t, r) = f(t, m0, m1, r, τ)

t = range(0, 100, step = 0.2)
plot(t, f1.(t, 0.05))

#####

df1 = CSV.read(
    "results/2022-02-22/25_case_ascertainment/soc_mob_R2022-02-22.csv", 
    DataFrame
)
df2 = CSV.read(
    "results/2022-02-22/38_case_ascertainment/soc_mob_R2022-02-22.csv", 
    DataFrame
)
df3 = CSV.read(
    "results/2022-02-22/50_case_ascertainment/soc_mob_R2022-02-22.csv", 
    DataFrame
)

state = "SA"
df1_local = filter(:type => ==("R_L"), df1)
df1_local = filter(:state => ==(state), df1_local)
df2_local = filter(:type => ==("R_L"), df2)
df2_local = filter(:state => ==(state), df2_local)
df3_local = filter(:type => ==("R_L"), df3)
df3_local = filter(:state => ==(state), df3_local)

f = plot(dpi = 92, legend = :outerright, legendfontsize=8)
plot!(
    f, 
    df1_local.date, 
    df1_local.median, 
    ribbon = (
        df1_local.median - df1_local.bottom, 
        df1_local.top - df1_local.median, 
    ),
    group = df1_local.state, 
    lc = 1, 
    label = "25%",
    fillalpha = 0.2,
)
plot!(
    f, 
    df2_local.date, 
    df2_local.median, 
    ribbon = (
        df2_local.median - df2_local.bottom, 
        df2_local.top - df2_local.median, 
    ),
    group = df2_local.state, 
    lc = 2, 
    label = "37.5%",
    fillalpha = 0.2,
)
plot!(
    f, 
    df3_local.date, 
    df3_local.median, 
    ribbon = (
        df3_local.median - df3_local.bottom, 
        df3_local.top - df3_local.median, 
    ),
    group = df3_local.state, 
    lc = 3, 
    label = "50%",
    fillalpha = 0.2,
)
hline!(f, [1.0], lc = :black, ls = :dash)
xlims!((Dates.Date("2021-12-01"), maximum(df3_local.date)))
ylims!((0.5, 3.0))
display(f)

savefig(f, "local_TP_differing_CA.pdf")

#####

df1 = CSV.read("results/2022-03-01/50_case_ascertainment/soc_mob_R2022-03-01.csv", DataFrame) 
df2 = CSV.read("results/2022-03-01/50_case_ascertainment/soc_mob_R_adjusted2022-03-01.csv", DataFrame) 
df1 = filter(:type => ==("R_L"), df1)
df2 = filter(:type => ==("R_L"), df2)
# df2 = CSV.read("data/r_eff_12_local_samples.csv", DataFrame)
# df2 = CSV.read("data/r_eff_1_local_samples.csv", DataFrame)
# df3 = CSV.read("results/EpyReff/Reff2022-03-01tau_5.csv", DataFrame)


fig = plot(layout = (4, 2), size = (800, 800))
plot!(fig, df1.date, df1.median, group = df1.state)
plot!(fig, df2.date, df2.median, group = df2.state)
# plot!(fig, df2.date, median(Matrix(df2[:, 4:end]), dims = 2)[:], group = df2.state)
# plot!(fig, df3.INFECTION_DATES, df3.median, group = df3.STATE)
xlims!(Dates.Date.(("2021-11-01", df1.date[end])))
ylims!(0, 3)

#####

nsims = 100000
t1 = [sample_times(0, omicron = true) for _ in 1:nsims]
t1_inf = [t1[i][1] for i in eachindex(t1)]
t1_ons = [t1[i][2] - t1[i][1] for i in eachindex(t1)]
t2_inf = rand(Gamma(1.58, 1.32), nsims)
t2_ons = rand(Gamma(3.33, 1.34), nsims)
t2_inf = ceil.(Int, t2_inf)
t2_ons = ceil.(Int, t2_ons)

using KernelDensity
using StatsPlots 
histogram(t1_inf, alpha = 0.2, normalize = :probability)
histogram!(t2_inf, alpha = 0.2, normalize = :probability)
plot!(truncated(Gamma(1.58, 1.32), 1, 21), alpha = 0.2)
histogram(t1_ons .- 1, alpha = 0.2, normalize = :probability)
histogram!(t2_ons .- 1, alpha = 0.2, normalize = :probability)
plot!(truncated(Gamma(3.33, 1.34), 1, 21), alpha = 0.2)

p = [cdf(Truncated(Gamma(1.58, 1.32), 0, 21), x) for x in 1:21]

#####

# (TP_indices, TP_local, TP_import) = create_state_TP_matrices(
#     forecast_start_date, 
#     file_date, 
#     state; 
#     p_detect_omicron = p_detect_omicron,
#     adjust_TP = false,
# )

# TP_local = TP_local[TP_indices .>= 0, :]
forecast_end_date = Dates.Date(forecast_start_date) + Dates.Day(size(TP_local, 1) - 1)
date_range = Dates.Date(forecast_start_date):Dates.Day(1):Dates.Date(forecast_end_date)

df1 = CSV.read("results/2022-03-08/50_case_ascertainment/soc_mob_R_adjusted2022-03-08.csv", DataFrame) 
df1 = @chain df1 begin 
    filter(:type => ==("R_L"), _)
    filter(:state => ==("SA"), _)
end

df2 = CSV.read("results/2022-03-08/50_case_ascertainment_delta/soc_mob_R_adjusted2022-03-08.csv", DataFrame) 
df2 = @chain df2 begin 
    filter(:type => ==("R_L"), _)
    filter(:state => ==("SA"), _)
end

fig = plot(legend = false)
plot!(fig, df1.date, df1.median, lc = 1)
plot!(fig, df1.date, df1.bottom, lc = 1)
plot!(fig, df1.date, df1.top, lc = 1)
plot!(fig, df2.date, df2.median, lc = 2)
plot!(fig, df2.date, df2.bottom, lc = 2)
plot!(fig, df2.date, df2.top, lc = 2)
xlims!((Dates.Date("2021-12-01"), Dates.Date("2022-01-01")))
ylims!(0.5,2.5)

#####

df = CSV.read(
    "./data/interim_linelist_2022-03-01.csv", 
    DataFrame
)

df2 = @chain df begin
    filter(:date_onset => !=("NA"), _)
    filter(:import_status => ==("local"), _)
end

df2.date_onset = Dates.Date.(df2.date_onset)

df3 = @chain df2 begin
    filter(:date_onset => >=(Dates.Date("2021-12-01")), _)
    filter(:date_onset => <=(Dates.Date("2022-01-01")), _)
end

time_lag = Dates.value.(df2.date_confirmation - df2.date_onset)
# time_lag = Dates.value.(df3.date_confirmation - df3.date_onset)
using KernelDensity
using StatsPlots

histogram(time_lag, normalize = :probability)
plot!(1 + Gamma(1.28, 2.31))
xlims!(0, 25)

plot(Gamma(1.28, 2.31))
plot!(Gamma(2.33, 1.35))


df1 = CSV.read("results/EpyReff/Reff_delta_samples2022-03-08tau_5.csv", DataFrame)
df2 = CSV.read("results/EpyReff/Reff_omicron_samples2022-03-08tau_5.csv", DataFrame)
df3 = CSV.read("results/EpyReff/Reff_samples2022-03-08tau_5.csv", DataFrame)

df1_SA = @chain df1 begin
    subset(
        :STATE => (ByRow(==("SA"))), 
    )
end

df2_SA = @chain df2 begin
    subset(
        :STATE => (ByRow(==("SA"))), 
    )
end

df3_SA = @chain df3 begin
    subset(
        :STATE => (ByRow(==("SA"))), 
    )
end

# df3_SA = vcat(
#     df1_SA[df1_SA.INFECTION_DATES .<= Dates.Date("2021-12-15"), :], 
#     df2_SA[df2_SA.INFECTION_DATES .> Dates.Date("2021-12-15"), :], 
# )

fig = plot(legend = true)
plot!(fig, df1_SA.INFECTION_DATES, median(Matrix(df1_SA[:, begin:end-2]), dims = 2), lc = 1, label = "Delta")
plot!(fig, df2_SA.INFECTION_DATES, median(Matrix(df2_SA[:, begin:end-2]), dims = 2), lc = 2, label = "Omicron")
# plot!(fig, df3_SA.INFECTION_DATES, median(Matrix(df3_SA[:, begin:end-2]), dims = 2), lc = 2, label = "Omicron")
vline!(fig, [Dates.Date("2021-12-15")], lc = :black, ls = :dash, label = "changeover point \n 15/12/2021")
xlims!((Dates.Date("2021-11-01"), Dates.Date("2022-03-01")))
ylims!((0.5, 2.5))

using StatsPlots
GI_delta = Truncated(Gamma(2.7, 1.0), 0, 21)
GI_omicron = Truncated(Gamma(1.58, 1.32), 0, 21)
IP_delta = Truncated(Gamma(5.807, 0.948), 0, 21)
IP_omicron = Truncated(Gamma(3.33, 1.34), 0, 21)

fig = plot(layout = (2, 1))
plot!(fig, subplot = 1, GI_delta)
plot!(fig, subplot = 1, GI_omicron)
plot!(fig, subplot = 2, IP_delta)
plot!(fig, subplot = 2, IP_omicron)
xlims!(0, 10)

a = zeros(100000)
for i in eachindex(a)
    x = rand(GI_delta) - 1
    while x < 0
        x = rand(GI_delta) - 1
    end
    
    a[i] = x    
end

###################

start_date = forecast_start_date
end_date = "2022-04-05"

# date vectors to indicate which period we're in
CAR_normal_before = Dates.Date(start_date):Dates.Day(1):Dates.Date("2021-12-08")
CAR_decrease = Dates.Date("2021-12-09"):Dates.Day(1):Dates.Date("2021-12-15")
CAR_normal_after = Dates.Date("2021-12-16"):Dates.Day(1):Dates.Date(end_date)
# all dates
CAR_dates = [
    CAR_normal_before;
    CAR_decrease;
    CAR_normal_after
]

# initialise the 
CAR = 0.75 * ones(Float64, length(CAR_dates))
idx = (CAR_dates .>= CAR_decrease[begin]) .& (CAR_dates .<= CAR_decrease[end])
CAR[idx] = collect(range(0.75, 0.5, 7))
idx = CAR_dates .> CAR_decrease[end]
CAR[idx] .= 0.5

p_symp = zeros(Float64, length(CAR_dates))
p_detect_given_asymp = zeros(Float64, length(CAR_dates))
p_detect_given_symp = zeros(Float64, length(CAR_dates))
p_symp_given_detect = zeros(Float64, length(CAR_dates))
p_detect_import = zeros(Float64, length(CAR_dates))
k = zeros(Float64, length(CAR_dates))

for (i, d) in enumerate(CAR_dates)
    p_detect = CAR[i]
    
    if d <= Dates.Date("2021-12-09")
        p_symp[i] = 0.7
        # p(d) = p(d|s)p(s) + p(d|a)p(a) and taking p(d|s)/p(d|a) ~ 4 yields the following
        p_detect_given_symp[i] = 0.9375
        p_detect_given_asymp[i] = 0.3125
        p_detect_import[i] = 0.98
    else
        p_symp[i] = 0.4
        # p(d) = p(d|s)p(s) + p(d|a)p(a) and taking p(d|a)/p(d|s) = r 
        # where we have reduced r to reflect heightend chance of detecting asymptomatics due
        # to much larger case numbers and testing.
        
        p_detect_given_asymp[i] = max(0.3, (p_detect - p_symp[i]) / (1 - p_symp[i]), p_detect_given_asymp[i-1])
        p_detect_given_symp[i] = (
            (p_detect - p_detect_given_asymp[i] * (1 - p_symp[i])) / p_symp[i]
        )
        p_detect_import[i] = p_detect
    end
    
    if d <= Dates.Date("2021-12-01")
        k[i] = 0.15
    else
        k[i] = 0.6
    end   
    
    p_symp_given_detect[i] = p_detect_given_symp[i] * p_symp[i] / p_detect
    
end

calculated_CAR = p_symp .* p_detect_given_symp + (1 .- p_symp) .* p_detect_given_asymp

plot(p_symp, label = "p_symp")
plot!(calculated_CAR, ls = :dash, label = "calculated_CAR")
plot!(CAR,ls = :dash, label = "CAR")
plot!(p_detect_given_symp, label = "p_detect_given_symp")
plot!(p_detect_given_asymp, label = "p_detect_given_asymp")
ylims!(0, 1)

γ = 0.5     # relative infectiousness of asymptomatic
α_s = 1 ./ (p_symp .+ γ * (1 .- p_symp))
α_a = γ * α_s



