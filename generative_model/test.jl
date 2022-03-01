"""
This file is used to test the simulation. This is just the simulate_states code unwrapped 
so will be a little less optimal than when directly run but should enable an easier 
observation of correctness. 
"""

using Revise
using Distributed

parallel = true

if parallel 
    # add number of cores 
    if nprocs() == 1
        addprocs(6)
    end
end

include("simulate_states.jl")

D = []
U = []
TP_local = []

# parameters to pass to the main function
file_date = "2022-02-22"

# set seed for consistent plots (NOTE: this is not useful when multithreading 
# enabled as we use separate seeds but the simulation pool should handle that)
rng = Random.Xoshiro(2022)

jurisdiction_assumptions = JurisdictionAssumptions()
    
# get the latest onset date
latest_start_date = Dates.Date(
    maximum(v for v in values(jurisdiction_assumptions.simulation_start_dates))
)

omicron_dominant_date = "2021-12-15"
(local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
dates = local_case_dict["date"]
last_date_in_data = dates[end]
forecast_end_date = last_date_in_data + Dates.Day(35)

# create vector for dates
onset_dates = latest_start_date:Dates.Day(1):forecast_end_date

# add a small truncation to the simulations as we don't trust the most recent data
truncation_days = 7

# states to simulate
state = "SA"
nsims = 1000
p_detect_omicron = 0.25

forecast_start_date = Dates.Date(
    jurisdiction_assumptions.simulation_start_dates[state]
)

local_cases = local_case_dict[state]
import_cases = import_case_dict[state]

# named tuple for initial conditions
D0 = jurisdiction_assumptions.initial_conditions[state]
N = jurisdiction_assumptions.pop_sizes[state]

# get the observed cases
cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
local_cases = local_case_dict[state][dates .>= forecast_start_date]
# cutoff the last bit of the local cases
import_cases = import_case_dict[state]
local_cases = local_cases[begin:end-truncation_days]
import_cases = import_cases[begin:end-truncation_days]

include("simulate_states.jl")

(D, U, TP_local, scale_factor, Z_historical) = simulate_branching_process(
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
    filter_results = true,
    parallel = true,
)

save_simulations(
    D,
    TP_local,
    state,
    file_date,
    onset_dates,
    rng,
)

Z_historical_cumsum = cumsum(Z_historical, dims = 1)
s = (N .- Z_historical_cumsum) ./ N


# D_50 = deepcopy(D)
# D_38 = deepcopy(D)
D_25 = deepcopy(D)

let 
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end-truncation_days]
    
    D_50_local = D_50[:, 1, :] + D_50[:, 2, :]
    D_50_local_median = median(D_50_local, dims = 2)
    D_38_local = D_38[:, 1, :] + D_38[:, 2, :]
    D_38_local_median = median(D_38_local, dims = 2)
    D_25_local = D_25[:, 1, :] + D_25[:, 2, :]
    D_25_local_median = median(D_25_local, dims = 2)
    
    f = plot(layout = (3, 1))
    plot!(f, subplot = 1, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 1, D_50_local, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, subplot = 1, scale_factor[36:end, :], legend = false, lc = 4, linealpha = 0.5)
    plot!(f, subplot = 1, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 1, D_50_local_median, legend = false, lc = 3)
    vline!(f, subplot = 1, [19], lc = :black)
    xlims!(f, subplot = 1, 0, length(local_cases) + 35)
    ylims!(f, subplot = 1, 0, 5000)
    plot!(f, subplot = 2, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 2, D_38_local, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, subplot = 2, scale_factor[36:end, :], legend = false, lc = 4, linealpha = 0.5)
    plot!(f, subplot = 2, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 2, D_38_local_median, legend = false, lc = 3)
    vline!(f, subplot = 2, [19], lc = :black)
    xlims!(f, subplot = 2, 0, length(local_cases) + 35)
    ylims!(f, subplot = 2, 0, 5000)
    plot!(f, subplot = 3, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 3, D_25_local, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, subplot = 3, scale_factor[36:end, :], legend = false, lc = 4, linealpha = 0.5)
    plot!(f, subplot = 3, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 3, D_25_local_median, legend = false, lc = 3)
    vline!(f, subplot = 3, [19], lc = :black)
    xlims!(f, subplot = 3, 0, length(local_cases) + 35)
    ylims!(f, subplot = 3, 0, 5000)
end

let 
    plot(s, lc = 1, legend = false, alpha = 0.2)
end
    
let 
    plot(scale_factor, lc = 1, alpha = 0.2, legend = false)
end

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end-truncation_days]
    
    D_local = D[:, 1, :] + D[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    D_local_mean = mean(D_local, dims = 2)
    import_cases = import_case_dict[state][dates .>= forecast_start_date]
    import_cases = import_cases[begin:end-truncation_days]
    D_import = D[:, 3, :]
    D_import_median = median(D_import, dims = 2)
    D_import_mean = mean(D_import, dims = 2)
    f = plot(layout = (3, 1))
    plot!(f, subplot = 1, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 1, D_local, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, subplot = 1, scale_factor[36:end, :], legend = false, lc = 4, linealpha = 0.5)
    plot!(f, subplot = 1, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 1, D_local_median, legend = false, lc = 3)
    vline!(f, subplot = 1, [19], lc = :black)
    xlims!(f, subplot = 1, 0, length(local_cases) + 35)
    ylims!(f, subplot = 1, 0, 50000)
    # ylims!(f, subplot = 1, 0, 5000)
    # ylims!(f, subplot = 1, 0, 50000)
    # ylims!(0, 30000)
    # ylims!(0, 50000)
    # ylims!(0, 3000)
    # ylims!(0, 15000)
    plot!(f, subplot = 2, import_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 2, D_import, legend = false, lc = 2, linealpha = 0.5)
    plot!(f, subplot = 2, import_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, subplot = 2, D_import_median, legend = false, lc = 3)
    # plot!(D_import_mean, legend = false, lc = 3)
    xlims!(f, subplot = 2, 0, length(import_cases) + 35)
    # xlims!(length(local_cases) - 60, length(local_cases) + 35)
    ylims!(f, subplot = 2, 0, 60)
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

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    local_cases = local_cases[begin:end-truncation_days]
    
    D_local = D[:, 1, :] + D[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    D_local_mean = mean(D_local, dims = 2)
    import_cases = import_case_dict[state][dates .>= forecast_start_date]
    import_cases = import_cases[begin:end-truncation_days]
    D_import = D[:, 3, :]
    D_import_median = median(D_import, dims = 2)
    D_import_mean = mean(D_import, dims = 2)
    f = plot()
    # plot!(f, median(D_local, dims = 1), legend = false, lc = 2, linealpha = 0.5)
    plot!(f, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, median(Z_historical[36:end, :], dims = 2)[:], legend = false, lc = 4)
    # plot!(f, (1:length(local_cases)) .+ 7, local_cases, legend = false, linealpha = 1, lc = 1)
    plot!(f, D_local_median, legend = false, lc = 3)
    vline!(f, [19], lc = :black)
    xlims!(f, 0, length(local_cases) + 35)
    ylims!(f, 0, 200)
    ylims!(f, 0, 6000)
    # ylims!(f, 0, 50000)
    # ylims!(0, 30000)
    # ylims!(0, 50000)
    # ylims!(0, 3000)
    # ylims!(0, 15000)
    display(f)
end

let
    forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])
    local_cases = local_case_dict[state]
    # local_cases = local_cases[begin:end-truncation_days]
    
    D_local = D[:, 1, :] + D[:, 2, :]
    D_local_median = median(D_local, dims = 2)
    D_local_mean = mean(D_local, dims = 2)
    f = plot()
    plot!(
        f, 
        forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1),
        D_local, 
        legend = false, 
        lc = 2, 
        linealpha = 0.5
    )
    plot!(
        f, 
        forecast_start_date:Dates.Day(1):forecast_start_date + Dates.Day(size(D, 1)-1),
        D_local_median, 
        legend = false, 
        lc = 3, 
        linealpha = 0.5
    )
    plot!(
        f, 
        Dates.Date("2020-03-01"):Dates.Day(1):Dates.Date("2020-03-01") + Dates.Day(length(local_cases)-1),
        local_cases, 
        legend = false, 
        linealpha = 1, 
        lc = 1
    )
    
    x1 = forecast_start_date
    x2 = forecast_start_date + Dates.Day(size(D, 1)-1)
    
    # plot!(f, (1:length(local_cases)) .+ 7, local_cases, legend = false, linealpha = 1, lc = 1)
    # plot!(f, D_local_median, legend = false, lc = 3)
    vline!(f, [Dates.Date("2021-12-01")], lc = :black)
    xlims!((x1, x2))
    ylims!(f, 0, 200)
    ylims!(f, 0, 5000)
end

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
    ylims!(0, 40000)
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
