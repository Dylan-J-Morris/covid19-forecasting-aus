"""
This file is used to test the simulation. This is just the simulate_states code unwrapped 
so will be a little less optimal than when directly run but should enable an easier 
observation of correctness. 
"""

using Revise
using Distributed
using Plots
using Chain
using ProfileView

include("simulate_states.jl")

# parameters to pass to the main function

file_date = "2022-04-12"

# set seed for consistent plots
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
state = "QLD"
nsims = 1000
forecast_start_date = Dates.Date(jurisdiction_assumptions.simulation_start_dates[state])

# named tuple for initial conditions
D0 = jurisdiction_assumptions.initial_conditions[state]
N = jurisdiction_assumptions.pop_sizes[state]

# get the observed cases
local_cases = local_case_dict[state][dates .>= forecast_start_date]
# cutoff the last bit of the local cases
import_cases = import_case_dict[state]
local_cases = Int.(local_cases[begin:end-truncation_days+1])
import_cases = Int.(import_cases)

local_cases_input = CSV.read("data/local_cases_input_2022-04-13.csv", DataFrame)
local_cases_input = filter(:state => ==(state), local_cases_input)

plot(dates, local_case_dict[state])
plot!(local_cases_input.date_onset, local_cases_input.count)
xlims!((Dates.Date("2022-01-01"), Dates.Date("2022-04-15")))

include("simulate_states.jl")

(Z, D, U) = simulate_branching_process(
    D0,
    N,
    nsims,
    local_cases,
    import_cases,
    forecast_start_date,
    file_date,
    state,
    adjust_TP = false,
)

save_simulations(
    D,
    state,
    file_date,
    onset_dates,
    rng,
)

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
    # plot!(f, dates[dates .>= forecast_start_date][begin:end-truncation_days+1], local_cases[begin:end-truncation_days+1], legend = false, lc = :black, lw = 2)
    plot!(f, dates[dates .>= forecast_start_date], local_cases, legend = false, lc = :black, lw = 2)
    # plot!(f, local_cases2.date_onset, local_cases2.count, legend = false, linealpha = 1, lc = 1)
    plot!(f, D_local_dates, D_local_median, legend = false, lc = 2, lw = 2)
    # plot!(f, df_state[!, "onset date"], Matrix(Int.(df_state[!, 3:end-1])))
    vline!(f, [Dates.Date("2021-11-15")], ls = :dash, lc = :black)
    # xlims!(f, 0, length(local_cases) + 35)
    ylims!(f, 0, 40000)
    xlims!(f, Dates.value(Dates.Date("2021-12-15")), Dates.value(Dates.Date("2022-05-20")))
    # ylims!(f, 0, 500)
end

cases_adjusted = CSV.read("results/EpyReff/cases_adjusted_2022-04-12.csv", DataFrame)

cases_adjusted_state = filter(:STATE => ==(state), cases_adjusted)

plot(cases_adjusted_state.local_scaled_14)
plot!(cases_adjusted_state.local_scaled_28)
plot!(cases_adjusted_state.local_scaled_42)
plot!(cases_adjusted_state.local_scaled_72)
plot!(cases_adjusted_state.local_scaled_102)
plot!(cases_adjusted_state.local_scaled_132)

ϕ = read_in_susceptible_depletion(file_date)
ϕ_mean = mean(ϕ, dims = 1)

ψ = (
    ϕ_mean[6] * (cases_adjusted_state.local_scaled_14) / N
    + ϕ_mean[5] * (cases_adjusted_state.local_scaled_28) / N
    + ϕ_mean[4] * (cases_adjusted_state.local_scaled_42) / N
    + ϕ_mean[3] * (cases_adjusted_state.local_scaled_72) / N
    + ϕ_mean[2] * (cases_adjusted_state.local_scaled_102) / N
    + ϕ_mean[1] * (cases_adjusted_state.local_scaled_132) / N
)

scaling_factor = 1 .- ψ

scaling_factor_sim = zeros(size(Z, 1))
prop_inf_full = zeros(size(Z, 1), 6)

Z_historical = mean(Z, dims = 2)

for day in 1:length(Z_historical)
    
    ψ = 0.0
    
    τ_horizons_start = [0, 14, 28, 42, 72, 102]
    τ_horizons_end = [1, 14, 28, 42, 72, 102, 132] .- 1
    
    for i in 1:length(τ_horizons_start) 
        
        # Get starting and ending index. Start must be at least the beginning of the array and end 
        # must be at least the forecast length.
        τ_start = max(day - τ_horizons_end[i + 1], 1)
        τ_end = min(day - τ_horizons_start[i], size(Z_historical, 1))
        
        # sum infections over the time period of interest
        infections = sum(@view Z_historical[τ_start:τ_end])
        
        # we index from 4 - 1 as the largest reduction occurs as a result of recent infections 
        prop_inf = min(1.0, infections / N)
        ψ += ϕ_mean[7 - i] * prop_inf
        
        prop_inf_full[day, i] = prop_inf
        
    end
    
    scale_factor = max(0.0, 1.0 - ψ)
    
    scaling_factor_sim[day] = scale_factor
    
end

d_start = Dates.Date(forecast_start_date) - Dates.Day(30)
Δd = Dates.Day(1)
d_end = Dates.Date(forecast_start_date) + Δd * (size(Z, 1) - 1) - Dates.Day(30)
D_local_dates = d_start:Δd:d_end

plot(cases_adjusted_state.date_inferred, scaling_factor)
plot!(D_local_dates, scaling_factor_sim)
xlims!((Dates.Date("2021-12-01"), D_local_dates[end]))
ylabel!("Scaling factor")

fig = plot(layout = (3, 2), legend = false)
plot!(
    fig, 
    subplot = 1, 
    cases_adjusted_state.date_inferred,
    cases_adjusted_state.local_scaled_14 ./ N, 
    title = "14 days", titlefontsize=8
)
plot!(
    fig, 
    subplot = 2, 
    cases_adjusted_state.date_inferred,
    cases_adjusted_state.local_scaled_28 ./ N, 
    title = "28-14 days", titlefontsize=8
)
plot!(
    fig, 
    subplot = 3, 
    cases_adjusted_state.date_inferred,
    cases_adjusted_state.local_scaled_42 ./ N, 
    title = "42-28 days", titlefontsize=8
)
plot!(
    fig, 
    subplot = 4, 
    cases_adjusted_state.date_inferred,
    cases_adjusted_state.local_scaled_72 ./ N, 
    title = "72-42 days", titlefontsize=8
)
plot!(
    fig, 
    subplot = 5, 
    cases_adjusted_state.date_inferred,
    cases_adjusted_state.local_scaled_102 ./ N, 
    title = "102-72 days", titlefontsize=8)
plot!(
    fig, 
    subplot = 6, 
    cases_adjusted_state.date_inferred,
    cases_adjusted_state.local_scaled_132 ./ N, 
    title = "132-102 days", titlefontsize=8)
plot!(
    fig, 
    subplot = 1, 
    D_local_dates, 
    prop_inf_full[:, 1], 
    ls = :dash
)
plot!(
    fig, 
    subplot = 2, 
    D_local_dates, 
    prop_inf_full[:, 2], 
    ls = :dash
)
plot!(
    fig, 
    subplot = 3, 
    D_local_dates, 
    prop_inf_full[:, 3], 
    ls = :dash
)
plot!(
    fig, 
    subplot = 4, 
    D_local_dates, 
    prop_inf_full[:, 4], 
    ls = :dash
)
plot!(
    fig, 
    subplot = 5, 
    D_local_dates, 
    prop_inf_full[:, 5], 
    ls = :dash
)
plot!(
    fig, 
    subplot = 6, 
    D_local_dates, 
    prop_inf_full[:, 6], 
    ls = :dash
)
xlims!((Dates.Date("2021-12-01"), D_local_dates[end]))
ylims!((0, 0.25))



