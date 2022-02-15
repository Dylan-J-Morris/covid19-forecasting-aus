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
        addprocs(4)
    end
end

include("simulate_states.jl")
# parameters to pass to the main function 
const file_date = "2022-02-09"
nsims = 500

# states to simulate 
const states_to_run = [
    "NSW",
]

# set seed for consistent plots (NOTE: this is not useful when multithreading 
# enabled as we use separate seeds but the simulation pool should handle that)
rng = Random.Xoshiro(2022)

simulation_start_dates = Dict{String, String}(
    "NSW" => "2021-06-23",
    "QLD" => "2021-11-01",
    "SA" => "2021-11-01",
    "TAS" => "2021-11-01",
    "WA" => "2021-11-01",
    "ACT" => "2021-08-05",
    "NT" => "2021-11-01",
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
    "NSW" => (S = 2, A = 2, I = 0),
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

# add a small truncation to the simulations as we don't trust the most recent data 
truncation_days = 7

include("simulate_states.jl")
D = []
U = []
Z = []
Z_historical = []

for state in states_to_run

    forecast_start_date = Dates.Date(simulation_start_dates[state])
    local_cases = local_case_dict[state]
    import_cases = import_case_dict[state]

    # named tuple for initial condi
    D0 = initial_conditions[state]
    N = pop_sizes[state]

    # get the observed cases up to file_date - truncation_days
    cases_pre_forecast = sum(local_case_dict[state][dates .< forecast_start_date])
    local_cases = local_case_dict[state][dates .>= forecast_start_date]
    import_cases = import_case_dict[state]
    local_cases = local_cases[begin:end-truncation_days]
    import_cases = import_cases[begin:end-truncation_days]
    
    (D, U, TP_local) = simulate_branching_process(
        D0, 
        N, 
        nsims, 
        local_cases, 
        import_cases, 
        cases_pre_forecast,
        forecast_start_date, 
        file_date, 
        omicron_dominant_date, 
        state,
    )
    # (D, Z, U, Z_historical, TP_local) = simulate_branching_process(
    #     D0, 
    #     N, 
    #     nsims, 
    #     local_cases, 
    #     import_cases, 
    #     cases_pre_forecast,
    #     forecast_start_date, 
    #     file_date, 
    #     omicron_dominant_date, 
    #     state,
    # )
    
    save_simulations(D,TP_local,state,file_date,onset_dates,rng)

end

forecast_start_date = Dates.Date(simulation_start_dates[states_to_run[1]])
local_cases = local_case_dict[states_to_run[1]][dates .>= forecast_start_date]
local_cases = local_cases[begin:end-7]
ma = zeros(Float64, length(local_cases))
moving_average!(ma, local_cases, 7)

D_local = D[:, 1, :] + D[:, 2, :]
D_local_med = median(D_local, dims = 2)

D_local_tmp = D_local[
    :, 
    (D_local[50, :] .< 50) .& (D_local[75, :] .< 500)
]

min_ma = 0.25 * local_cases
max_ma = 2.0 * local_cases

min_ma[min_ma .< 50] .= 0
max_ma[max_ma .< 50] .= 50

plot(D[:,:,10])

let 
    plot(ma, legend = false, linealpha = 1, lc = 1)
    plot!(min_ma, legend = false, linealpha = 1, lc = 1, ls = :dash)
    plot!(max_ma, legend = false, linealpha = 1, lc = 1, ls = :dash)
    # plot!(D_med, legend = false, linealpha = 1)
    # plot!(D_local_tmp, legend = false, lc = :grey, linealpha = 0.5)
    # plot!(D_local, legend = false, lc = :grey, linealpha = 0.5)
    plot!(D_local_med, legend = false, lc = :grey, linealpha = 0.5)
    xlims!(50, 125)
    ylims!(0, 2000)
end

# # merge all the simulation and TP files for states into single CSV
# merge_simulation_files(file_date)    
# merge_TP_files(file_date)
# plot_all_forecast_intervals(file_date, states_to_run, local_case_dict)