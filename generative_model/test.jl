"""
This file is used to test the simulation. This is just the simulate_states code unwrapped 
so will be a little less optimal than when directly run but should enable an easier 
observation of correctness. 
"""

using Revise
using Distributed

if nprocs() == 1
    addprocs(4)
end

include("simulate_states.jl")
# parameters to pass to the main function 
const file_date = "2022-02-09"
nsims = 1000

# states to simulate 
const states_to_run = [
    "SA",
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
    "NT" => "2021-12-01",
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
    "WA" => (S = 3, A = 2, I = 0),
    "ACT" => (S = 0, A = 0, I = 0),
    "NT" => (S = 3, A = 2, I = 0),    
)

(local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
dates = local_case_dict["date"]
last_date_in_data = dates[end]
forecast_end_date = last_date_in_data + Dates.Day(35)

# create vector for dates 
onset_dates = latest_start_date:Dates.Day(1):forecast_end_date

# add a small truncation to the simulations as we don't trust the most recent data 
truncation_days = 7

D = []
U = []

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
    
    save_simulations(D,TP_local,state,file_date,onset_dates,rng)

end

forecast_start_date = Dates.Date(simulation_start_dates[states_to_run[1]])
local_cases = local_case_dict[states_to_run[1]][dates .>= forecast_start_date]

D_med = median(D[:, 1, :] + D[:, 2, :], dims = 2)

bar(local_cases)
plot!(D_med, legend = false, linealpha = 1)
# plot!(D[:, 1, :] + D[:, 2, :], legend = false, lc = :grey, linealpha = 0.5)
# xlims!(150, 250)
# ylims!(0, 50000)


# # merge all the simulation and TP files for states into single CSV
# merge_simulation_files(file_date)    
# merge_TP_files(file_date)
# plot_all_forecast_intervals(file_date, states_to_run, local_case_dict)