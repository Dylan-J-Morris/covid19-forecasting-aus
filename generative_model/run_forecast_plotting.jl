include("simulate_states.jl")
include("forecast_plots.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]

states = [
    "ACT",
    # "NSW",
    # "NT",
    "QLD",
    "SA",
    "TAS",
    # "VIC",
    "WA",
]

# set seed for consistent plots (NOTE: this is not useful when multithreading 
# enabled as we use separate seeds but the simulation pool should handle that)
rng = Random.Xoshiro(2022)

jurisdiction_assumptions = JurisdictionAssumptions()

# get the latest onset date
latest_start_date = Dates.Date(
    maximum(v for v in values(jurisdiction_assumptions.simulation_start_dates))
)

(dates, local_case_dict, import_case_dict) = read_in_cases(file_date, rng)
last_date_in_data = dates[end]
forecast_end_date = last_date_in_data + Dates.Day(35)

# create vector for dates 
onset_dates = latest_start_date:Dates.Day(1):forecast_end_date

# merge all the simulation and TP files for states into single CSV
merge_simulation_files(file_date)    
plot_all_forecast_intervals(file_date, states, dates, local_case_dict)