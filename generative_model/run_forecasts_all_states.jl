"""
This file is used to run the 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]
const nsims = parse(Int, ARGS[2])

# states to simulate 
const states_to_run = [
    "NSW",
    "QLD",
    "SA",
    "VIC",
    "WA",
    "ACT",
    # "NT",
    "TAS",
]

# run main 
simulate_all_states(file_date, states_to_run, nsims)

file_date = "2022-01-18"
