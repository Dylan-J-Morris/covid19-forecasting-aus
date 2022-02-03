"""
This file is used to run the 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]
# const nsims = parse(Int, ARGS[2])

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
    
const nsims = Dict{String, Int}(
    "NSW" => 5000,
    "QLD" => 5000,
    "SA" => 5000,
    "VIC" => 5000,
    "WA" => 5000,
    "ACT" => 5000,
    "NT" => 5000,
    "TAS" => 5000,
)

# const nsims = Dict{String, Int}(
#     "NSW" => 1000,
#     "QLD" => 1000,
#     "SA" => 1000,
#     "VIC" => 1000,
#     "WA" => 1000,
#     "ACT" => 1000,
#     "NT" => 1000,
#     "TAS" => 1000,
# )

# run main 
simulate_all_states(file_date, states_to_run, nsims)


