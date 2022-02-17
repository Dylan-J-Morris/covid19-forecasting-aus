"""
This file is used to run the full simulation pipeline, including running the generative 
model, merging and saving files and creating plots of all case forecasts. 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]

run_simulation = true 

if length(ARGS) > 1
    run_simulation = ARGS[2] == "false" ? false : true 
end

# states to simulate 
const states_to_run = [
    "ACT",
    "NSW",
    "NT",
    "QLD",
    "SA",
    "TAS",
    "VIC",
    "WA",
]

const nsims = Dict{String, Int}(
    "NSW" => 10000,
    "QLD" => 10000,
    "SA" => 10000,
    "VIC" => 10000,
    "WA" => 10000,
    "ACT" => 10000,
    "NT" => 10000,
    "TAS" => 10000,
)

# run main 
simulate_all_states(file_date, states_to_run, nsims, run_simulation)