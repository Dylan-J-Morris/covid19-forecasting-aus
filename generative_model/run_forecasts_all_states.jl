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
    "NSW" => 3000,
    "QLD" => 3000,
    "SA" => 3000,
    "VIC" => 3000,
    "WA" => 3000,
    "ACT" => 3000,
    "NT" => 3000,
    "TAS" => 3000,
)

# run main 
simulate_all_states(file_date, states_to_run, nsims, run_simulation)