"""
This file is used to run the full simulation pipeline, including running the generative 
model, merging and saving files and creating a 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]
const state = ARGS[2]

run_simulation = true 
if length(ARGS) > 2
    run_simulation = ARGS[2] == "false" ? false : true 
end

# const nsims = parse(Int, ARGS[2])

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
simulate_single_state(file_date, state, nsims[state], run_simulation)