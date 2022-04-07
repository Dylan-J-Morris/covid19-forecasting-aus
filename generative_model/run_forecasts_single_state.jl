"""
This file is used to run the full simulation pipeline, including running the generative 
model, merging and saving files and creating a 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]
const state = ARGS[2]

const run_simulation = true 
if length(ARGS) > 2
    run_simulation = ARGS[3] == "false" ? false : true 
end

# const nsims = parse(Int, ARGS[2])

const nsims = Dict{String, Int}(
    "NSW" => 1000,
    "QLD" => 1000,
    "SA" => 1000,
    "VIC" => 1000,
    "WA" => 1000,
    "ACT" => 1000,
    "NT" => 1000,
    "TAS" => 1000,
)

# run main 
simulate_single_state(file_date, state, nsims[state], run_simulation, adjust_TP = true)