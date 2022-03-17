"""
This file is used to run the full simulation pipeline, including running the generative 
model, merging and saving files and creating plots of all case forecasts. 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]

const run_simulation = true 
if length(ARGS) > 2
    run_simulation = ARGS[2] == "false" ? false : true 
end

# const nsims = parse(Int, ARGS[2])

const nsims = Dict{String, Int}(
    "NSW" => 30000,
    # "QLD" => 10000,
    # "SA" => 10000,
    "VIC" => 30000,
    # "WA" => 5000,
    # "ACT" => 10000,
    # "NT" => 10000,
    # "TAS" => 10000,
)

for state in keys(nsims)
    # run main 
    simulate_single_state(file_date, state, nsims[state], run_simulation, adjust_TP = true)
end
