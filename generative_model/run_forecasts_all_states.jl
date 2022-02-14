"""
This file is used to run the full simulation pipeline, including running the generative 
model, merging and saving files and creating a 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
const file_date = ARGS[1]

run_simulation = true 

if length(ARGS) > 1
    run_simulation = ARGS[2] == "false" ? false : true 
end

# const nsims = parse(Int, ARGS[2])

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
    
# const nsims = Dict{String, Int}(
#     "ACT" => 20000,
#     "NSW" => 6000,
#     "NT" => 20000,
#     "QLD" => 3000,
#     "SA" => 3000,
#     "TAS" => 3000,
#     "VIC" => 5000,
#     "WA" => 3000,
# )

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
simulate_all_states(file_date, states_to_run, nsims, run_simulation)