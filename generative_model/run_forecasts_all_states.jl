"""
This file is used to run the 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
file_date = ARGS[1]
nsims = parse(Int, ARGS[2])
# run main 
simulate_all_states(file_date,nsims)