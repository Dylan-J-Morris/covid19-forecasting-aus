"""
This file is used to run the 
"""

include("simulate_states.jl")

# parameters to pass to the main function 
# const file_date = "2022-01-04"
# const nsims = 10000
# file_date = "2022-01-04"
# nsims = 10000
(file_date, nsims) = ARGS
# run main 
simulate_all_states(file_date,nsims)