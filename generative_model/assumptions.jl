using Distributions
using Random

function sample_infection_time(;omicron=false)
    """
    Sample infection times for num individuals based on the generation 
    interval distribution, Gamma(shape_gen, scale_gen). 
    """

    (shape_gen, scale_gen) = (2.75, 1.00)
    (shape_gen_omicron, scale_gen_omicron) = (1.58, 1.32)
    
    shape = (1-omicron)*shape_gen + omicron*shape_gen_omicron
    scale = (1-omicron)*scale_gen + omicron*scale_gen_omicron
    
    infection_time = ceil(Int, rand(Gamma(shape, scale)))
    
    return infection_time
    
end


function sample_onset_time(;omicron=false)
    """
    Sample incubation times for num individuals based on incubation period 
    distribution, Gamma(shape_inc, scale_inc). 
    """
    
    (shape_inc, scale_inc) = (5.807, 0.948)
    (shape_inc_omicron, scale_inc_omicron) = (3.33, 1.34)
    
    shape = (1-omicron)*shape_inc + omicron*shape_inc_omicron
    scale = (1-omicron)*scale_inc + omicron*scale_inc_omicron
    
    onset_time = ceil(Int, rand(Gamma(shape, scale)))
    
    return onset_time
    
end


function set_simulation_constants(state)
    """
    Contains the assumptions for simulation parameters. This includes all the dynamical
    constants: 
        - k = heterogeneity parameter
        - p_symp = probability of symptoms 
        - γ = relative infectiousness of asymptomatic individuals 
        - p_symp_given_detect = probability of symptoms given detection
        - p_asymp_given_detect = probability of being asymptomatic given detection
        - consistency_multiplier = chosen such that sim_cases < 
            consistency_multiplier*actual cases 
            results in cases being injected into the simulation. This is used to account 
            for superspreading events after periods of low incidence. 
    These values are stored in sim_constants which is a dictionary indexed by the 
    parameter name and ultimately stored on the stack in a SimulationParameters object. 
    """
    # get the simulation constants
    simulation_constants = Constants(state)
    # mapping between types 
    individual_type_map = IndividualTypeMap()

    return (simulation_constants, individual_type_map)
    
end
