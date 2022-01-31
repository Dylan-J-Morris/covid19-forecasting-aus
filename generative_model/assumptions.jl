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
    ## Delta assumptions
    k_delta = 0.15
    p_symp_delta = 0.7
    p_detect_given_symp_delta_dict = Dict{String, Float64}(
        "NSW" => 0.95,
        "QLD" => 0.95,
        "SA" => 0.95,
        "TAS" => 0.95,
        "WA" => 0.95,
        "ACT" => 0.95,
        "NT" => 0.95,
        "VIC" => 0.95,
    )
    
    p_detect_given_asymp_delta_dict = Dict{String, Float64}(
        "NSW" => 0.1,
        "QLD" => 0.1,
        "SA" => 0.1,
        "TAS" => 0.1,
        "WA" => 0.1,
        "ACT" => 0.1,
        "NT" => 0.1,
        "VIC" => 0.1,
    )
    
    p_detect_given_symp_delta = p_detect_given_symp_delta_dict[state]
    p_detect_given_asymp_delta = p_detect_given_asymp_delta_dict[state]
    p_detect_delta = p_symp_delta*p_detect_given_symp_delta + 
        (1-p_symp_delta)*p_detect_given_asymp_delta 
    p_symp_given_detect_delta = p_detect_given_symp_delta*p_symp_delta/p_detect_delta 
    # prob of detecting an international import 
    p_detect_import_delta = 0.98
    
    ## omicron assumptions 
    k_omicron = 0.6
    
    # assumptions surrouding the probability of symptomatic, 
    # relative infectiousness γ and the ratio of Reff (α's) 
    p_symp_omicron = 0.4
    
    p_detect_given_symp_omicron_dict = Dict{String, Float64}(
        "NSW" => 0.7,
        "QLD" => 0.7,
        "SA" => 0.7,
        "TAS" => 0.7,
        "WA" => 0.7,
        "ACT" => 0.7,
        "NT" => 0.7,
        "VIC" => 0.7,
    )
    
    p_detect_given_asymp_omicron_dict = Dict{String, Float64}(
        "NSW" => 0.467,
        "QLD" => 0.467,
        "SA" => 0.467,
        "TAS" => 0.467,
        "WA" => 0.467,
        "ACT" => 0.467,
        "NT" => 0.467,
        "VIC" => 0.467,
    )
    
    # solve the system: 
    p_detect_given_symp_omicron = p_detect_given_symp_omicron_dict[state]
    p_detect_given_asymp_omicron = p_detect_given_asymp_omicron_dict[state]
    p_detect_omicron = p_symp_omicron*p_detect_given_symp_omicron + 
        (1-p_symp_omicron)*p_detect_given_asymp_omicron 
    # prob symptomatic given detect
    p_symp_given_detect_omicron = p_detect_given_symp_omicron*p_symp_omicron/p_detect_omicron 
    # as of 1/1/2022, same as the probability of detection for local omicron cases 
    p_detect_import_omicron = p_detect_omicron  
    
    γ = 0.5     # relative infectiousness of asymptomatic
    α_s_delta = 1/(p_symp_delta + γ*(1-p_symp_delta))
    α_a_delta = γ * α_s_delta
    α_s_omicron = 1/(p_symp_omicron + γ*(1-p_symp_omicron))
    α_a_omicron = γ * α_s_omicron
    
    # store all parameters in named tuples indexed by strain
    α_s = (delta = α_s_delta, omicron = α_s_omicron)
    α_a = (delta = α_a_delta, omicron = α_a_omicron)
    k = (delta = k_delta, omicron = k_omicron)
    p_symp = (delta = p_symp_delta, omicron = p_symp_omicron)
    p_detect_given_symp = (
        delta = p_detect_given_symp_delta, 
        omicron = p_detect_given_symp_omicron, 
    )
    p_detect_given_asymp = (
        delta = p_detect_given_asymp_delta, 
        omicron = p_detect_given_asymp_omicron, 
    )
    p_detect = (
        delta = p_detect_delta, 
        omicron = p_detect_omicron, 
    )
    p_symp_given_detect = (
        delta = p_symp_given_detect_delta,
        omicron = p_symp_given_detect_omicron,  
    )
    p_detect_import = (delta = p_detect_import_delta, omicron = p_detect_import_omicron)
    
    # other parameters 
    consistency_multiplier = 5.0
    # prior parametes for the import model
    prior_alpha = 0.5
    prior_beta = 0.2
    # ema smoothing factor 
    ϕ = 0.1

    simulation_constants = Constants(
        k,
        p_symp,
        p_detect_given_symp,
        p_detect_given_asymp,
        p_detect, 
        p_symp_given_detect,
        α_s,
        α_a,
        p_detect_import,
        γ,
        prior_alpha,
        prior_beta,
        ϕ,
        consistency_multiplier,
    )

    # mapping between types 
    individual_type_map = IndividualTypeMap(1, 2, 3)

    return (simulation_constants, individual_type_map)
    
end
