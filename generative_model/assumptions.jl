using Distributions
using Random

function sample_infection_time()
    """
    Sample infection times for num individuals based on the generation 
    interval distribution, Gamma(shape_gen, scale_gen). 
    """
    
    shape_gen = 2.75
    scale_gen = 1.00
    infection_time = ceil(Int, rand(Gamma(shape_gen, scale_gen)))
    
    return infection_time
end

function sample_onset_time()
    """
    Sample incubation times for num individuals based on incubation period 
    distribution, Gamma(shape_inc, scale_inc). 
    """
    
    shape_inc = 5.807  
    scale_inc = 0.948   
    onset_time = ceil(Int, rand(Gamma(shape_inc, scale_inc)))
    
    return onset_time
end


function set_simulation_constants()
    """
    Contains the assumptions for simulation parameters. This includes all the dynamical constants: 
        - k = heterogeneity parameter
        - p_symp = probability of symptoms 
        - γ = relative infectiousness of asymptomatic individuals 
        - p_symp_given_detect = probability of symptoms given detection
        - p_asymp_given_detect = probability of being asymptomatic given detection
        - consistency_multiplier = chosen such that sim_cases < consistency_multiplier*actual cases 
            results in cases being injected into the simulation. This is used to account 
            for superspreading events after periods of low incidence. 
    These values are stored in sim_constants which is a dictionary indexed by the parameter name and 
    ultimately stored on the stack in a SimulationParameters object. 
    """

    # overdispersion parameter 
    k = 0.15
    # assumptions surrouding the probability of symptomatic, 
    # relative infectiousness γ and the ratio of Reff (α's) 
    p_symp = 0.7
    γ = 0.5     # relative infectiousness of asymptomatic
    # solve the system α_s*ps + α_a(1-ps) = 1 with α_a = γ*α_s
    α_s = 1/(p_symp+γ*(1-p_symp))
    α_a = γ * α_s
    p_detect_given_symp = 0.95
    p_detect_given_asymp = 0.1
    consistency_multiplier = 5.0

    # prob of detecting an international import 
    qi = 0.98
    # prior parametes for the import model
    prior_alpha = 0.5
    prior_beta = 0.2
    # ema smoothing factor 
    ϕ = 0.1

    simulation_constants = SimulationConstants(
        k,
        p_symp,
        p_detect_given_symp,
        p_detect_given_asymp,
        γ,
        α_s,
        α_a,
        qi,
        prior_alpha,
        prior_beta,
        ϕ,
        consistency_multiplier,
    )

    # mapping between types 
    individual_type_map = IndividualTypeMap(1, 2, 3)

    return (simulation_constants, individual_type_map)
end
