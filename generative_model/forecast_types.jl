struct Features
    """
    A type for holding all the fixed quantities during a simulation. This involves 
    simulation constants for detection probabiliites, constraints for consistency 
    with data and some data based measures like the local cases population size, 
    duration of sims and the length of the observation perod.
    """
    # constraints for the simulation 
    max_forecast_cases::Int
    cases_pre_forecast::Int
    N::Int
    T_observed::Int
    T_end::Int
    omicron_dominant_day::Int
end


struct Realisations
    """
    A type for holding the realisations of the simulations. This is a cleaner way of 
    holding the information for the three different matrices used. Z is for infections 
    D is for observed cases and U is for undetected cases. 
    """
    Z::Array{Int}
    Z_historical::Array{Int}
    D::Array{Int}
    U::Array{Int}
    
    function Realisations(
        sim_duration, 
        observation_period, 
        nsims,
    )
        """
        Initialising a state array object with arrays of zeros. 
        We pad the infection array Z with 35 days to account for 
        infections occuring prior to the simulation period. We do this
        separately (and not in a struct) as the arrays are large. 
        """
        
        Z = zeros(Int, sim_duration+35, 3, nsims)
        Z_historical = zeros(Int, sim_duration+35, nsims)
        D = zeros(Int, sim_duration, 3, nsims)
        U = zeros(Int, sim_duration, 3, nsims)
        
        return new(Z, Z_historical, D, U) 
    end
end



struct Constants{S, T}
    """
    A type for the dynamical / simulation constants 
    """
    k::S
    p_symp::S
    p_detect_given_symp::S
    p_detect_given_asymp::S
    p_detect::S
    p_symp_given_detect::S
    α_s::S
    α_a::S
    qi::S
    γ::T
    # import model parameters
    prior_alpha::T
    prior_beta::T
    ϕ::T
    # multiplier for the consistency between observed cases 
    consistency_multiplier::T
    
    function Constants(state)
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
        p_detect_given_symp_omicron = p_detect_given_symp_omicron_dict[state]
        p_detect_given_asymp_omicron = p_detect_given_asymp_omicron_dict[state]
        p_detect_omicron = p_symp_omicron*p_detect_given_symp_omicron + 
            (1-p_symp_omicron)*p_detect_given_asymp_omicron 
        p_symp_given_detect_omicron = (
            p_detect_given_symp_omicron*p_symp_omicron/p_detect_omicron 
        )
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
        p_detect_import = (
            delta = p_detect_import_delta, 
            omicron = p_detect_import_omicron
        )
        
        # other parameters 
        consistency_multiplier = 10.0
        # prior parametes for the import model
        prior_alpha = 0.5
        prior_beta = 0.2
        # ema smoothing factor 
        ϕ = 0.1
        
        S = typeof(α_s)
        T = typeof(prior_alpha)
        
        return new{S, T}(
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
    end
end


struct IndividualTypeMap
    S::Int
    A::Int
    I::Int
    
    function IndividualTypeMap()
        """
        Constructor for the mapping between type symbols S, A, I and integers. 
        """
        S = 1
        A = 2 
        I = 3
        
        return new(S, A, I)
    end
end


struct Forecast
    """
    This type wraps all the information relating to the forecast in a single accessible 
    object. This drastically simplifies the layout of the code at no performance hit. 
    """
    sim_features::Features
    sim_realisations::Realisations
    sim_constants::Constants
    individual_type_map::IndividualTypeMap
end