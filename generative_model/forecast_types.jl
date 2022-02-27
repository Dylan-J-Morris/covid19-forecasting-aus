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


mutable struct Realisations
    """
    A type for holding the realisations of the simulations. This is a cleaner way of 
    holding the information for the three different matrices used. Z is for infections 
    D is for observed cases and U is for undetected cases. 
    """
    Z::SharedArray{Int}
    Z_historical::SharedArray{Int}
    D::SharedArray{Int}
    U::SharedArray{Int}
    
    function Realisations(
        sim_duration,  
        nsims,
    )
        """
        Initialising a state array object with arrays of zeros. 
        We pad the infection array Z with 35 days to account for 
        infections occuring prior to the simulation period. We do this
        separately (and not in a struct) as the arrays are large. 
        """
        
        Z = SharedArray(zeros(Int, sim_duration + 35, 3, nsims))
        Z_historical = SharedArray(zeros(Int, sim_duration + 35, nsims))
        D = SharedArray(zeros(Int, sim_duration, 3, nsims))
        U = SharedArray(zeros(Int, sim_duration, 3, nsims))
        
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
    
    function Constants(state; p_detect_omicron = 0.5)
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
        p_detect_delta = p_symp_delta * p_detect_given_symp_delta + 
            (1 - p_symp_delta) * p_detect_given_asymp_delta 
        p_symp_given_detect_delta = (
            p_detect_given_symp_delta * p_symp_delta / p_detect_delta 
        )
        # prob of detecting an international import 
        p_detect_import_delta = 0.98
        
        ## omicron assumptions 
        k_omicron = 0.6
        p_symp_omicron = 0.4
        # solve the system:
        r = 2 / 3
        p_ds = p_detect_omicron / (p_symp_omicron + (1 - p_symp_omicron) * r) 
                
        p_detect_given_symp_omicron_dict = Dict{String, Float64}(
            "NSW" => p_ds,
            "QLD" => p_ds,
            "SA" => p_ds,
            "TAS" => p_ds,
            "WA" => p_ds,
            "ACT" => p_ds,
            "NT" => p_ds,
            "VIC" => p_ds,
        )
        p_detect_given_asymp_omicron_dict = Dict{String, Float64}(
            "NSW" => r * p_ds,
            "QLD" => r * p_ds,
            "SA" => r * p_ds,
            "TAS" => r * p_ds,
            "WA" => r * p_ds,
            "ACT" => r * p_ds,
            "NT" => r * p_ds,
            "VIC" => r * p_ds,
        )
        p_detect_given_symp_omicron = p_detect_given_symp_omicron_dict[state]
        p_detect_given_asymp_omicron = p_detect_given_asymp_omicron_dict[state]
        p_symp_given_detect_omicron = (
            p_detect_given_symp_omicron * p_symp_omicron / p_detect_omicron 
        )
        # as of 1/1/2022, same as the probability of detection for local omicron cases 
        p_detect_import_omicron = p_detect_omicron
        
        γ = 0.5     # relative infectiousness of asymptomatic
        α_s_delta = 1 / (p_symp_delta + γ * (1 - p_symp_delta))
        α_a_delta = γ * α_s_delta
        α_s_omicron = 1 / (p_symp_omicron + γ * (1 - p_symp_omicron))
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
            omicron = p_detect_import_omicron,
        )
        
        # other parameters 
        consistency_multiplier = 5.0
        # prior parametes for the import model
        prior_alpha = 0.5
        prior_beta = 0.1
        # ema smoothing factor 
        ϕ = 0.5
        
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


struct JurisdictionAssumptions
    """
    This type holds all the assumptions for each jurisdiction. This 
    involves the assumed starting dates, population sizes and initial conditions. It also 
    includes key dates we don't have stored elsewhere. 
    """
    simulation_start_dates::Dict{String, String}
    pop_sizes::Dict{String, Int}
    initial_conditions::Dict{
        String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}
    }
    omicron_dominant_date::Date

    function JurisdictionAssumptions()
        simulation_start_dates = Dict{String, String}(
            "NSW" => "2021-06-01",
            "QLD" => "2021-11-20",
            "SA" => "2021-11-01",
            "TAS" => "2021-11-01",
            "WA" => "2021-12-15",
            "ACT" => "2021-12-01",
            "NT" => "2021-12-01",
            "VIC" => "2021-08-01",
        )
        
        # date we want to apply increase in cases due to Omicron 
        omicron_dominant_date = Dates.Date("2021-12-15")
        
        pop_sizes = Dict{String, Int}(
            "NSW" => 8189266,
            "QLD" => 5221170,
            "SA" => 1773243,
            "TAS" => 541479,
            "VIC" => 6649159,
            "WA" => 2681633,
            "ACT" => 432266,
            "NT" => 246338,
        )
            
        initial_conditions = Dict{
            String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}
        }(
            "NSW" => (S = 0, A = 0, I = 0),
            "QLD" => (S = 1, A = 1, I = 0),
            "SA" => (S = 0, A = 0, I = 0),
            "TAS" => (S = 0, A = 0, I = 0),
            "VIC" => (S = 5, A = 5, I = 0),
            "WA" => (S = 0, A = 0, I = 0),
            "ACT" => (S = 12, A = 10, I = 0),
            "NT" => (S = 0, A = 0, I = 0),
        )
        
        return new(
            simulation_start_dates,
            pop_sizes,
            initial_conditions,
            omicron_dominant_date,
        )
        
    end
end