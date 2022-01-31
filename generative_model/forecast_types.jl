struct Features{T}
    """
    A type for holding all the fixed quantities during a simulation. This involves 
    simulation constants for detection probabiliites, constraints for consistency 
    with data and some data based measures like the local cases population size, 
    duration of sims and the length of the observation perod.
    """
    # constraints for the simulation 
    max_forecast_cases::T
    cases_pre_forecast::T
    N::T
    T_observed::T
    T_end::T
    omicron_dominant_day::T
end


struct Realisations{T1, T2}
    """
    A type for holding the realisations of the simulations. This is a cleaner way of 
    holding the information for the three different matrices used. Z is for infections 
    D is for observed cases and U is for undetected cases. 
    """
    Z::T1
    D::T2
    U::T2
end


struct Constants{T, S}
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
end


struct IndividualTypeMap{T}
    S::T
    A::T
    I::T
end


struct Forecast{T1,T2,T3,T4,T5}
    """
    This type wraps all the information relating to the forecast in a single accessible 
    object. This drastically simplifies the layout of the code at no performance hit. 
    """
    sim_features::Features{T1}
    sim_realisations::Realisations{T2,T3}
    sim_constants::Constants{T4}
    individual_type_map::IndividualTypeMap{T5}
end