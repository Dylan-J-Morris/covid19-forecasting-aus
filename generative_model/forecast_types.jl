struct Features
    """
    A type for holding all the fixed quantities during a simulation. This involves 
    simulation constants for detection probabiliites, constraints for consistency 
    with data and some data based measures like the local cases population size, 
    duration of sims and the length of the observation perod.
    """
    # constraints for the simulation 
    max_forecast_cases::Int
    N::Int
    T_observed::Int
    T_end::Int
    # number of days for each consistency check 
    τ::Int
    min_cases::Vector{Int}
    max_cases::Vector{Int}
    idxs_limits::Vector{UnitRange{Int64}}
    state::String
    # truncations for the various bits and pieces
    omicron_start_day::Int
    omicron_only_day::Int
end


mutable struct Realisation
    """
    A type for holding the realisations of the simulations. This is a cleaner way of 
    holding the information for the three different matrices used. Z is for infections 
    D is for observed cases and U is for undetected cases. 
    """
    Z::Array{Int}
    Z_historical::Array{Int}
    D::Array{Int}
    U::Array{Int}
    
    function Realisation(
        sim_duration,  
    )
        """
        Initialising a state array object with arrays of zeros. 
        We pad the infection array Z with 35 days to account for 
        infections occuring prior to the simulation period. We do this
        separately (and not in a struct) as the arrays are large. 
        """
        
        Z = zeros(Int, sim_duration + 30, 3)
        Z_historical = zeros(Int, sim_duration + 30)
        D = zeros(Int, sim_duration, 3)
        U = zeros(Int, sim_duration, 3)
        
        return new(Z, Z_historical, D, U) 
    end
end

mutable struct Results
    """
    A type for holding the realisations of the simulations. This is a cleaner way of 
    holding the information for the three different matrices used. Z is for infections 
    D is for observed cases and U is for undetected cases. 
    """
    Z::Array{Int}
    Z_historical::Array{Int}
    D::Array{Int}
    U::Array{Int}
    
    function Results(
        sim_duration,  
        nsims,
    )
        """
        Initialising a state array object with arrays of zeros. 
        We pad the infection array Z with 35 days to account for 
        infections occuring prior to the simulation period. We do this
        separately (and not in a struct) as the arrays are large. 
        """
        
        Z = zeros(Int, sim_duration + 30, 3, nsims)
        Z_historical = zeros(Int, sim_duration + 30, nsims)
        D = zeros(Int, sim_duration, 3, nsims)
        U = zeros(Int, sim_duration, 3, nsims)
        
        return new(Z, Z_historical, D, U) 
    end
end


struct Constants{S, T}
    """
    A type for the dynamical / simulation constants 
    """
    p_detect::S
    p_symp::S
    p_detect_given_symp::S
    p_detect_given_asymp::S
    p_symp_given_detect::S
    α_s::S
    α_a::S
    p_detect_import::S
    prior_alpha::T
    prior_beta::T
    ϕ::T
    consistency_multiplier::T
    
    function Constants(start_date, end_date)
        """
        Constructor for the older version of the model. This assumes a simpler form of the CAR and 
        doesn't have jurisdiction dependent assumptions in relation to the CAR.
        """
        
        # date vectors to indicate which period we're in. Noting that this is +5 days (mean inc 
        # period) compared to the ranges used in the fitting. This is to deal with detection being 
        # related to detection of cases and not infection dates. 
        CAR_normal_before = Dates.Date(start_date):Dates.Day(1):Dates.Date("2021-12-09")
        CAR_low = Dates.Date("2021-12-10"):Dates.Day(1):Dates.Date(end_date)
        # all dates
        CAR_dates = [
            CAR_normal_before;
            CAR_low
        ]
        
        # initialise the 
        CAR = 0.75 * ones(Float64, length(CAR_dates))
        # idx = CAR_dates .>= Dates.Date("2021-12-10")
        idx = CAR_dates .>= CAR_low[begin]
        CAR[idx] .= 0.5
        
        p_symp = zeros(Float64, length(CAR_dates))
        p_detect_given_asymp = zeros(Float64, length(CAR_dates))
        p_detect_given_symp = zeros(Float64, length(CAR_dates))
        p_symp_given_detect = zeros(Float64, length(CAR_dates))
        p_detect_import = zeros(Float64, length(CAR_dates))
        
        for (i, d) in enumerate(CAR_dates)
            p_detect = CAR[i]
            
            if d <= Dates.Date("2021-11-01")
                p_symp[i] = 0.7
            else
                p_symp[i] = 0.4
            end
            
            if d <= Dates.Date("2021-11-01")
                # p_symp = 0.7 and p_d = 0.75
                p_detect_given_symp[i] = 0.9375
                p_detect_given_asymp[i] = 0.3125
                p_detect_import[i] = 0.98
            elseif d <= Dates.Date("2021-12-10")
                # p_symp = 0.4 and p_d = 0.75
                p_detect_given_symp[i] = 0.9
                p_detect_given_asymp[i] = 0.65
                p_detect_import[i] = p_detect
            else
                # p_symp = 0.4 and p_d = 0.5 noting that the line defined by 
                # p(d|s) = -p(a)/p(s) p(d|a) + p(d)/p(s) 
                # translates vertically when the CA changes and so we just find the point on 
                # the reduced CA line which is closest to the previous values 
                # (p(d|a), p(d|s)) = (0.65, 0.9)
                p_detect_given_asymp[i] = 0.3615
                p_detect_given_symp[i] = 0.7076
                p_detect_import[i] = p_detect
            end
            
            p_symp_given_detect[i] = p_detect_given_symp[i] * p_symp[i] / p_detect
            
        end
        
        p_detect = CAR
        
        γ = 0.5     # relative infectiousness of asymptomatic
        α_s = 1 ./ (p_symp .+ γ * (1 .- p_symp))    # contribution of TP attributable to symptomatic
        α_a = γ * α_s   # contribution of TP attributable to asymptomatic
        
        # for checking consistency when we miss an outbreak 
        consistency_multiplier = 5.0
        # prior parametes for the import model
        prior_alpha = 0.5
        prior_beta = 0.1
        # ema smoothing factor for the import model 
        ϕ = 0.5
        
        # use this to ensure the Constants type is correct
        S = typeof(p_symp)
        T = typeof(prior_alpha)
        
        return new{S, T}(
            p_detect, 
            p_symp,
            p_detect_given_symp,
            p_detect_given_asymp,
            p_symp_given_detect,
            α_s,
            α_a,
            p_detect_import,
            prior_alpha,
            prior_beta,
            ϕ,
            consistency_multiplier,
        )
    end
    
    function Constants(start_date, end_date, state)
        """
        Newer version constructor. This is distinguished from the other by the extra parameter
        `state` which enables us to produce jurisdiction specific scenarios. 
        """
        # date vectors to indicate which period we're in
        CAR_normal_before = Dates.Date(start_date):Dates.Day(1):Dates.Date("2021-12-07")
        CAR_low = Dates.Date("2021-12-08"):Dates.Day(1):Dates.Date("2022-01-17")
        CAR_normal_after = Dates.Date("2022-01-18"):Dates.Day(1):Dates.Date(end_date)
        # all dates
        CAR_dates = [
            CAR_normal_before;
            CAR_low; 
            CAR_normal_after
        ]
        
        # initialise the 
        CAR = 0.75 * ones(Float64, length(CAR_dates))
        
        # if we are one of the jurisdictions that got overwhelmed, apply time varying CAR
        if state in ("NSW", "VIC", "ACT", "QLD")
            idx = (CAR_dates .>= CAR_low[begin]) .& (CAR_dates .<= CAR_low[end])
            CAR[idx] .= 0.333
            idx = CAR_dates .> CAR_low[end]
            CAR[idx] .= 0.75
        end
        
        p_symp = zeros(Float64, length(CAR_dates))
        p_detect_given_asymp = zeros(Float64, length(CAR_dates))
        p_detect_given_symp = zeros(Float64, length(CAR_dates))
        p_symp_given_detect = zeros(Float64, length(CAR_dates))
        p_detect_import = zeros(Float64, length(CAR_dates))
        
        for (i, d) in enumerate(CAR_dates)
            p_detect = CAR[i]
            
            if d <= Dates.Date("2021-11-01")
                p_symp[i] = 0.7
            else
                p_symp[i] = 0.4
            end
            
            if d <= Dates.Date("2021-11-01")
                # p_symp = 0.7 and p_d = 0.75
                p_detect_given_symp[i] = 0.9375
                p_detect_given_asymp[i] = 0.3125
                p_detect_import[i] = 0.98
            elseif d <= Dates.Date("2021-12-07") || d >= Dates.Date("2022-01-18")
                # p_symp = 0.4 and p_d = 0.75
                p_detect_given_symp[i] = 0.9
                p_detect_given_asymp[i] = 0.65
                p_detect_import[i] = p_detect
            else
                # p_symp = 0.4 and p_d = 0.333
                p_detect_given_asymp[i] = 0.16885
                p_detect_given_symp[i] = 0.57923
                p_detect_import[i] = p_detect
            end
            
            p_symp_given_detect[i] = p_detect_given_symp[i] * p_symp[i] / p_detect
            
        end
        
        p_detect = CAR
        
        γ = 0.5     # relative infectiousness of asymptomatic
        α_s = 1 ./ (p_symp .+ γ * (1 .- p_symp))
        α_a = γ * α_s
        
        # other parameters 
        consistency_multiplier = 5.0
        # prior parametes for the import model
        prior_alpha = 0.5
        prior_beta = 0.1
        # ema smoothing factor 
        ϕ = 0.5
        
        S = typeof(p_symp)
        T = typeof(prior_alpha)
        
        return new{S, T}(
            p_detect,
            p_symp,
            p_detect_given_symp,
            p_detect_given_asymp,
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
    sim_realisation::Realisation
    sim_results::Results
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
        String, 
        NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}},
    }
    omicron_start_date::Date
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
        # put a small delay on this as we use the infection dates to index things but in 
        # the fitting we used the confirmation dates 
        omicron_start_date = Dates.Date("2021-11-15")
        omicron_dominant_date = Dates.Date("2021-12-15") - Dates.Day(5)
        
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