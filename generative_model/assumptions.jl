struct JurisdictionAssumptions

    simulation_start_dates::Dict{String, String}
    pop_sizes::Dict{String, Int}
    initial_conditions::Dict{
        String, NamedTuple{(:S, :A, :I), Tuple{Int64, Int64, Int64}}
    }
    omicron_dominant_date::Date

    function JurisdictionAssumptions()
        simulation_start_dates = Dict{String, String}(
            "NSW" => "2021-06-23",
            "QLD" => "2021-11-01",
            "SA" => "2021-11-01",
            "TAS" => "2021-11-01",
            "WA" => "2021-12-15",
            "ACT" => "2021-08-01",
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
            "NSW" => (S = 5, A = 8, I = 0),
            "QLD" => (S = 0, A = 0, I = 0),
            "SA" => (S = 0, A = 0, I = 0),
            "TAS" => (S = 0, A = 0, I = 0),
            "VIC" => (S = 20, A = 20, I = 0),
            "WA" => (S = 3, A = 2, I = 0),
            "ACT" => (S = 0, A = 0, I = 0),
            "NT" => (S = 3, A = 2, I = 0),
        )
        
        return new(
            simulation_start_dates, 
            pop_sizes, 
            initial_conditions, 
            omicron_dominant_date,
        )
        
    end
    

end