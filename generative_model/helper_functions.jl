using Random
using Distributions
using CSV 
using DataFrames
using Pipe
using Dates 

function NegativeBinomial2(μ, ϕ)
    """
    Function for parameterisation of the negative Binomial in terms of mean and variance.
    NOTE THAT THIS IS CURRENTLY NOT IN USE.
    """    
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomial(r, p)
end

function sample_negative_binomial_limit(s, p; approx_limit = 1000)
    """
    Samples from a NegBin(s, p) distribution. This uses a normal approximation 
    when mu is large (i.e. s > approx_limit) to get a 10x runtime improvement.
    """
    X = zero(Int)
    
    # mean of NegBin(s, p) => this will boil down to N*TP
    μ = s/p - s
    
    # if μ <= approx_limit
    #     X = rand(NegativeBinomial(s, p))
    # else
    #     σ = sqrt(s*(1-p)/p^2)
    #     X = ceil(Int, rand(Normal(μ, σ)))
    # end
    X = rand(NegativeBinomial(s, p))
    
    return X 
end


function sample_binomial_limit(n, p; approx_limit = 1000)
    """
    Samples from a Bin(n, p) distribution. This uses a normal approximation 
    for np > approx_limit or n(1-p) > approx_limit to acheive a 10x runtime 
    improvement.
    """
    X = zero(Int)
    
    if n*p <= approx_limit || n*(1-p) <= approx_limit
        X = rand(Binomial(n, p))
    else
        μ = n*p
        σ = sqrt(n*p*(1-p))
        X = ceil(Int, rand(Normal(μ, σ)))
    end
    
    return X 
end


function read_in_TP(date, state)
    
    # read in the reff file
    TP_file_name = "results/soc_mob_R"*date*".csv"
    # drop the first column 
    df = CSV.read(TP_file_name, DataFrame, drop=[1])
    # extract the unique states
    unique_states = unique(df[!,"state"])

    TP_dict_local = Dict{String, Array}()
    # get the dates 
    TP_dict_local["date"] = @pipe df |>
        filter(:state => ==(state), _) |> 
        filter(:type => ==("R_L"), _) |> 
        select(_, :date) |> 
        Matrix(_)[:]
    # create a vector of the indices
    TP_dict_local["t_indices"] = collect(1:length(TP_dict_local["date"]))
    
    TP_dict_import = Dict{String, Array}() 
    # get the dates 
    TP_dict_import["date"] = @pipe df |>
        filter(:state => ==(state), _) |> 
        filter(:type => ==("R_L"), _) |> 
        select(_, :date) |> 
        Matrix(_)[:]
    # create a vector of the indices
    TP_dict_import["t_indices"] = collect(1:length(TP_dict_import["date"]))

    for state in unique_states 
        # filter by state and RL and keep only TP trajectories 
        df_state_matrix = @pipe df |>
            filter(:state => ==(state), _) |> 
            filter(:type => ==("R_L"), _) |>
            select(_, Not(1:8)) |>
            Matrix(_)
        TP_dict_local[state] = df_state_matrix
        # filter by state and RL and keep only TP trajectories 
        df_state_matrix = @pipe df |>
            filter(:state => ==(state), _) |> 
            filter(:type => ==("R_I"), _) |>
            select(_, Not(1:8)) |>
            Matrix(_)
        TP_dict_import[state] = df_state_matrix
    end
    
    return (TP_dict_local, TP_dict_import)
    
end


function create_state_TP_matrices(forecast_start_date, date, state)
    """
    Read in the TP for a given state from csv with filedate, date. Adjust the indices 
    according to the forecast_start_date and then create matrices of local and import TP's. 
    The indices vector is used to allow for pre-forecast window infection dates. 
    """
    
    # read in the TP 
    (TP_dict_local, TP_dict_import) = read_in_TP(date, state)
    # adjust the indices so that the forecast start date is t = 0
    TP_indices = convert.(Int, Dates.value.(TP_dict_local["date"] - forecast_start_date))
    # make a matrix for a given state 
    TP_local = TP_dict_local[state]
    TP_import = TP_dict_import[state]
    
    return (TP_indices, TP_local, TP_import)
    
end


function read_in_susceptible_depletion(file_date)
    """
    Read in the posterior drawn susceptible_depletion factors. This will be sorted/sampled 
    in the same order as the posterior predictive TP's to ensure we use the appropriate
    posterior draws. 
    """
    
    susceptible_depletion = Vector(
        CSV.read(
            "results/forecasting/sampled_susceptible_depletion_"*file_date*".csv", 
            DataFrame, 
            drop=[1],
        )[:,1]
    )
    
    return susceptible_depletion
    
end

