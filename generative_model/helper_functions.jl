using Random
using Distributions
using CSV 
using DataFrames

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