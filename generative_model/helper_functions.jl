using Random
using Distributions

function sample_negative_binomial_limit(s, p; approx_limit = 1000)
    """
    Samples from a NegBin(s, p) distribution. This uses a normal approximation 
    when s is large (i.e. s > approx_limit) to get a 10x runtime improvement.
    """
    X = zero(Int)
    
    if s <= approx_limit
        X = rand(NegativeBinomial(s, p))
    else
        mu = s/p - s
        sig = sqrt(s*(1-p)/p^2)
        X = round(Int, rand(Normal(mu, sig)))
    end
    
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
        mu = n*p
        sig = sqrt(n*p*(1-p))
        X = round(Int, rand(Normal(mu, sig)))
    end
    
    return X 
end

function read_in_susceptible_depletion(file_date)
    """
    Using the stan fit posterior_sample_YYYY-MM-DD.csv we extract the susceptible_depletion
    posterior draws. 
    """
    
    samples = CSV.read("results/posterior_sample_"*file_date*".csv",DataFrame)
    susceptible_depletion = samples.susceptible_depletion_factor
    
    return susceptible_depletion
end