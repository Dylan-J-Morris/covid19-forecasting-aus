using Random
using Distributions

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

function sample_negative_binomial_limit(s, p; approx_limit = 500)
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

function sample_binomial_limit(n, p; approx_limit = 500)
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
    
    samples = CSV.read("daily_Reff/data/samples_mov_gamma.csv",DataFrame)
    susceptible_depletion = samples.susceptible_depletion_factor
    
    return susceptible_depletion
end