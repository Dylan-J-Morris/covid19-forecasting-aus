function ρ(X, X_sim; metric = "absolute")
    """
    Calculate the sum of the absolute difference at each time point between observed data 
    X and simulated data X_sim noting that each row of these corresponds to a realisation. 
    that 
    """
    
    ϵ = zeros(size(X, 2))
    
    if metric == "absolute"
        ϵ .= sum(abs.(X - X_sim), dims = 1)[:]
    elseif metric == "MSE"
        ϵ .= sum((X - X_sim) .^ 2, dims = 1)[:]
    elseif metric == "RMSE"
        N = size(X, 1)
        ϵ .= sqrt.(1 / N * sum((X - X_sim) .^ 2, dims = 1)[:])
    end
    
    return ϵ
    
end

function final_check(D_good, local_cases; p = 0.1, metric = "absolute")
    """
    Determines the top p×100% of simulations based on some metric ρ(X, X_sim) and returns 
    those indices.  
    """
    # get only the observed cases over the period we have cases for
    D_observed = D_good[begin:length(local_cases), 1, :] + 
        D_good[begin:length(local_cases), 2, :]
    # tile local_cases to easily calculate the distance ρ(X, X_sim)
    local_cases_mat = repeat(local_cases, 1, size(D_observed, 2))
    
    # calculate distance between data and the simulations
    ϵ = ρ(local_cases_mat, D_observed, metric = metric)
    # determine the sim indices of the top p×100% 
    good_inds = ϵ .< quantile(ϵ, p)
    
    return good_inds 
    
end

function count_cases2!(
    case_counts, 
    forecast::Forecast, 
    sim,
)

    D = forecast.sim_realisations.D
    
    day = 1
    tmp = 0
    
    for i in 1:length(case_counts)
        case_counts[day] = D[i, 1, sim] + D[i, 2, sim]
        day += 1
    end
    
    return nothing
    
end

function calculate_bounds2(local_cases, τ, state)
    
    # tolerance for when we consider cases to be stable
    ϵ = 0.0
    
    # multipliers on the n-day average 
    (ℓ, u) = (0.5, 2.5)
    
    # observation period 
    T = length(local_cases)
    # the slope over the n-day period 
    m = 0.0
    Mₜ = zeros(Float64, T)
    
    Lₜ = zeros(Float64, T)
    Uₜ = zeros(Float64, T) 
    
    # consider τ = 3 and t = (0, 1, 2), clearly n = 2 - 0 = 2
    n = τ - 1
    
    n2 = 7
    
    for t in 1:T
        
        # if state == "NSW"  
        #     if t < 30
        #         (ℓ, u) = (0, 10)
        #     elseif t < T - 7
        #         (ℓ, u) = (0.5, 2.0)
        #     else
        #         (ℓ, u) = (0.5, 2.0)
        #     end
        # elseif state == "NT"
        #     if t < 30
        #         (ℓ, u) = (0, 4.0)
        #     elseif t < T - 7
        #         (ℓ, u) = (0.25, 4.0)
        #     else
        #         (ℓ, u) = (0.5, 3.0)
        #     end
        # end
        
        if t < T - 7
            (ℓ, u) = (0.25, 2.0)
        else
            (ℓ, u) = (0.5, 2.0)
        end
        
        
        # approximate the slope naively 
        if t < T - n
            m = 1 / n * (
                local_cases[t + n] - 
                local_cases[t]
            )
        else
            m = 1 / n * (
                local_cases[t] - 
                local_cases[t - n]
            ) 
        end
        
        Mₜ[t] = m
        
        # depending on the sign of the slope, take average of future or past cases 
        if m < -ϵ
            Lₜ[t] = ℓ * mean(
                local_cases[t:min(T, t + n)]
            ) 
            Uₜ[t] = u * mean(
                local_cases[max(1, t - n):t]
            ) 
            # Lₜ[t] = mean(local_cases[t:min(T, t + n)]) 
            # Uₜ[t] = mean(local_cases[max(1, t - n):t]) 
        elseif m > ϵ
            Lₜ[t] = ℓ * mean(
                local_cases[max(1, t - n):t]
            ) 
            Uₜ[t] = u * mean(
                local_cases[t:min(T, t + n)]
            ) 
            # Lₜ[t] = mean(local_cases[max(1, t - 2):t]) 
            # Uₜ[t] = mean(local_cases[t:min(T, t + 2)]) 
        else
            n2 = n ÷ 2
            Lₜ[t] = ℓ * mean(
                local_cases[
                    max(1, t - n2):min(T, t + n2)
                ]
            ) 
            Uₜ[t] = u * mean(
                local_cases[
                    max(1, t - n2):min(T, t + n2)
                ]
            ) 
            # Lₜ[t] = mean(local_cases[max(1, t - n2):min(T, t + n2)]) 
            # Uₜ[t] = mean(local_cases[max(1, t - n2):min(T, t + n2)]) 
        end
        
        # adjust the bounds for periods with low cases
        if Lₜ[t] < 50
            Lₜ[t] = 0
        end
        
        if Uₜ[t] < 50
            Uₜ[t] = 50
        end
        
        if isnan(Lₜ[t])
            Lₜ[t] = Lₜ[t-1]
        end
        
        if isnan(Uₜ[t])
            Uₜ[t] = Uₜ[t-1]
        end
    end
    
    return (Lₜ, Uₜ) 
    
end