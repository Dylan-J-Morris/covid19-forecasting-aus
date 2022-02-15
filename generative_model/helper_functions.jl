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

function sample_infection_time(;omicron=false)
    """
    Sample infection times for num individuals based on the generation 
    interval distribution, Gamma(shape_gen, scale_gen). 
    """

    (shape_gen, scale_gen) = (2.75, 1.00)
    (shape_gen_omicron, scale_gen_omicron) = (1.58, 1.32)
    
    shape = (1 - omicron) * shape_gen + omicron * shape_gen_omicron
    scale = (1 - omicron) * scale_gen + omicron * scale_gen_omicron
    
    infection_time = ceil(Int, rand(Gamma(shape, scale)))
    
    return infection_time
    
end


function sample_onset_time(;omicron=false)
    """
    Sample incubation times for num individuals based on incubation period 
    distribution, Gamma(shape_inc, scale_inc). 
    """
    
    (shape_inc, scale_inc) = (5.807, 0.948)
    (shape_inc_omicron, scale_inc_omicron) = (3.33, 1.34)
    
    shape = (1 - omicron) * shape_inc + omicron * shape_inc_omicron
    scale = (1 - omicron) * scale_inc + omicron * scale_inc_omicron
    
    onset_time = ceil(Int, rand(Gamma(shape, scale)))
    
    return onset_time
    
end


function set_simulation_constants(state)
    """
    Contains the assumptions for simulation parameters. This includes all the dynamical
    constants: 
        - k = heterogeneity parameter
        - p_symp = probability of symptoms 
        - γ = relative infectiousness of asymptomatic individuals 
        - p_symp_given_detect = probability of symptoms given detection
        - p_asymp_given_detect = probability of being asymptomatic given detection
        - consistency_multiplier = chosen such that sim_cases < 
            consistency_multiplier*actual cases 
            results in cases being injected into the simulation. This is used to account 
            for superspreading events after periods of low incidence. 
    These values are stored in sim_constants which is a dictionary indexed by the 
    parameter name and ultimately stored on the stack in a SimulationParameters object. 
    """
    # get the simulation constants
    simulation_constants = Constants(state)
    # mapping between types 
    individual_type_map = IndividualTypeMap()

    return (simulation_constants, individual_type_map)
    
end


function sample_negative_binomial_limit(s, p; approx_limit = 1000)
    """
    Samples from a NegBin(s, p) distribution. This uses a normal approximation 
    when mu is large (i.e. s > approx_limit) to get a 10x runtime improvement.
    """
    X = zero(Int)
    
    # mean of NegBin(s, p) => this will boil down to N*TP
    μ = s/p - s
    
    if μ <= approx_limit
        X = rand(NegativeBinomial(s, p))
    else
        σ = sqrt(s*(1-p)/p^2)
        X = ceil(Int, rand(Normal(μ, σ)))
    end
    # X = rand(NegativeBinomial(s, p))
    
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
    unique_states = unique(df[!, "state"])

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

function read_in_cases(
	date, 
	rng; 
	apply_inc = false, 
	omicron_dominant_date = nothing
)

	# read in the reff file
	case_file_name = "data/interim_linelist_" * date * ".csv"
	# drop the first column 
	df = CSV.read(case_file_name, DataFrame)
	# indicator for the NA dates
	is_confirmation = df.date_onset .== "NA"
	# get the confirmation dates 
	confirm_dates = convert(Vector, df.date_confirmation)
	confirm_dates = Date.(confirm_dates)
	# shift them appropriately
	shape_rd = 1.28
	scale_rd = 2.31
	# sample from delay distribtuion
	rd = rand(rng, Gamma(shape_rd, scale_rd), length(confirm_dates))
	confirm_dates = confirm_dates - round.(rd) * Day(1)
	# adjust confirmation dates to get to onset 
	# rd = ceil(Int, mean(Gamma(shape_rd, scale_rd)))
	# confirm_dates = confirm_dates .- rd * Day(1)

	# initialise array for complete_onset_dates
	complete_dates = deepcopy(confirm_dates)
	# fill the array with the most informative date 
	complete_dates[.!is_confirmation] = Date.(
		convert(Vector, df.date_onset[.!is_confirmation])
	)
	complete_dates[is_confirmation] = confirm_dates[is_confirmation]
	
	# if want to apply delay, subtract an incubation period per individual 
	if apply_inc
		shape_inc = 5.807  # taken from Lauer et al. 2020 #1.62/0.418
		scale_inc = 0.948  # 0.418
		inc = rand(rng, Gamma(shape_inc, scale_inc), length(complete_dates))
		
		# now apply different incubation period for Omicron 
		apply_omicron = !isnothing(omicron_dominant_date) ? true : false
		if apply_omicron
			is_omicron = complete_dates .>= Dates.Date(omicron_dominant_date)
			shape_inc_omicron = 3.33
			scale_inc_omicron = 1.34
			inc_omicron = rand(
				rng, 
				Gamma(shape_inc_omicron, scale_inc_omicron), 
				length(complete_dates),
			)
			# add the incubation for omicron dates in 
			inc = (1 - is_omicron) * inc + is_omicron * inc_omicron
		end
		
		complete_dates = complete_dates - round.(inc) * Day(1)
	end

	# other useful fields
	state = df.state
	import_status = convert.(Int, df.import_status .== "imported")

	# construct the full timeseries of counts
	dates_since_start = range(
		Date("2020-03-01"), 
		maximum(complete_dates),
		step = Day(1),
	)

	complete_dates_local = complete_dates[import_status .== 0]
	complete_dates_import = complete_dates[import_status .== 1]
	state_local = state[import_status .== 0]
	state_import = state[import_status .== 1]

	# vectors to hold the number of cases each day
	local_cases = zeros(Int, length(dates_since_start))
	import_cases = zeros(Int, length(dates_since_start))

	# construct df to hold all the linelists	
	new_df = DataFrame(
		"date" => dates_since_start
	)

	# construct df to hold all the linelists	
	local_case_dict = Dict()
	import_case_dict = Dict()
	local_case_dict["date"] = dates_since_start
	import_case_dict["date"] = dates_since_start
	
	# initialise arrays in the dictionary to copy into 
	for s in unique(state) 
		# filter by state	
		local_case_dict[s] = zeros(length(dates_since_start))
		import_case_dict[s] = zeros(length(dates_since_start))
	end
	# initialise arrays to hold daily cases in 
	local_cases = zeros(length(dates_since_start))
	import_cases = zeros(length(dates_since_start))
	
	# loop over states and then sum the number of cases on that day
	for s in unique(state)
		# filter by state	
		complete_dates_local_tmp = complete_dates_local[state_local .== s]
		complete_dates_import_tmp = complete_dates_import[state_import .== s]
		# get cases on each day 
		local_cases .= sum.(
			@views complete_dates_local_tmp .== dss for dss in dates_since_start
		)
		import_cases .= sum.(
			@views complete_dates_import_tmp .== dss for dss in dates_since_start
		)
		# append to the df with a deepcopy to avoid a 0 d
		local_case_dict[s] .= local_cases
		import_case_dict[s] .= import_cases
		# reset the case vectors
		local_cases .= 0
		import_cases .= 0
	end

    return (local_case_dict, import_case_dict)
    
end

