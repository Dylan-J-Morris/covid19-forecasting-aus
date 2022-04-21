using Random
using Distributions
using CSV 
using DataFrames
using Pipe
using Dates 
using DataStructures

function map_day_to_index_Z(day)
    """
    Map the day to the appropriate index for the infection array Z noting the 35 day
    padding at the start. 
    """
    return day + 31
    
end

function map_day_to_index_p(day)
    """
    Map the day to the appropriate index for the actual case data. This just wraps the UD 
    mapping for clarity in the code. 
    """
    return map_day_to_index_Z(day)
    
end


function map_day_to_index_UD(day)
    """
    Map the day to the appropriate index for the "detection" arrays U and D. This accounts 
    for the fact that the first index corresponds to day 0. 
    """
    # a branchless if statement for mapping between day and index 
    return (day <= 0) * 1 + (day > 0) * (day + 1)
    
end


function map_day_to_index_cases(day)
    """
    Map the day to the appropriate index for the actual case data. This just wraps the UD 
    mapping for clarity in the code. 
    """
    return map_day_to_index_UD(day)
    
end


function sample_inf_time_delta()
    # these are the calculated cdf from doing 
    # p = [pdf(Gamma(2.7, 1.0), x) for x in 1:21]
    # c = cumsum(p / sum(p))
    # we store c as a tuple as this treats it as a constant
    c = (
        0.23123495324177792, 0.5173637000503444, 0.7313700767804615, 0.8616192638327465, 0.9324257531818554, 0.968263917871601, 0.9855305708537918, 0.9935547309702418, 0.9971823528990054, 0.9987870872347043, 0.9994845898851373, 0.9997833904140334, 0.9999098409183975, 0.9999628010773516, 0.9999847842534525, 0.9999938383729761, 0.9999975420011928, 0.999999047825051, 0.9999996567609942, 0.9999999018144428, 1.0
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_inf_time_omicron()
    
    c = (
        0.4068944,  0.66971566, 0.82150389, 0.90524088, 0.95029074,
        0.97414539, 0.98663916, 0.99313045, 0.99648247, 0.99820507, 0.99908686,
        0.99953678, 0.99976574, 0.99988197, 0.99994087, 0.99997066, 0.9999857,
        0.99999329, 0.99999711, 0.99999903, 1.0
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_det_time_delta()
    c = (
        0.005481604976502295, 0.058918643630252246, 0.18959485142323212, 0.37100454410185607, 0.5556720050758871, 0.7101611913610963, 0.8230354111406564, 0.8977224959544793, 0.9435387039488634, 0.9700150992568369, 0.98459373725705, 0.9923072112983735, 0.9962539186531401, 0.998216511248062, 0.9991687521425704, 0.9996209873403942, 0.9998317584234413, 0.9999283682832782, 0.9999719976126707, 0.99999143977769, 1.0
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_det_time_omicron()
    c = (
        0.05465972865344886, 0.20227289941119453, 0.39199410142203933, 0.5719151467164512, 0.7163612824630651, 0.8207303835305828, 0.890853438104743, 0.9355256297540415, 0.9628506906875296, 0.9790374698285156, 0.9883806383087265, 0.9936593224349681, 0.9965885012866724, 0.9981892154091149, 0.9990524882440441, 0.999512733615265, 0.9997556426337206, 0.9998827037093614, 0.9999486387768106, 0.9999826099893431, 1.0
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_inf_time(omicron)
    """
    Sample infection times for num individuals based on the generation 
    interval distribution, Gamma(shape_gen, scale_gen). 
    """
    omicron == false && return sample_inf_time_delta()
    omicron == true && return sample_inf_time_omicron()
    
end


function sample_onset_time(omicron)
    """
    Sample incubation times for num individuals based on incubation period 
    distribution, Gamma(shape_inc, scale_inc). 
    """
    omicron == false && return sample_det_time_delta()
    omicron == true && return sample_det_time_omicron()
    
end


function sample_times(t, omicron)
    """
    Assuming an initial day t, sample the time until infection and onset for an individual 
    and only take the ceiling once we have assigned them. This helps reduce the effect of 
    ceiling errors induced by adding ceil numbers. 
    """
    infection_time = t + sample_inf_time(omicron)
    onset_time = infection_time + sample_onset_time(omicron)
    
    return (infection_time, onset_time)
    
end

function set_simulation_constants(forecast_start_date, forecast_end_date, state)
    """
    Creates structures of the constants and type mappings. . This includes all the dynamical
    constants: 
    - k = heterogeneity parameter
    - p_symp = probability of symptoms 
    - γ = relative infectiousness of asymptomatic individuals 
    - p_symp_given_detect = probability of symptoms given detection
    - p_asymp_given_detect = probability of being asymptomatic given detection
    - consistency_multiplier = chosen such that 
        sim_cases < consistency_multiplier * actual cases 
        results in cases being injected into the simulation. This is used to account 
        for superspreading events after periods of low incidence. 
    """
    # simulation_constants = Constants(forecast_start_date, forecast_end_date, state)
    simulation_constants = Constants(forecast_start_date, forecast_end_date)
    individual_type_map = IndividualTypeMap()

    return (simulation_constants, individual_type_map)
    
end


function NegativeBinomial2(μ, k)
    """
    Function for parameterisation of the negative Binomial in terms of mean and dispersion.
    """
    p = 1 / (1 + μ / k)
    
    return NegativeBinomial(k, p)

end


function sample_negative_binomial_limit(μ, k; approx_limit = 1000)
    """
    Samples from a NegBin(s, p) distribution. This uses a normal approximation 
    when mu is large (i.e. s > approx_limit) to get a 10x runtime improvement.
    """
    X = zero(Int)
    
    X = rand(NegativeBinomial2(μ, k))
    # mean of NegBin(s, p) => this will boil down to TP
    # if μ <= approx_limit
    #     X = rand(NegativeBinomial2(μ, ϕ))
    # else
    #     σ = sqrt(ϕ)
    #     # use the standard normal transform X = μ + σZ 
    #     X = ceil(Int, μ + σ * randn())
    # end
    
    return X 
    
end


function sample_binomial_limit(n, p; approx_limit = 1000)
    """
    Samples from a Bin(n, p) distribution. This uses a normal approximation 
    for np > approx_limit or n(1-p) > approx_limit to acheive a 10x runtime 
    improvement.
    """
    X = zero(Int)
    
    X = rand(Binomial(n, p))
    # if n*p <= approx_limit || n*(1-p) <= approx_limit
    #     X = rand(Binomial(n, p))
    # else
    #     μ = n*p
    #     σ = sqrt(n*p*(1-p))
    #     # use standard normal transform 
    #     X = ceil(Int, μ + σ * randn())
    # end
    
    return X 
    
end


function read_in_TP(
    date, 
    state; 
    strain = "Delta",
    adjust_TP = false,
)
    
    # read in the reff file
    TP_file_name = "results/" * date * "/"
        
    if adjust_TP 
        TP_file_name = TP_file_name * "soc_mob_R_adjusted_" * strain * date * ".csv"        
    else
        TP_file_name = TP_file_name * "soc_mob_R_" * strain * date * ".csv"
    end

    # drop the first column 
    df = CSV.read(TP_file_name, DataFrame, drop = [1])
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

    # read in the posterior draws 
    samples = CSV.read("results/" * date * "/posterior_sample_" * date * ".csv", DataFrame)
    R_I = samples.R_I 
    R_I_omicron = samples.R_I
    voc_effect = zeros(size(R_I))

    if strain == "Delta"
        voc_effect .= samples.voc_effect_delta
    else
        voc_effect .= samples.voc_effect_omicron
    end

    R_I_omicron = R_I_omicron .* voc_effect

    # initialise an import matrix to use for each loop
    TP_import_matrix = zeros(length(TP_dict_import["t_indices"]), 2000)

    omicron_start_date = Dates.Date("2021-11-15")

    for state in unique_states 
        # filter by state and RL and keep only TP trajectories 
        df_state_matrix = @pipe df |>
            filter(:state => ==(state), _) |> 
            filter(:type => ==("R_L"), _) |>
            select(_, Not(1:8)) |>
            Matrix(_)
        TP_dict_local[state] = df_state_matrix
        
        # fill the pre-Omicron R_I
        TP_import_matrix[TP_dict_import["date"] .< omicron_start_date, :] .= R_I[1:2000]'
        # fill the post-Omicron R_I
        TP_import_matrix[TP_dict_import["date"] .>= omicron_start_date, :] .= R_I_omicron[1:2000]'
        # add to the TP dict
        TP_dict_import[state] = TP_import_matrix
    end
    
    return (TP_dict_local, TP_dict_import)
    
end


function get_single_state_TP(
    forecast_start_date, 
    date, 
    state; 
    strain = "Delta",
    adjust_TP = false,
)
    """
    Read in the TP for a given state from csv with filedate, date. Adjust the indices 
    according to the forecast_start_date and then create matrices of local and import TP's. 
    The indices vector is used to allow for pre-forecast window infection dates. 
    """
    
    # read in the TP 
    (TP_dict_local, TP_dict_import) = read_in_TP(
        date, 
        state, 
        strain = strain, 
        adjust_TP = adjust_TP
    )
    # adjust the indices so that the forecast start date is t = 0
    TP_dates = TP_dict_local["date"]
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
    
    # read in the reff file
    file_name = "results/" * 
        file_date * 
        "/" * 
        "sampled_susceptible_depletion_" * 
        file_date * 
        ".csv"
    
    # susceptible_depletion = Vector(CSV.read(file_name, DataFrame, drop = [1],)[:, 1])
    susceptible_depletion = Matrix(CSV.read(file_name, DataFrame, drop = [1],))
    
    return susceptible_depletion
    
end


function read_in_prop_omicron(file_date, state)
    """
    Read in the posterior drawn susceptible_depletion factors. This will be sorted/sampled 
    in the same order as the posterior predictive TP's to ensure we use the appropriate
    posterior draws. 
    """
    
    # prefix and suffix for the file name
    file_name_prefix = "results/" * 
        file_date * 
        "/"  
    
    file_name_suffix = "_" *
        state *  
        file_date * 
        ".csv"
    
    # these vars are used to calculate the omicron proportion
    prop_vars = ("m0", "m1", "r", "tau")
    # prop_vars = ("m0", "r", "tau")
    # create vector of vectors to store the various parameter samples
    res = Dict{String, Vector{Float64}}()
    
    for v in prop_vars
        file_name = file_name_prefix * v * file_name_suffix
        res[v] = Vector(CSV.read(file_name, DataFrame, drop = [1],)[:, 1])
    end
    
    return res
    
end


function read_in_cases(
	date, 
	rng; 
    apply_delay = true,
    use_mean = false,
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
    
	if apply_delay
        # shift them appropriately
        shape_rd = 2.33
        scale_rd = 1.35
        # sample from delay distribtuion
        rd = zeros(length(confirm_dates))
        if use_mean
            rd .= mean(Gamma(shape_rd, scale_rd))
        else
            # use the discretised pmf to sample the reporting delays 
            rd = round.(rand(Gamma(shape_rd, scale_rd), length(confirm_dates)))
        end
        
        confirm_dates = confirm_dates - rd * Dates.Day(1)
    end

	# initialise array for complete_onset_dates
	complete_dates = deepcopy(confirm_dates)
	# fill the array with the most informative date 
	complete_dates[.!is_confirmation] = Date.(
		convert(Vector, df.date_onset[.!is_confirmation])
	)
	complete_dates[is_confirmation] = confirm_dates[is_confirmation]

	# other useful fields
	state = df.state
	import_status = convert.(Int, df.import_status .== "imported")

	# create vector of all dates from the first (good) reported data in 2020 to the most
    # recent data in the linelist. 
    # Note that in previous weeks, this was an issue as the data being provided 
    # had onset dates which were imputed (and shouldn't have been)
	dates_since_start = range(
		Date("2020-03-01"), 
		# Dates.Date(date),
		maximum(complete_dates),
		step = Day(1),
	)

	complete_dates_local = complete_dates[import_status .== 0]
	complete_dates_import = complete_dates[import_status .== 1]
	state_local = state[import_status .== 0]
	state_import = state[import_status .== 1]

	# construct df to hold all the linelists	
	new_df = DataFrame("date" => dates_since_start)

	# construct df to hold all the linelists	
    date_vec = dates_since_start
	local_case_dict = Dict{String, Vector{Int}}()
	import_case_dict = Dict{String, Vector{Int}}()
	
	# initialise arrays in the dictionary to copy into 
	for s in unique(state) 
		# filter by state	
		local_case_dict[s] = zeros(Int, length(dates_since_start))
		import_case_dict[s] = zeros(Int, length(dates_since_start))
	end
    
	# initialise arrays to hold daily cases in 
	local_cases = zeros(Int, length(dates_since_start))
	import_cases = zeros(Int, length(dates_since_start))
	
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
		local_case_dict[s] .= Int.(local_cases)
		import_case_dict[s] .= Int.(import_cases)
		# reset the case vectors
		local_cases .= 0
		import_cases .= 0
	end

    return (date_vec, local_case_dict, import_case_dict)
    
end