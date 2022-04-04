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
    return day + 36
    
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
    c = (
        0.110973577180615, 0.27329217944045014, 0.25160723139643126, 0.16887549417969105, 0.09717236270111923, 0.051042787594637214, 0.025243529370285504, 0.011961070187119777, 0.005489051747539894, 0.002457257541327242, 0.0010784609533814435, 0.0004657307434988743, 0.00019843640955813553, 8.359282266014512e-5, 3.487287711664492e-5, 1.4425964749702236e-5, 5.9238009894200945e-6, 2.4167578116632567e-6, 9.802978433867657e-7, 3.9558551639514304e-7, 1.5889321780907003e-7, 6.355444004537977e-8
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_inf_time_omicron()
    c = (
        0.3380452569369169, 0.282235881463163, 0.1714064992645785, 0.09670470439110702, 0.052693788873186195, 0.02813279681100046, 0.014819457215524679, 0.0077324169685374314, 0.00400600259867144, 0.0020640233460315987, 0.0010587877673257182, 0.0005411833219447294, 0.0002757928163397227, 0.00014019273315940716, 7.110982970736625e-5, 3.600155092866423e-5, 1.819710629544052e-5, 9.184531291818476e-6, 4.6297243425726e-6, 2.3310713684505575e-6, 1.17248764846839e-6, 5.891909304320088e-7
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_det_time_delta()
    c = (
        0.0011126391007854188, 0.02500420040955002, 0.09212834653575538, 0.1598528912582521, 0.18672124923850889, 0.1713965419627255, 0.1338612067897289, 0.093130770831524, 0.0594080686600261, 0.03542290867839186, 0.020011392631891424, 0.010816684757237507, 0.005635561270221634, 0.0028462260340724726, 0.0013996698416887197, 0.0006725943266925051, 0.00031674442983966136, 0.00014653036428194404, 6.672230620232283e-5, 2.9954689646549925e-5, 1.3277780752749815e-5, 5.8181022243337385e-6
    )
    rn = rand()
    
    return findfirst(rn <= cᵢ for cᵢ in c)
    
end


function sample_det_time_omicron()
    c = (
        0.018354699747111635, 0.1034401575402339, 0.17391188097485316, 0.18821550005575002, 0.16337692773297102, 0.12428616295092726, 0.08660866309282085, 0.05669614954427924, 0.03541801295237932, 0.021336008540979234, 0.012485118515689909, 0.007134542168699892, 0.003997226760909039, 0.0022024006815003225, 0.001196242657140024, 0.0006417403547129652, 0.0003405585380316517, 0.0001790079255101512, 9.329612752480465e-5, 4.8256449656731913e-5, 2.4790108915296424e-5, 1.265657940358179e-5
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

function set_simulation_constants(state; p_detect_omicron = 0.5)
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
    simulation_constants = Constants(state, p_detect_omicron = p_detect_omicron)
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
    p_detect_omicron = 0.5, 
    strain = "Delta",
    adjust_TP = false,
)
    
    # read in the reff file
    TP_file_name = "results/" * 
        date * 
        "/" * 
        string(round(Int, p_detect_omicron * 100)) * 
        "_case_ascertainment" *
        "/"
    
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


function get_single_state_TP(
    forecast_start_date, 
    date, 
    state; 
    p_detect_omicron = 0.5,
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
        p_detect_omicron = p_detect_omicron,
        strain = strain, 
        adjust_TP = adjust_TP
    )
    # adjust the indices so that the forecast start date is t = 0
    TP_dates = TP_dict_local["date"]
    TP_indices = convert.(
        Int, Dates.value.(TP_dict_local["date"] - forecast_start_date)
    ) 
    # make a matrix for a given state 
    TP_local = TP_dict_local[state]
    TP_import = TP_dict_import[state]
    
    return (TP_indices, TP_local, TP_import)
    
end


function read_in_susceptible_depletion(file_date; p_detect_omicron = 0.5)
    """
    Read in the posterior drawn susceptible_depletion factors. This will be sorted/sampled 
    in the same order as the posterior predictive TP's to ensure we use the appropriate
    posterior draws. 
    """
    
    # read in the reff file
    file_name = "results/" * 
        file_date * 
        "/" * 
        string(round(Int, p_detect_omicron * 100)) *
        "_case_ascertainment" * 
        "/" * 
        "sampled_susceptible_depletion_" * 
        file_date * 
        ".csv"
    
    susceptible_depletion = Vector(CSV.read(file_name, DataFrame, drop = [1],)[:, 1])
    
    return susceptible_depletion
    
end


function read_in_prop_omicron(file_date, state; p_detect_omicron = 0.5)
    """
    Read in the posterior drawn susceptible_depletion factors. This will be sorted/sampled 
    in the same order as the posterior predictive TP's to ensure we use the appropriate
    posterior draws. 
    """
    
    # prefix and suffix for the file name
    file_name_prefix = "results/" * 
        file_date * 
        "/" * 
        string(round(Int, p_detect_omicron * 100)) *
        "_case_ascertainment" * 
        "/"  
    
    file_name_suffix = "_" *
        state *  
        file_date * 
        ".csv"
    
    # these vars are used to calculate the omicron proportion
    prop_vars = ("m0", "r", "tau")
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
	apply_inc = false, 
	omicron_dominant_date = nothing,
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
            p = [pdf(Gamma(shape_rd, scale_rd), x) for x in 0:21]
            c = cumsum(p / sum(p))
            rd = [findfirst(rand(rng) <= cᵢ for cᵢ in c) for _ in 1:length(confirm_dates)]
        end
        
        confirm_dates = confirm_dates - round.(rd) * Dates.Day(1)
    end

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
        inc = zeros(length(complete_dates))
        
        if use_mean 
            inc .= mean(Gamma(shape_inc, scale_inc))
        else
		    inc = rand(rng, Gamma(shape_inc, scale_inc), length(complete_dates))
        end
		
		# now apply different incubation period for Omicron 
		apply_omicron = !isnothing(omicron_dominant_date) ? true : false
		if apply_omicron
			is_omicron = complete_dates .>= Dates.Date(omicron_dominant_date)
			shape_inc_omicron = 3.33
			scale_inc_omicron = 1.34
            
            inc = zeros(length(complete_dates))
            if use_mean 
                inc_omicron .= mean(Gamma(shape_inc_omicron, scale_inc_omicron))
            else
                inc_omicron = rand(
                    rng, 
                    Gamma(shape_inc_omicron, scale_inc_omicron), 
                    length(complete_dates),
                )
            end
			# add the incubation for omicron dates in 
			inc = (1 .- is_omicron) .* inc + is_omicron .* inc_omicron
		end
        
		complete_dates = complete_dates - ceil.(inc) * Day(1)
        
	end

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

