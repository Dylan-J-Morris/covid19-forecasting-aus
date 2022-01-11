using CSV
using DataFrames
using Dates
using Distributions
using Random

function read_in_cases(date, rng; apply_inc = false)

	# read in the reff file
	case_file_name = "daily_Reff/data/interim_linelist_"*date*".csv"
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
    # adjust confirmation dates 
    confirm_dates = confirm_dates - round.(rd) * Day(1)

    # initialise array for complete_onset_dates
    complete_dates = deepcopy(confirm_dates)
	# fill the array with the most informative date 
    complete_dates[.!is_confirmation] = Date.(convert(Vector, df.date_onset[.!is_confirmation]))
    complete_dates[is_confirmation] = confirm_dates[is_confirmation]
	
	# if want to apply delay, subtract an incubation period per individual 
	if apply_inc
		shape_inc = 5.807  # taken from Lauer et al. 2020 #1.62/0.418
		scale_inc = 0.948  # 0.418
		inc = rand(rng, Gamma(shape_inc, scale_inc), length(complete_dates))
		complete_dates = complete_dates - round.(inc) * Day(1)
	end
	
	# other useful fields
    state = df.state
    import_status = convert.(Int, df.import_status .== "imported")
	
	# construct the full timeseries of counts
	dates_since_start = range(
		Date("2020-03-01"), 
		maximum(complete_dates),
		step = Day(1)
	)
	
	# vectors to hold the number of cases each day
	local_cases = zeros(length(dates_since_start))
	import_cases = zeros(length(dates_since_start))
	
	# construct df to hold all the linelists	
	new_df = DataFrame(
		"date" => dates_since_start
	)
	
	# construct df to hold all the linelists	
	local_case_dict = Dict()
	import_case_dict = Dict()
	local_case_dict["date"] = dates_since_start
	import_case_dict["date"] = dates_since_start
	
	# loop over states and then sum the number of cases on that day
	for s in unique(state)
		for i in eachindex(dates_since_start)
			local_cases[i] = sum(
				(complete_dates .== dates_since_start[i]) .&
				(import_status .== 0) .& 
				(state .== s)
			)
			import_cases[i] = sum(
				(complete_dates .== dates_since_start[i]) .&
				(import_status .== 1) .& 
				(state .== s)
			)
		end
		# append to the df with a deepcopy to avoid a 0 df
		local_case_dict[s] = deepcopy(convert.(Int, local_cases))
		import_case_dict[s] = deepcopy(convert.(Int, import_cases))
		# reset the case vectors
		local_cases .= 0
		import_cases .= 0
	end

    return (local_case_dict, import_case_dict)
    
end