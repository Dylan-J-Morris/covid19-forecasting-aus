from cmdstanpy import CmdStanModel

# to run the inference set run_inference to True in params
# path to the stan model 
# rho_model_gamma = "TP_model/fit_and_forecast/rho_model_gamma.stan"
rho_model_gamma = "TP_model/fit_and_forecast/rho_model_gamma_prod.stan"

# compile the stan model
model = CmdStanModel(stan_file=rho_model_gamma)