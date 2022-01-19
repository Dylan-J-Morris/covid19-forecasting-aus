from sys import argv
from run_stan import get_data_for_posterior, run_stan, plot_and_save_posterior_samples

def main(data_date):
    """
    Runs the stan model in parts to cut down on memory. 
    """
    
    # get_data_for_posterior(data_date=data_date)
    # run_stan(data_date=data_date)
    plot_and_save_posterior_samples(data_date=data_date)
    
    return None 
    

if __name__ == "__main__":
    data_date = argv[1]
    main(data_date)
