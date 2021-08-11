# A collection of helper functions that are used throughout. This file is aimed to avoid replication of code.

import pandas as pd

def read_in_NNDSS(date_string):
    """
    A general function to read in the NNDSS data. Alternatively this can be manually set to read in the linelist instead.
    Args:
        date_string: (str) a string of the date of the data file.

    Returns:
        A dataframe of all NNDSS data.
    """

    from datetime import timedelta
    import glob
    from params import use_linelist, assume_local_cases_if_unknown

    if not use_linelist: 
        # On occasion the date string in NNDSS will be missing the leading 0  (e.g. 2Aug2021 vs 02Aug2021). In this case manually add the zero.
        case_file_date = pd.to_datetime(date_string).strftime("%d%b%Y") 
        path = "data/COVID-19 UoM "+case_file_date+"*.xlsx"

        for file in glob.glob(path): # Allows us to use the * option
            df = pd.read_excel(file, 
                            parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'], 
                            dtype= {'PLACE_OF_ACQUISITION':str})
        if len(glob.glob(path))!=1:
            print("There are %i files with the same date" %len(glob.glob(path)))
        if len(glob.glob(path)) == 0:
            raise FileNotFoundError("NNDSS no found. Did you want to use a linelist? Or is the file named wrong?")

        # Fixes errors in updated python versions
        df.TRUE_ONSET_DATE = pd.to_datetime(df.TRUE_ONSET_DATE, errors='coerce') 
        df.NOTIFICATION_DATE = pd.to_datetime(df.NOTIFICATION_DATE, errors='coerce')

        # Find most representative date 
        df['date_inferred'] = df.TRUE_ONSET_DATE
        df.loc[df.TRUE_ONSET_DATE.isna(),'date_inferred'] = df.loc[df.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
        df.loc[df.date_inferred.isna(),'date_inferred'] = df.loc[df.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)
    
        # The first 4 digits is the country code. We use this to determin if the cases is local or imported. We can choose which assumption we keep. This should be set to true during local outbreak waves.
        if assume_local_cases_if_unknown:
            # Fill blanks with local code
            df.PLACE_OF_ACQUISITION.fillna('11019999',inplace=True)
        else:
            # Fill blanks with unknown international code
            df.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) 

        # IMPORTANT NOTE: State of infection is determined by the STATE column, not the PLACE_OF_ACQUISITION column

        # Set imported cases, local cases have 1101 as first 4 digits.
        df['imported'] = df.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[:4]!='1101' else 0)
        df['local'] = 1 - df.imported

        return df

    else:
        # The linelist, currently produce by Gerry Ryan, has had the onset dates and local / imported status vetted by a human. This can be a lot more reliable during an outbreak.
        
        case_file_date = pd.to_datetime(date_string).strftime("%Y-%m-%d")
        path = "data/*linelist_"+case_file_date+"*.csv"
        for file in glob.glob(path): # Allows us to use the * option
            df = pd.read_csv(file)

        if len(glob.glob(path)) == 0:
            raise FileNotFoundError("Linelist no found. Did you want to use NNDSS?")

        df['date_onset']= pd.to_datetime(df['date_onset'], errors='coerce') 
        df['date_detection'] = pd.to_datetime(df['date_detection'], errors='coerce')

        df['date_inferred'] = df['date_onset']
        df.loc[df['date_onset'].isna(),'date_inferred'] = df.loc[df['date_onset'].isna()]['date_detection'] - timedelta(days=3) # Fill missing days

        df['imported'] = [1 if stat=='imported' else 0 for stat in df['import_status']]
        df['local'] = 1 - df.imported
        df['STATE'] = df['state']
        # df['NOTIFICATION_RECEIVE_DATE'] = df['date_detection'] # Only used by EpyReff. Possible improvement here.
        return df


def read_in_Reff_file(file_date, VoC_flag=None, scenario=''):
    """
    Read in Reff h5 file produced by generate_RL_forecast. 
    Args:
        file_date: (date as string) date of data file
        VoC_date: (date as string) date from which to increase Reff by VoC
    """
    from scipy.stats import beta
    from params import VoC_start_date, use_vaccine_effect
    
    if file_date is None:
        raise Exception('Need to provide file date to Reff read.')

    file_date = pd.to_datetime(file_date).strftime("%Y-%m-%d")
    df_forecast = pd.read_hdf('results/soc_mob_R'+file_date+scenario+'.h5', key='Reff')

    if (VoC_flag != '') and (VoC_flag is not None):
        VoC_start_date  = pd.to_datetime(VoC_start_date)

        if VoC_flag == 'Alpha':
            print('This VoC will be deprecated in future.')
            # Here we apply the  beta(6,14)+1 scaling from VoC to the Reff.
            # We do so by editing a slice of the data frame. Forgive me for my sins.
            row_bool_to_apply_VoC = (df_forecast.type == 'R_L') & (pd.to_datetime(df_forecast.date, format='%Y-%m-%d') >= VoC_start_date)
            index_map = df_forecast.index[row_bool_to_apply_VoC]
            # Index 9 and onwards are the 2000 Reff samples.
            df_slice_after_VoC = df_forecast.iloc[index_map, 8:] 
            multiplier = beta.rvs(6,14, size = df_slice_after_VoC.shape) + 1

        if VoC_flag == 'Delta': # Increase from Delta
            # Here we apply the  beta(2,2)+3 scaling from VoC to the Reff based on CDC results.
            # We do so by editing a slice of the data frame. Forgive me for my sins.
            row_bool_to_apply_VoC = (df_forecast.type == 'R_L') & (pd.to_datetime(df_forecast.date, format='%Y-%m-%d') >= VoC_start_date)
            index_map = df_forecast.index[row_bool_to_apply_VoC]
            # Index 9 and onwards are the 2000 Reff samples.
            df_slice_after_VoC = df_forecast.iloc[index_map, 8:] 
<<<<<<< HEAD
            multiplier = beta.rvs(7, 7, size = df_slice_after_VoC.shape) + 2.5 - 0.5 # Mean 2.1 Delta
=======
            multiplier = beta.rvs(3,3, size = df_slice_after_VoC.shape) + 2.1 - 0.5 # Mean 2.1 Delta
>>>>>>> 4db7f8d81cc17a5273460a46facea2d8ca6b2251

        df_forecast.iloc[index_map , 8:] = df_slice_after_VoC*multiplier
        
    if use_vaccine_effect:
        # Load in vaccination effect data
        vaccination_by_state = pd.read_csv('data/vaccination_by_state.csv', parse_dates=['date'])
        vaccination_by_state = vaccination_by_state[['state', 'date','overall_transmission_effect']]

        # Make datetime objs into strings
        vaccination_by_state['date_str'] = pd.to_datetime(vaccination_by_state['date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        df_forecast['date_str'] = pd.to_datetime(df_forecast['date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')

        # Filling in future days will the same vaccination level as current.
        for state, forecast_df_state in df_forecast.groupby('state'):
            latest_Reff_data_date = max(forecast_df_state.date_str)
            latest_vaccination_data_date = max(vaccination_by_state.groupby('state').get_group(state)['date'])
            latest_vaccination_date_effect = vaccination_by_state.groupby(['state', 'date']).get_group((state, latest_vaccination_data_date))['overall_transmission_effect'].iloc[0]
            # Fill in the future dates with the same level of vaccination.
            vaccination_by_state = vaccination_by_state.append(pd.DataFrame([(state, pd.to_datetime(date), latest_vaccination_date_effect, date.strftime('%Y-%m-%d')) for date in pd.date_range(latest_vaccination_data_date, latest_Reff_data_date)], columns = ['state', 'date', 'overall_transmission_effect', 'date_str']))
        
        # Create a (state,date) indexed map of transmission effect
        overall_transmission_effect = vaccination_by_state.set_index(['state', 'date_str'])['overall_transmission_effect'].to_dict()
            
        # Apply this effect to the forecast
        vaccination_multiplier = df_forecast.apply(lambda row: 1 if row['type']!='R_L' else overall_transmission_effect.get((row['state'], row['date_str']),1), axis=1)
        df_forecast = df_forecast.drop('date_str', axis='columns')
        # Apply the vaccine effect to the forecast. The 8:onwards columns are all the Reff paths.
        df_forecast.iloc[: , 8:] = df_forecast.iloc[: , 8:].multiply(vaccination_multiplier.to_numpy(), axis='rows')
    return df_forecast