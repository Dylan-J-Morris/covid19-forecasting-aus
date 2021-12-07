# A collection of helper functions that are used throughout. This file is aimed to avoid replication of code.

import pandas as pd

def read_in_NNDSS(date_string, apply_delay_at_read=False, apply_inc_at_read=False, running_epyreff=False):
    """
    A general function to read in the NNDSS data. Alternatively this can be manually set to read in the linelist instead.
    Args:
        date_string: (str) a string of the date of the data file.

    Returns:
        A dataframe of all NNDSS data.
    """

    import numpy as np
    from datetime import timedelta
    import glob
    from params import use_linelist, assume_local_cases_if_unknown
    from params import scale_gen, shape_gen, scale_inc, shape_inc, scale_rd, shape_rd, offset_rd, offset_inc

    if not use_linelist:
        # On occasion the date string in NNDSS will be missing the leading 0  (e.g. 2Aug2021 vs 02Aug2021). In this case manually add the zero.
        case_file_date = pd.to_datetime(date_string).strftime("%d%b%Y")
        path = "data/COVID-19 UoM "+case_file_date+"*.xlsx"

        for file in glob.glob(path):  # Allows us to use the * option
            df = pd.read_excel(file,
                               parse_dates=['SPECIMEN_DATE', 'NOTIFICATION_DATE',
                                            'NOTIFICATION_RECEIVE_DATE', 'TRUE_ONSET_DATE'],
                               dtype={'PLACE_OF_ACQUISITION': str})
        if len(glob.glob(path)) != 1:
            print("There are %i files with the same date" % len(glob.glob(path)))
        if len(glob.glob(path)) == 0:
            raise FileNotFoundError(
                "NNDSS no found. Did you want to use a linelist? Or is the file named wrong?")

        # Fixes errors in updated python versions
        df.TRUE_ONSET_DATE = pd.to_datetime(df.TRUE_ONSET_DATE, errors='coerce')
        df.NOTIFICATION_DATE = pd.to_datetime(df.NOTIFICATION_DATE, errors='coerce')

        # Find most representative date
        df['date_inferred'] = df.TRUE_ONSET_DATE
        df.loc[df.TRUE_ONSET_DATE.isna(), 'date_inferred'] = df.loc[df.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
        df.loc[df.date_inferred.isna(), 'date_inferred'] = df.loc[df.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)

        # The first 4 digits is the country code. We use this to determin if the cases is local or imported. We can choose which assumption we keep. This should be set to true during local outbreak waves.
        if assume_local_cases_if_unknown:
            # Fill blanks with local code
            df.PLACE_OF_ACQUISITION.fillna('11019999', inplace=True)
        else:
            # Fill blanks with unknown international code
            df.PLACE_OF_ACQUISITION.fillna('00038888', inplace=True)

        # IMPORTANT NOTE: State of infection is determined by the STATE column, not the PLACE_OF_ACQUISITION column

        # Set imported cases, local cases have 1101 as first 4 digits.
        df['imported'] = df.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[:4] != '1101' else 0)
        df['local'] = 1 - df.imported

        return df

    else:
        # The linelist, currently produce by Gerry Ryan, has had the onset dates and local / imported status vetted by a human. This can be a lot more reliable during an outbreak.
            
        case_file_date = pd.to_datetime(date_string).strftime("%Y-%m-%d")
        path = "data/interim_linelist_"+case_file_date+"*.csv"
            
        for file in glob.glob(path):  # Allows us to use the * option
            df = pd.read_csv(file)

        if len(glob.glob(path)) == 0:
            raise FileNotFoundError("Calculated linelist not found. Did you want to use NNDSS or the imputed linelist?")
        
        if running_epyreff:
            df['date_confirmation'] = pd.to_datetime(df['date_confirmation'], errors='coerce')
            # set the known confirmation dates 
            df['date_inferred'] = df['date_confirmation']
        
        else:
            # take the representative dates 
            df['date_onset'] = pd.to_datetime(df['date_onset'], errors='coerce')
            # create boolean of when confirmation dates used
            df['date_confirmation'] = pd.to_datetime(df['date_confirmation'], errors='coerce')
            df['is_confirmation'] = df['date_onset'].isna()
            # set the known onset dates 
            df['date_inferred'] = df['date_onset']
        
        if apply_delay_at_read:
            # calculate number of delays to sample 
            n_delays = df['date_inferred'].isna().sum()
            # sample that number of delays from the distribution and take the ceiling. 
            # This was fitted to the third and second wave data, looking at the common differences 
            # between onsets and confirmations
            rd = offset_rd + np.random.gamma(shape=shape_rd, scale=scale_rd, size=n_delays)
            rd = np.ceil(rd) * timedelta(days=1)
            
            # fill missing days with the confirmation date, noting that this is adjusted when used
            df.loc[df['date_inferred'].isna(), 'date_inferred'] = df.loc[df['date_inferred'].isna(), 'date_confirmation'] - rd
        else:
            # just apply the confirmation date and let EpyReff handle the delay distribution 
            df.loc[df['date_inferred'].isna(), 'date_inferred'] = df.loc[df['date_inferred'].isna(), 'date_confirmation'] 
            
        # now we apply the incubation period to the inferred onset date. Note that this should never be done in the 
        # absence of the delay 
        if apply_inc_at_read:
            # assuming that the date_onset field is valid, this is the actual date that individuals get symptoms
            n_infs = df['date_inferred'].shape[0]
            inc = np.random.gamma(shape=shape_inc, scale=scale_inc, size=n_infs)
            # need to take the ceiling of the incubation period as otherwise the merging in generate_posterior 
            # doesnt work properly
            inc = np.ceil(inc) * timedelta(days=1)
            df['date_inferred'] = df['date_inferred'] - inc
            
        df['imported'] = [1 if stat =='imported' else 0 for stat in df['import_status']]
        df['local'] = 1 - df.imported
        df['STATE'] = df['state']
        
        return df


def read_in_Reff_file(file_date, adjust_TP_forecast=False):
    """
    Read in Reff h5 file produced by generate_RL_forecast. 
    """

    if file_date is None:
        raise Exception('Need to provide file date to Reff read.')

    file_date = pd.to_datetime(file_date).strftime("%Y-%m-%d")

    if adjust_TP_forecast:
        df_forecast = pd.read_hdf('results/soc_mob_R_adjusted'+file_date+'.h5', key='Reff')
    else:
        df_forecast = pd.read_hdf('results/soc_mob_R'+file_date+'.h5', key='Reff')

    return df_forecast
