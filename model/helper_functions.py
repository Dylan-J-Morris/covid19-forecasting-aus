# A collection of helper functions that are used throughout. This file is aimed to avoid replication of code.

def read_in_NNDSS(date_string):
    """
    A general function to read in the NNDSS data. Alternatively this can be manually set to read in the linelist instead.
    Args:
        date_string: (str) a string of the date of the data file.

    Returns:
        A dataframe of all NNDSS data.
    """

    import pandas as pd
    from datetime import timedelta
    import glob

    use_linelist = True # If something goes wrong on a day you can set this to True to use the linelist

    if not use_linelist: 
        case_file_date = pd.to_datetime(date_string).strftime("%d%b%Y")
        path = "data/COVID-19 UoM "+case_file_date+"*.xlsx"

        for file in glob.glob(path): # Allows us to use the * option
            df = pd.read_excel(file, 
                            parse_dates=['SPECIMEN_DATE','NOTIFICATION_DATE','NOTIFICATION_RECEIVE_DATE','TRUE_ONSET_DATE'], 
                            dtype= {'PLACE_OF_ACQUISITION':str})
        if len(glob.glob(path))!=1:
            print("There are %i files with the same date" %len(glob.glob(path)))

        # Fixes errors in updated python versions
        df.TRUE_ONSET_DATE = pd.to_datetime(df.TRUE_ONSET_DATE, errors='coerce') 
        df.NOTIFICATION_DATE = pd.to_datetime(df.NOTIFICATION_DATE, errors='coerce')

        # Find most representative date 
        df['date_inferred'] = df.TRUE_ONSET_DATE
        df.loc[df.TRUE_ONSET_DATE.isna(),'date_inferred'] = df.loc[df.TRUE_ONSET_DATE.isna()].NOTIFICATION_DATE - timedelta(days=5)
        df.loc[df.date_inferred.isna(),'date_inferred'] = df.loc[df.date_inferred.isna()].NOTIFICATION_RECEIVE_DATE - timedelta(days=6)
    
        # Set imported cases, local cases have 1101 as first 4 digits. This is the country code. 
        df.PLACE_OF_ACQUISITION.fillna('00038888',inplace=True) # Fill blanks with simply unknown
        df['imported'] = df.PLACE_OF_ACQUISITION.apply(lambda x: 1 if x[:4]!='1101' else 0)
        df['local'] = 1 - df.imported
        return df

    else:
        case_file_date = pd.to_datetime(date_string).strftime("%Y%m%d")
        path = "data/*linelist_"+case_file_date+"*.csv"
        for file in glob.glob(path): # Allows us to use the * option
            df = pd.read_csv(file)

        df['date_inferred'] = df['date_onset']
        df.loc[df['date_onset'].isna(),'date_inferred'] = df.loc[df['date_onset'].isna()]['date_detection'] - timedelta(days=3) # Fill missing days

        df['imported'] = [1 if stat=='imported' else 0 for stat in df['import_status']]
        df['local'] = 1 - df.imported
        df['STATE'] = df['state']
        df['NOTIFICATION_RECEIVE_DATE'] = df['date_detection'] # Only used by EpyReff. Possible improvement here.
        return df
