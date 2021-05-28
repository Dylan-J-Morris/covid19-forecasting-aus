import pandas as pd
from sys import argv
start_s = "2020-12-01"
start_dt = pd.to_datetime(start_s,format="%Y-%m-%d")

if len(argv)>1:
    num_forecast_days = int(argv[1])
else:
    num_forecast_days = 35

end_date = pd.to_datetime("today",format="%Y-%m-%d") + pd.Timedelta(days =num_forecast_days)

num_days = (end_date - start_dt).days
print("Number of days to simulate: %i days" % num_days)
