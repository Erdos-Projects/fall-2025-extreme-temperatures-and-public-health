import numpy as np
import pandas as pd
import scipy
from scipy.optimize import *
from statsmodels.nonparametric.smoothers_lowess import lowess

list_of_regions = [
    "NorthEast",
    "NorthWest",
    "YorkshireandtheHumber",
    "EastMidlands",
    "WestMidlands",
    "East",
    "London",
    "SouthEast",
    "SouthWest",
    "Wales"
]

list_of_region_names_for_plots = [
    "North East",
    "North West",
    "Yorkshire and the Humber",
    "East Midlands",
    "West Midlands",
    "East",
    "London",
    "South East",
    "South West",
    "Wales"
]

list_of_region_codes = [
    "E12000001",
    "E12000002",
    "E12000003",
    "E12000004",
    "E12000005",
    "E12000006",
    "E12000007",
    "E12000008",
    "E12000009",
    "W92000004"
]

list_of_cities = [
    "newcastle-upon-tyne",
    "manchester",
    "leeds",
    "nottingham",
    "birmingham",
    "norwich",
    "london",
    "brighton-and-hove",
    "bristol",
    "cardiff"
]

#open deaths data
deaths_data_loc = "../../HealthData/Weekly_deaths_by_age_and_region_1981_2022/"
deaths_data_file = "weeklydeathoccurrences19812022final.csv"
df = pd.read_csv(deaths_data_loc + deaths_data_file)

df["Week_start_date_(Saturday)"] = pd.to_datetime(df["Week_start_date_(Saturday)"], errors="coerce", format="mixed")
df["Week_end_date_(Friday)"] = pd.to_datetime(df["Week_end_date_(Friday)"], errors="coerce", format="mixed")
df = df.sort_values("Week_start_date_(Saturday)")

#group by week, so male and female rows are combined
df = df.groupby("Week_start_date_(Saturday)", as_index=False).sum(numeric_only=True)
df = df.sort_values("Week_start_date_(Saturday)")
#keep only rows up to December 31, 2019
df = df[df["Week_start_date_(Saturday)"] <= "2019-12-31"]

#make column of total weekly deaths for each region
for i in range(len(list_of_region_codes)):
    column_name = list_of_region_codes[i] + "_total_deaths"
    region_cols = [c for c in df.columns if c.startswith(list_of_region_codes[i])]
    df[column_name] = df[region_cols].sum(axis=1)

#for each week, make columns of the weekly means for the many weather variables
weather_vars = ["tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike",
    "humidity","precip","precipprob","precipcover","snowdepth",
    "windspeed","winddir","sealevelpressure","cloudcover","visibility"]

#open weather data for each of the regions
for i in range(len(list_of_regions)):
    weather_data_loc = f"../../Weather_data/{list_of_regions[i]}/"
    weather_data_file = f"{list_of_cities[i]}_1981-01-01_to_2019-12-31.csv"
    weather_df = pd.read_csv(weather_data_loc + weather_data_file)

    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    #create week start date (Saturday) for grouping
    weather_df["Week_start_date_(Saturday)"] = weather_df["datetime"] - pd.to_timedelta(
        (weather_df["datetime"].dt.dayofweek - 5) % 7, unit="D")

    #coerce requested columns to numeric so weekly means work even if strings appear
    for c in weather_vars:
        if c in weather_df.columns:
            weather_df[c] = pd.to_numeric(weather_df[c], errors="coerce")

    #group by week and compute weekly means
    weekly = (
        weather_df.groupby("Week_start_date_(Saturday)", as_index=False)[weather_vars]
        .mean()
        .rename(columns={v: f"{list_of_region_codes[i]}_mean_{v}" for v in weather_vars}))

    #merge weekly weather into deaths dataframe
    df = df.merge(weekly, on="Week_start_date_(Saturday)", how="left")

'''#subtract LOESS-smoothed trend (local regression) from deaths to better show the extreme
#weather effect and remove the seasonal effects i.e. the increase in deaths from seasonal flu etc.
for code in list_of_region_codes:
    col = code + "_total_deaths"
    y = df[col].values
    x = np.arange(len(df))

    #frac=0.004 seems to work best
    smoothed = lowess(y, x, frac=0.004, return_sorted=False)

    #subtract smoothed baseline
    df[col] = y - smoothed'''

#save to CSV
rows = []
for i in range(len(list_of_regions)):
    code = list_of_region_codes[i]
    # map merged column names -> requested output names
    colmap = {f"{code}_mean_{v}": v for v in weather_vars}

    n = len(df)
    #make 'region' a column that repeats the code for each row
    tmp = pd.DataFrame({"region": [code] * n})

    #add weather columns (use NaN if a column is missing)
    for k, v in colmap.items():
        tmp[v] = pd.to_numeric(df[k], errors="coerce")

    #deaths are the detrended totals computed above
    y_col = f"{code}_total_deaths"
    tmp["deaths"] = pd.to_numeric(df[y_col], errors="coerce")

    #snowdepth: fill empties with 0
    if "snowdepth" in tmp.columns:
        tmp["snowdepth"] = tmp["snowdepth"].fillna(0.0)

    rows.append(tmp)

out = pd.concat(rows, ignore_index=True)
out.to_csv("region_weather_vs_excess_deaths.csv", index=False)