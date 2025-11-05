import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex = True)  #TeX text on plots
import scipy
from scipy.optimize import *
plt.rcParams.update({'font.size': 12})
from matplotlib import colormaps
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

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

age_groups = [
    "<1", "01-04", "05-09", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85-89",
    "90-94", "95+"
]

deaths_data_loc = "../../../HealthData/Weekly_deaths_by_age_and_region_1981_2022/"
deaths_data_file = "weeklydeathoccurrences19812022final.csv"
full_path = deaths_data_loc + deaths_data_file

df = pd.read_csv(full_path)
print(df.head())

# convert date columns and sort by week-start to mirror original script
if "Week_start_date_(Saturday)" in df.columns:
    df["Week_start_date_(Saturday)"] = pd.to_datetime(df["Week_start_date_(Saturday)"], errors="coerce")
else:
    # if only Week_of_occurrence is present, convert it to a week start (Saturday) for consistency
    # Week_of_occurrence is ISO week like YYYY-WW; append '-6' get to Saturday
    if "Week_of_occurrence" in df.columns:
        df["Week_start_date_(Saturday)"] = pd.to_datetime(df["Week_of_occurrence"] + "-6", format="%Y-%W-%w", errors="coerce")
    else:
        raise ValueError("Expected a week identifier column in the deaths file.")

if "Week_end_date_(Friday)" in df.columns:
    df["Week_end_date_(Friday)"] = pd.to_datetime(df["Week_end_date_(Friday)"], errors="coerce")

df = df.sort_values("Week_start_date_(Saturday)")

# group by week, so male and female rows are combined (consistent with original)
df = df.groupby("Week_start_date_(Saturday)", as_index=False).sum(numeric_only=True)
df = df.sort_values("Week_start_date_(Saturday)")

# Keep only rows up to December 31, 2019 (consistent with original 1981-2020 window)
df = df[df["Week_start_date_(Saturday)"] <= "2019-12-31"]

#find weekly mean of daily max temperature for each region
for i in range(len(list_of_regions)):
    weather_data_loc = f"../../../Weather_data/{list_of_regions[i]}/"
    weather_data_file = f"{list_of_cities[i]}_1981-01-01_to_2019-12-31.csv"
    weather_full_path = weather_data_loc + weather_data_file

    weather_df = pd.read_csv(weather_full_path)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])    
    # create week start date (Saturday) for grouping
    weather_df["Week_start_date_(Saturday)"] = weather_df["datetime"] - pd.to_timedelta(
        (weather_df["datetime"].dt.dayofweek - 5) % 7, unit="D"
    )

    weekly_temp = (
        weather_df.groupby("Week_start_date_(Saturday)", as_index=False)["tempmax"]
        .mean()
        .rename(columns={"tempmax": f"{list_of_region_codes[i]}_mean_tempmax"})
    )

    df = df.merge(weekly_temp, on="Week_start_date_(Saturday)", how="left")

#build deaths columns per region and per age group
#For each region code and age group, sum across all sex sub-columns whose names
#start with the region code and end exactly with the age group label.
for code in list_of_region_codes:
    for group in age_groups:
        cols = [c for c in df.columns if c.startswith(code) and c.endswith(group)]
        out_name = f"{code}_deaths_" + group.replace('<','under').replace('+','plus').replace(' ','')
        if len(cols) == 0:
            # create an empty column to keep downstream logic simple
            df[out_name] = 0.0
        else:
            df[out_name] = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

# subtract LOESS-smoothed trend (local regression) for each region+age series ---
for code in list_of_region_codes:
    for group in age_groups:
        col = f"{code}_deaths_" + group.replace('<','under').replace('+','plus').replace(' ','')
        y = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
        x = np.arange(len(df))
        if len(y) >= 3:
            smoothed = lowess(y, x, frac=0.005, return_sorted=False)
            df[col] = y - smoothed
        else:
            # too short to smooth sensibly
            df[col] = y

#make region folders and per-age-group plots
for i in range(len(list_of_regions)):
    region_slug = list_of_regions[i]
    region_name = list_of_region_names_for_plots[i]
    code = list_of_region_codes[i]

    #make folder for this region
    os.makedirs(region_slug, exist_ok=True)

    x_data = f"{code}_mean_tempmax"

    for group in age_groups:
        y_data = f"{code}_deaths_" + group.replace('<','under').replace('+','plus').replace(' ','')

        plt.scatter(df[x_data], df[y_data], s = 2)
        plt.xlabel(r'\rm Mean daily maximum temperature / $^{\\circ}$C')
        plt.ylabel(r'\rm Number of excess deaths')
        group_label = group.replace('<', r'$<$').replace('+', r'$+$')
        title_str = rf"\rm Weekly max temperature vs. deaths (excess), 1981-2020: {region_name} \ (Age {group_label})"
        plt.title(title_str)
        save_str_group = group.replace('<','under').replace('+','plus').replace(' ','')
        save_str = os.path.join(region_slug, f"Weekly_max_temp_vs_deaths_1981-2020_{region_slug}_age_{save_str_group}.png")
        plt.savefig(save_str, dpi = 300)
        #plt.show()
        plt.close()