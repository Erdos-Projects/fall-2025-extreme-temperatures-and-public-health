import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex = True)  #TeX text on plots
import scipy
from scipy.optimize import *
plt.rcParams.update({'font.size': 12})
from matplotlib import colormaps

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
print(df.head())

df["Week_start_date_(Saturday)"] = pd.to_datetime(df["Week_start_date_(Saturday)"], errors="coerce")
df["Week_end_date_(Friday)"] = pd.to_datetime(df["Week_end_date_(Friday)"], errors="coerce")
df = df.sort_values("Week_start_date_(Saturday)")
	
#group by week, so male and female rows are combined
df = df.groupby("Week_start_date_(Saturday)", as_index=False).sum(numeric_only=True)
df = df.sort_values("Week_start_date_(Saturday)") 
#Keep only rows up to December 31, 2019
df = df[df["Week_start_date_(Saturday)"] <= "2019-12-31"]

#make column of total weekly deaths for each region
for i in range(len(list_of_region_codes)):
	column_name = list_of_region_codes[i] + "_total_deaths"
	region_cols = [c for c in df.columns if c.startswith(list_of_region_codes[i])]
	df[column_name] = df[region_cols].sum(axis=1) 

#for each week, make column of the mean of the 7 daily max temperatures
for i in range(len(list_of_regions)):
    #open weather data
    weather_data_loc = f"../../Weather_data/{list_of_regions[i]}/"
    weather_data_file = f"{list_of_cities[i]}_1981-01-01_to_2019-12-31.csv"
    weather_df = pd.read_csv(weather_data_loc + weather_data_file)

    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    #create week start date (Saturday) for grouping
    weather_df["Week_start_date_(Saturday)"] = weather_df["datetime"] - pd.to_timedelta(
        (weather_df["datetime"].dt.dayofweek - 5) % 7, unit="D"
    )

    #group by week and compute average of max temperature
    weekly_temp = (
        weather_df.groupby("Week_start_date_(Saturday)", as_index=False)["tempmax"]
        .mean()
        .rename(columns={"tempmax": f"{list_of_region_codes[i]}_mean_tempmax"})
    )    

    #merge weekly temperature into deaths dataframe
    df = df.merge(weekly_temp, on="Week_start_date_(Saturday)", how="left")

#make individual plots for each region   
for i in range(len(list_of_regions)):
	x_data = list_of_region_codes[i] + "_mean_tempmax"
	y_data = list_of_region_codes[i] + "_total_deaths"
	plt.scatter(df[x_data], df[y_data], s = 2)
	plt.xlabel(r'\rm Mean daily maximum temperature / $^{\circ}$C')
	plt.ylabel(r'\rm Number of deaths')
	title_str = r"\rm Weekly max temperature vs. deaths, 1981-2020: " + list_of_region_names_for_plots[i]
	plt.title(title_str)
	save_str = "Weekly_max_temp_vs_deaths_1981-2020_" + list_of_regions[i] + ".png"
	plt.savefig(save_str, dpi = 300)
	#plt.show()
	plt.close()

#makea single normalised plot for all regions	
for i in range(len(list_of_regions)):
	x_data = list_of_region_codes[i] + "_mean_tempmax"
	y_data = list_of_region_codes[i] + "_total_deaths"
	plt.scatter(df[x_data], df[y_data]/np.mean(df[y_data]), s = 2, color = 'tab:blue')
plt.xlabel(r'\rm Mean daily maximum temperature / $^{\circ}$C')
plt.ylabel(r'\rm Number of deaths (normalised)')
title_str = r"\rm Avg. max. weekly temperature vs. deaths, 1981-2020: all regions"
plt.title(title_str)
save_str = "Weekly_max_temp_vs_deaths_1981-2020_all_regions.png"
plt.savefig(save_str, dpi = 300)
plt.show()

