import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex = True)  #TeX text on plots
import scipy
from scipy.optimize import *
plt.rcParams.update({'font.size': 12})

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

#at the moment, just one city per region, so one for loop
for i in range(len(list_of_regions)):
	#open data
	weather_data_loc = "../../Weather_data/" + list_of_regions[i] + "/"
	weather_data_file = list_of_cities[i] + "_1981-01-01_to_2019-12-31.csv"
	df = pd.read_csv(weather_data_loc + weather_data_file)
	print(df.head())

	#convert date format
	df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d")
	
	fig = plt.figure(figsize = (9,5))
	plt.scatter(df["datetime"], df["tempmax"], s = 5)
	plt.xlabel(r'\rm Year')
	plt.ylabel(r'\rm Maximum daily temperature / $^{\circ}$C')
	title_str = "Region: " + list_of_regions[i] + ", city: " + list_of_cities[i]
	plt.title(title_str)
	plt.savefig(list_of_regions[i] + "_" + list_of_cities[i] + '_daily_maximum_temperature.png', dpi=300)
	#plt.show()