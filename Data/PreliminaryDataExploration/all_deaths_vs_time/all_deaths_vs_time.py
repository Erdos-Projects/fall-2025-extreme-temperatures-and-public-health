import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex = True)  #TeX text on plots
import scipy
from scipy.optimize import *
plt.rcParams.update({'font.size': 12})

#open data
deaths_data_loc = "../../HealthData/Weekly_deaths_by_age_and_region_1981_2022/"
deaths_data_file = "weeklydeathoccurrences19812022final.csv"
df = pd.read_csv(deaths_data_loc + deaths_data_file)
print(df.head())

#group by week, so male and female rows are combined
df = df.groupby("Week_of_occurrence", as_index=False).sum(numeric_only=True)
#convert date format
df["Week_start"] = pd.to_datetime(df["Week_of_occurrence"] + "-1", format="%Y-%W-%w")
#sum all deaths
df["Weekly_deaths_all_ages"] = df.loc[:, "E12000001_<1":"W92000004_95+"].sum(axis=1)

fig = plt.figure(figsize = (9,5))
plt.scatter(df["Week_start"], df["Weekly_deaths_all_ages"])
plt.xlabel(r'\rm Year')
plt.ylabel(r'\rm Weekly registered deaths')
plt.savefig('weekly_deaths_all_ages.png')
plt.show()