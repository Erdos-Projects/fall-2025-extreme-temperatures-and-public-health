import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex = True)  #TeX text on plots
import scipy
from scipy.optimize import *
plt.rcParams.update({'font.size': 12})
from matplotlib import colormaps

#open data
deaths_data_loc = "../../HealthData/Weekly_deaths_by_age_and_region_1981_2022/"
deaths_data_file = "weeklydeathoccurrences19812022final.csv"
df = pd.read_csv(deaths_data_loc + deaths_data_file)
print(df.head())

#group by week, so male and female rows are combined
df = df.groupby("Week_of_occurrence", as_index=False).sum(numeric_only=True)

#convert date format
df["Week_start"] = pd.to_datetime(df["Week_of_occurrence"] + "-1", format="%Y-%W-%w")

#sum deaths in each age group
age_groups = ["<1", "01-04", "05-09", "10-14", "15-19", "20-24", "25-29",
              "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
              "60-64", "65-69", "70-74", "75-79", "80-84", "85-89",
              "90-94", "95+"]
for group in age_groups:
    # match columns for different regions that end exactly with 'group', e.g. *_10-14
    cols = [c for c in df.columns if c.endswith(group)]

    out_name = f"deaths_{group.replace('<', 'under').replace('+', 'plus')}"
    # coerce to numeric in case anything is read as object
    df[out_name] = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)


fig = plt.figure(figsize = (10,5))
   
#plot selected age groups i.e. ignore those with small numbers
age_groups_to_plot = ["45-49", "50-54", "55-59", "60-64", "65-69", 
						"70-74", "75-79", "80-84", "85-89", "90-94"]
              
# sample evenly spaced colors from a continuous map
cmap = colormaps["tab10"]  # or "viridis", "plasma", "tab20b"
N = len(age_groups_to_plot)
colors = [cmap(i / max(N-1, 1)) for i in range(N)]
 
for i, group in enumerate(age_groups_to_plot):
    col_name = group.replace("<","under").replace("+","plus")
    legend_label = group.replace("<", r"$<$").replace("+", r"$+$")  # LaTeX-safe
    plt.scatter(df["Week_start"],df[f"deaths_{col_name}"],s=2,color=colors[i],label=legend_label)

#plt.xlim(pd.to_datetime('2003-01-01'), pd.to_datetime('2007-12-31'))
#plt.ylim(-100,3000)
plt.xlabel(r"\rm Year")
plt.ylabel(r"\rm Weekly registered deaths")
plt.title(r"\rm Weekly Deaths by Age Group")
plt.legend(title=r"\rm Age group", bbox_to_anchor=(1.05, 1), ncol = 1, loc="upper left")
plt.tight_layout()
plt.savefig("weekly_deaths_by_age_group.png", dpi=300)
plt.show()