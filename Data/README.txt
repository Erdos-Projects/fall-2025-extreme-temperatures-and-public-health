Health and weather data for UK regions from 1981 - 2022.

Directories and files included:

/HealthData - The health data we will primarily use is in HealthData/Weekly_deaths_by_age_and_region_1981_2022

labelled_regions_map.png  - shows the UK regions we are considering in this study. Note that although Scotland is truncated and labelled on the map, it is not included in our statistics. 

/PopulationStatistics - Populations for different local authorities by year since 1981. The regions we are looking at are made up of groups of these 'local authorities', so we need to sum the population in each local authority to get the region's population each year.

/WeatherData - Weather data for the largest city/cities in each region.

/BasicPolynomialModeling
Second order polynomial fits to the temperature vs excess deaths data

/PyTorchModellingAndPredictions
PyTorch CNN modelling of the weather and deaths data

Region codes:
Region Code, Region Name, Largest City
E12000001, North East, Newcastle-upon-Tyne
E12000002, North West, Manchester
E12000003, Yorkshire and the Humber, Leeds
E12000004, East Midlands, Nottingham
E12000005, West Midlands, Birmingham
E12000006, East, Norwich
E12000007, London, London
E12000008, South East, Brighton and Hove
E12000009, South West, Bristol
W92000004, Wales, Cardiff