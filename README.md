# fall-2025-extreme-temperatures-and-public-health

We investigate the effects of short periods of extreme weather on public health across UK regions (1981–2022).

## Background

Short bursts of heat and cold can affect a person's health, particularly the elderly and young. In turn, this can stress health systems and increase mortality. These effects vary by season, region, and population. This project links weekly deaths to weekly weather anomalies in order to quantify the effect of extreme weather and predict the implications of a potential changes to the climate.

## Problem Statement

We measure how weeks of unusually hot or cold weather shift mortality relative to expected baselines, by region and age group, while controlling for long-term trends and seasonality.

## Objectives

* Build population-adjusted weekly excess mortality rates for each region.
* Estimate short-term effects of heat and cold on mortality.
* Assess the impact of extreme weather based on e.g. region and age group.
* Produce simple indicators that can pair with forecasts for surge planning.

## KPIs

* Accuracy of excess mortality estimates on unseen historical weeks.
* Consistency of effect direction/magnitude across regions and age groups.
* Clarity and usefulness of stakeholder facing summaries.
* Reproducibility of data processing and modelling.

## Stakeholders

Public health agencies, emergency planners, health services, local authorities and UK residents.

## Project Setup

## Data

* Mortality: ONS (Office for National Statistics) weekly deaths by age and region (TBD link).
* Population: ONS local authority population estimates aggregated to regions (TBD link).
* Weather: Historical daily weather (e.g., temperature, humidity, precipitation) for the largest city/cities in each region (TBD provider and link).
* 2020 and onwards was excluded from out study due to the difficulty of modelling the effects of the Covid-19 pandemic.

## Repository Structure

* /HealthData
  * Weekly_deaths_by_age_and_region_1981_2022 (primary dataset)
* /Weather_Data
  * Daily weather for largest city/cities in each region
* /PreliminaryDataExploration
  * An initial look at the data to generate some feel for the data
* labelled_regions_map.png (Scotland labeled but excluded from stats)
* /BasicModelling
  * Modelling the data with a simply polynomial, assessing only the impact of temperature
* /PyTorchModellingAndPredictions
  * A CNN model, trained on the weekly data for weather and excess deaths

## Approach

Baseline Processing:

* We identify a seasonal trend, with a yearly peak at winder and dip in summer. This corresponds to the seasonal flu.
* We remove the seasonal effects from the data with LOESS-smoothed trend fitting. This then highlights the impact of shorter term weather effects.
* We define 'extreme temperatures' as being daily maximums outside the range of 8-21 degrees celsius. This is when an increase in deaths above the seasonal baseline is seen.

![Temperature versus excess deaths](/Data/PreliminaryDataExploration/temperature_vs_deaths/sub_moving_avg/Weekly_max_temp_vs_deaths_1981-2020_WestMidlands.png)

Simple modelling and projections:

* We show the average weekly temperature versus excess deaths for all weeks and years, with a breakdown by region and age. However, we find that the breaking the data down into this granularity leaves only a weak signal with a high level of noise. We therefore proceed with our modelling with a regional breakdown only.
*

Machine learning modelling:

*
*Validation:


## Results

*
*

## Conclusions and Future Implications

TBD: implications for heat-health and cold-weather plans, early warning triggers, and resource allocation; limits and next steps.

## References

* ONS mortality and population datasets (links to be added).
* Historical weather data provider (links to be added).
* Relevant epidemiology/climate-health literature (links to be added).

## Team Members

This project has been developed for the Fall 2025 Erdös Institute Data Science Boot Camp by:

* Tom Rose
* Sunit Patil
* Pratiti Deb
* Derek Zapata
