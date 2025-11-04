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

* Mortality: ONS (Office for National Statistics) weekly deaths by age and region (https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/adhocs/1125weeklydeathoccurrencesbysexfiveyearagegroupandregionenglandandwales1981to2021).
* Weather: Historical daily weather (e.g., temperature, humidity, precipitation) for the largest city/cities in each region (www.visualcrossing.com).
* 2020 and onwards was excluded from our study due to the difficulty of modelling the effects of the Covid-19 pandemic.

![Temperature versus time](/Data/PreliminaryDataExploration/temperature_vs_time/London_london_daily_maximum_temperature.png)
![Deaths versus time](/Data/PreliminaryDataExploration/deaths_vs_time_by_age/weekly_deaths_by_age_group.png)

## Repository Structure

* /HealthData
  * Weekly_deaths_by_age_and_region_1981_2022 (primary dataset)
* /Weather_Data
  * Daily weather for largest city/cities in each region
* /PreliminaryDataExploration
  * An initial look at the data to generate some feel for the data
* labelled_regions_map.png (Scotland labeled but excluded from stats)
  * A map of the 10 different regions
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

* We show the average weekly temperature versus excess deaths for all weeks and years, with a breakdown by region and age. However, we find that breaking the data down into this granularity leaves only a weak signal with a high level of noise. We therefore proceed with our modelling with a regional breakdown only.

### Machine learning modelling

We model the (detrended) weekly excess deaths for each region as a function of weekly aggregated weather features. The end-to-end pipeline is scripted and reproducible:

* **Data prep.** We merge ONS weekly deaths with weekly means of daily weather (e.g., tempmax, humidity, windspeed). Deaths are detrended with a LOESS smoother to remove long-run and seasonal structure before modelling, and 1981–2019 data are retained. Output: `region_weather_vs_excess_deaths.csv`. 
* **Per-region splits.** For each region we create train/test CSVs (80/20, seeded) and quick diagnostics plots of `tempmax` vs. excess deaths. 
* **Model.** A compact feed-forward MLP (64→32→1 with ReLU + dropout) is trained **separately per region** on six features: `tempmax, tempmin, windspeed, precip, snowdepth, humidity`. Inputs and target are standardized using **train-split** statistics; `snowdepth` NAs are set to 0. Training uses Adam (lr=2.5e-4), mini-batches, input Gaussian noise (σ=0.05), dropout (p=0.10), and early stopping (patience=50, min Δ=1e-3). We save `model.pt` and `preprocess.npz` (feature order and scaling). 
* **Test-time inference & plots.** For each region we reload the saved scaler + model to predict on the held-out set and export `test_temp_pred_actual.csv` plus scatter/histogram figures comparing **predicted vs. actual** excess deaths against `tempmax`. 
* **Counterfactual demo.** Using monthly median weather for June/December (1981–2019) we sweep ±5 °C shifts in `tempmax` to visualize the model’s implied change in monthly deaths (centered at Δ=0). 
* **One-click run.** `0_run_modelling.py` executes all steps (1 to 8) in order. 

### Validation

We evaluate prediction quality on TRAIN (held-out) weeks and probe for systematic errors:

* **Train/test protocol.** Within each region, rows are randomly split 80/20. All scaling is fit on the train portion only and applied to validation/test. 
* **Primary metrics (per region + TOTAL).** We report R^2, RMSE, MAE, NMSE, and a pseudo-R^2. We also benchmark against a zero baseline (predict 0 excess deaths), reporting RMSE of that baseline and % improvement vs. zero. A summary CSV with regional rows, plus a concatenated total row, is saved as metrics_summary.csv. 
* For each region we save:

  * **Prediction vs. truth** scatter/histogram (`temp_vs_actual_and_pred_TEST.png`) and predictions-only (`temp_vs_pred_ONLY_TEST.png`). 
  * **Residual structure checks.** We plot `(prediction − actual)` against `tempmax` and overlay 1,000 bootstrapped linear fits to estimate the mean slope +/-sd. A slope near 0 indicates no linear residual bias with respect to temperature. 
  * **Nonlinearity check.** We repeat with 1,000 bootstrapped quadratic fits and display the mean +/-sd of the linear and quadratic coefficients to detect curvature in residuals. 

*Interpretation guide:*
We find an R^2 near 0 and consistent (but minimal) improvement vs. the zero baseline. This implies that short-term weekly excess-death variability is dominated by noise and/or unmodelled drivers. Non-zero slopes/curvature in the residual vs temperature plots would signal misspecification (e.g., omitted interactions or thresholds). However, the flat residual trends we find suggest the ML model has captured what little signal exists in the data.

## Conclusions and Future Implications

* Periods of extreme heat and cold have a tangible effect of public health. They cause an increase in excess deaths.
* However, this effect is smaller than the random fluctuations (noise) is the weekly deaths data.
* Our models are therefore unlikely to be useful in predicting the need for surge planning, as was initially envisioned.
* Our models are likely more suited to predicting longer term effects e.g. how a long period of warm weather or climate change will affect public health in the long term.

## References

* Relevant epidemiology/climate-health literature with a different approach, but similar findings: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(14)62114-0/fulltext

## Team Members

This project has been developed for the Fall 2025 Erdös Institute Data Science Boot Camp by:

* Tom Rose
* Sunit Patil
* Pratiti Deb
* Derek Zapata
