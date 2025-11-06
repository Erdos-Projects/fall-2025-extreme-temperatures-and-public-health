# R Model Workflow 

This document outlines the workflow of the R script, which performs the main statistical analysis. The script's goal is to move from the pre-processed `London1_data.csv` file to the final epidemiological results, including the MMT, risk thresholds, and all output plots.

---

## Workflow Steps

1.  **Setup and Data Loading**
    * First, the script loads all necessary R packages (libraries), primarily `dlnm` for the core modeling, `splines` for creating confounder variables, and `ggplot2` for plotting.
    * It then loads the clean, pre-processed `London1_data.csv` file (created by the Python script).
    * Crucially, it converts the text columns from the CSV back into their correct R data types (e.g., `date_col` becomes a `Date`, `dow` becomes a `factor`), which is essential for the model to work.

2.  **Define the Core Predictor (The Cross-basis)**
    * This is the most important step in the DLNM. Instead of using fixed settings, the script uses an **adaptive** approach.
    * It first analyzes the `temp_0` column to find its 10th, 75th, and 90th percentiles. These "adaptive knots" make the model flexible in the right temperature zones for this specific region.
    * It feeds these knots into the `crossbasis` function to create the main `cb_temp` predictor. This single variable tells the model to look for two things at once:
        1.  A **Non-linear "U-shape"** for temperature (using the adaptive knots).
        2.  A **Distributed "Lag" effect** for 21 days (to capture delayed health impacts).
    

3.  **Fit the Statistical Model**
    * The script defines the full model formula: `deaths` are explained by `cb_temp` (our temperature predictor) **plus** the three confounder variables (`time` for long-term trends, `doy` for seasonality, and `dow` for day-of-week patterns).
    * It "fits" this formula using a **Poisson Generalized Linear Model (GLM)**, the standard statistical model for count data like daily deaths.

4.  **Generate the Risk Curve**
    * Once the model is built, the script uses `crosspred` to generate the relative risk (RR) for a range of temperatures (-5°C to 35°C).
    * This "prediction" creates the data object that holds the final **U-shaped curve**, its 95% confidence intervals, and all associated data.

5.  **Calculate Key Results (MMT & Thresholds)**
    * This is the main results-extraction step. The script analyzes the predicted U-curve data to find its *true lowest point*. This temperature is the **Minimum Mortality Temperature (MMT)**, or the "safest" temperature.
    * It then finds the **Cold and Heat Thresholds** by identifying the exact temperatures where the 95% confidence interval (the "error bar") first rises above a relative risk of 1.0. This is the point where the risk becomes "statistically significant."

6.  **Assess Model Quality (Fit Metrics)**
    * The script calculates the **R², MSE, and MAE** for the model. This is *not* for prediction but to check the **in-sample fit**—how well the model (including confounders) explains the historical data. A high R² (like 81.6% for London) indicates a very strong explanatory fit.

7.  **Visualize and Save All Plots**
    * The script saves three separate plots as PNG files for a complete analysis:
        1.  **The 2D RR Curve (`..._2D.png`):** This is the **main result**. It's a manually drawn plot showing the final U-curve, the grey confidence "error bars," and the MMT/Threshold lines.
        
        2.  **The 3D Risk Surface (`..._3D.png`):** This 3D plot shows the full model (Temperature vs. Lag vs. Risk) and is used to visualize *when* the risk occurs (e.g., heat risk peaking at 2-3 days).
        
        3.  **The Raw Scatter Plot (`..._scatter.png`):** This is a diagnostic `ggplot` scatter plot of the *raw* data (deaths vs. temp) to show the *unadjusted* relationship for a visual sanity check.

8.  **Quantify Excess Deaths (Attributable Deaths)**

    * The final section of the script attempts to calculate the *total number* of "excess deaths" over the 30-year period that are directly attributable to days being hotter or colder than the MMT found in Step 5.
