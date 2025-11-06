# --- 1. Load Libraries ---
library(splines)
library(dlnm) 
library(ggplot2)
library(mgcv) # For geom_smooth

# --- 2. Working Directory and Data ---

# Paste the path:
setwd("C:\\Users\\Sunit\\OneDrive - Virginia Tech\\Documents\\Erdos Data science\\weather and health project\\fall-2025-extreme-temperatures-and-public-health\\Data")

# *** MODIFIED FILENAME ***
print("Loading data from london_data.csv...")
df <- read.csv("London1_data.csv")

# CRITICAL: Convert columns to the correct R types
df$date <- as.Date(df$date)
df$dow <- as.factor(df$dow)
df$doy <- as.numeric(df$doy)
df$time <- as.numeric(df$time)

print("Data loaded and prepared:")
print(head(df))

# --- 3. Define Cross-basis (ADAPTIVE KNOTS) ---
print("Defining dlnm cross-basis with adaptive knots...")

# 1. Calculate the temperature quantiles FOR THIS REGION'S DATA (df)
# We use c(0.10, 0.75, 0.90) for the 10th, 75th, and 90th percentiles
temp_knots <- quantile(df$temp_0, probs = c(0.10, 0.75, 0.90), na.rm = TRUE)

cat("Using temperature knots at:", round(temp_knots, 1), "\n")

# 2. Feed these new knots into the crossbasis function
cb_temp <- crossbasis(
  df$temp_0,
  lag = 21,
  argvar = list(fun = "ns", knots = temp_knots), # Use the new adaptive knots
  arglag = list(fun = "ns", knots = c(3, 7))
)

# --- 4. Define Formula and Fit Model ---
print("Fitting GLM... (This may take a minute)")
formula <- deaths ~ cb_temp + ns(time, df = 15) + ns(doy, df = 12) + as.factor(dow)

model <- glm(
  formula,
  data = df,
  family = poisson,
  na.action = na.exclude # IMPORTANT: Keeps NAs for correct metric calculation
)
print("Model fitting complete.")

# --- 5. Generate DLNM Predictions ---
print("Generating 'crosspred' predictions...")
pred <- crosspred(
  cb_temp,
  model,
  at = -5:35 # Predict from -5°C to 35°C
)

# --- 6. NEW: Calculate In-Sample Fit Metrics ---
print("Calculating in-sample fit metrics (R2, MSE, MAE)...")

# Helper function
calc_metrics <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2, na.rm = TRUE)
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  r2  <- 1 - sum((actual - predicted)^2, na.rm = TRUE) /
             sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  list(MSE = mse, MAE = mae, R2 = r2)
}

# Get predictions on the response scale (death counts)
predicted_counts <- predict(model, type = "response")
actual_counts    <- df$deaths

# Calculate and print metrics
fit_metrics <- calc_metrics(actual_counts, predicted_counts)
cat("\n=== In-Sample Fit Metrics ===\n")
print(fit_metrics)
cat("-----------------------------\n\n")

# --- 7. Find MMT and Temperature Cutoffs ---
print("Extracting MMT and risk thresholds...")

# 1. Create data frame of the RR curve
rr_data <- data.frame(
  temp = pred$predvar,
  fit  = pred$allRRfit,
  low  = pred$allRRlow,
  high = pred$allRRhigh
)

# 2. Find the TRUE Minimum Mortality Temperature (MMT)
# This is the temperature with the lowest point on the curve (min 'fit')
mmt_row <- rr_data[which.min(rr_data$fit), ]
mmt_temp <- mmt_row$temp

cat(paste("\n--- Epidemiological Analysis Results --- \n"))
cat(paste("True Minimum Mortality Temperature (MMT):", round(mmt_temp, 1), "°C\n"))

# 3. Find COLD threshold:
# Find temps cooler than MMT where the low CI is > 1.0 (significant risk)
cold_risk_temps <- rr_data$temp[rr_data$temp < mmt_temp & rr_data$low > 1.0]
cold_cutoff <- ifelse(length(cold_risk_temps) > 0, max(cold_risk_temps), NA)
cat(paste("Cold Threshold (Significant risk begins below):", round(cold_cutoff, 1), "°C\n"))

# 4. Find HEAT threshold:
# Find temps warmer than MMT where the low CI is > 1.0 (significant risk)
heat_risk_temps <- rr_data$temp[rr_data$temp > mmt_temp & rr_data$low > 1.0]
heat_cutoff <- ifelse(length(heat_risk_temps) > 0, min(heat_risk_temps), NA)
cat(paste("Heat Threshold (Significant risk begins above):", round(heat_cutoff, 1), "°C\n"))
cat(paste("---------------------------------------- \n\n"))

# --- 8. Save DLNM RR Curve to File (MANUAL 2D PLOT) ---
# *** MODIFIED FILENAME AND TITLE ***
print("Saving 2D DLNM Relative Risk Curve to 'dlnm_rr_curve_2D_London.png'...")
png("dlnm_rr_curve_2D_London.png", width = 800, height = 600, res = 100) # Open PNG device

# We will build the plot from scratch using the data in 'pred'
x <- pred$predvar
y_fit <- pred$allRRfit
y_low <- pred$allRRlow
y_high <- pred$allRRhigh

# 1. Set up an empty plot with the correct axis limits

plot(x, y_fit,
     type = "n", # "n" = do not plot points
     ylim = range(y_low, y_high, na.rm = TRUE), # Set Y-axis to fit the confidence band
     xlab = "Temperature (°C)",
     ylab = "Relative Risk (RR)",
     main = "Temperature-Mortality Curve for London (Full Dataset)"
)

# 2. Draw the shaded confidence interval "error bar"
polygon(c(x, rev(x)), c(y_high, rev(y_low)), col = "grey90", border = NA)

# 3. Draw the main RR curve (the U-shape) on top
lines(x, y_fit, lwd = 2, col = "black")

# 4. Add a horizontal line at RR=1.0
abline(h = 1.0, lty = 2)

# 5. Add vertical lines for MMT and Cutoffs
abline(v = mmt_temp, lty = 2, col = "blue")
abline(v = c(cold_cutoff, heat_cutoff), lty = 2, col = "red")

# 6. Add the legend
legend(
  "top",
  legend = c("MMT", "Risk Thresholds", "RR", "95% CI"),
  col = c("blue", "red", "black", "grey90"),
  lty = c(2, 2, 1, 1),
  lwd = c(1, 1, 2, 10), # Makes the legend swatch for CI a thick box
  bty = "n"
)

dev.off() # Close the PNG device (important!)
print("DLNM 2D Relative Risk Curve for London saved.")

# --- 8.5 Save 3D DLNM RR-Lag Curve to File ---
# *** MODIFIED FILENAME AND TITLE ***
print("Saving 3D DLNM Lag-Response Curve to 'dlnm_rr_curve_3D_London.png'...")
png("dlnm_rr_curve_3D_London.png", width = 800, height = 600, res = 100)

# This is the default plot() command for a crosspred object

plot(pred,
     main = "3D Lag-Response Surface for London",
     xlab = "Temperature (°C)",
     ylab = "Lag (Days)",
     zlab = "Relative Risk (RR)"
)

dev.off() # Close the PNG device
print("DLNM 3D Lag-Response Curve saved.")


# --- 9. Save Raw Scatter Plot to File ---
# *** MODIFIED FILENAME AND TITLE ***
print("Saving Raw Scatter Plot to 'raw_deaths_vs_temp_scatter_London.png'...")
png("raw_deaths_vs_temp_scatter_London.png", width = 800, height = 600, res = 100) # Open PNG device
raw_scatter_plot <- ggplot(df, aes(x = temp_0, y = deaths)) +
  geom_point(color = "darkblue", alpha = 0.4) +
  geom_smooth(method = "gam", formula = y ~ s(x, k=4), color = "red", linewidth = 1) +
  labs(
    title = "Raw Daily Deaths vs. Mean Temperature (London)",
    subtitle = "Unadjusted relationship (shows the underlying U-shape)",
    x = "Daily Mean Temperature (°C)",
    y = "Daily Deaths"
  ) +
  theme_minimal()
print(raw_scatter_plot) # Print to the device
dev.off() # Close the PNG device (important!)
print("Raw Scatter Plot for London saved.")

# --- 10. ATTEMPTED TO CALCULATE ATTRIBUTABLE DEATHS (EXCESS DEATHS), FAILS ---
print("Calculating attributable deaths for heat and cold...")

# 1. Calculate deaths attributable to HEAT (dir = "high")
# We use dlnm:::attrdl because loading order (splines/dlnm) can cause conflicts
attr_heat <- dlnm:::attrdl(
  df$temp_0,
  cb_temp,
  coef = pred$coef,
  vcov = pred$vcov,
  type = "an",  # "an" = Attributable Number (i.e., deaths)
  dir  = "high", # "high" = temps above the MMT
  cen  = mmt_temp,
  tot  = TRUE
)

# 2. Calculate deaths attributable to COLD (dir = "low")
attr_cold <- dlnm:::attrdl(
  df$temp_0,
  cb_temp,
  coef = pred$coef,
  vcov = pred$vcov,
  type = "an",
  dir  = "low",  # "low" = temps below the MMT
  cen  = mmt_temp,
  tot  = TRUE
)

# 3. Print the results
total_deaths <- sum(df$deaths, na.rm = TRUE)
# *** MODIFIED LABEL ***
cat(paste("\n--- Attributable Deaths Analysis (Full Period, London) --- \n"))
cat(paste("Based on True MMT of:", round(mmt_temp, 1), "°C\n"))
cat(paste("Total deaths in period:", total_deaths, "\n\n"))

cat("Deaths Attributable to HEAT (Temps >", round(mmt_temp, 1), "°C):\n")
print(round(attr_heat, 0))

cat("\nDeaths Attributable to COLD (Temps <", round(mmt_temp, 1), "°C):\n")
print(round(attr_cold, 0))
cat(paste("-------------------------------------------------- \n\n"))

# --- Done ---