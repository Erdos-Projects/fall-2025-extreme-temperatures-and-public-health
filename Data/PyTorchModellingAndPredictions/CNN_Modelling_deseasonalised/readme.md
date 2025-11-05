# Modelling Pipeline

This folder contains a series of Python scripts. 
Each script does one step. 
They are run one after the other to go from raw data -> trained models -> evaluation/plots -> demonstrations. 
Earlier steps produce the files later steps expect.

## Quick start

From this folder, run the following and each file will run in sequence:
python 0_run_modelling.py

This executes the numbered scripts in order. They may also be run manually in order.

## What each step does

1. **`1_prep_data.py`** — load/clean/prepare raw data. Writes a processed dataset for later steps.
2. **`2_train_test_split.py`** — split the processed dataset into train/test sets and save the splits.
3. **`3_train_models.py`** — fit one or more models on the training set and save model.
4. **`4_test_and_plot.py`** — run the trained models on the test set, generate metrics and plots, save outputs.
5. **`5_goodness_of_fit_straight_line.py`** — goodness of fit checks for a basic, straight‑line (first‑order) model.
6. **`6_goodness_of_fit_second_order.py`** — goodness‑of‑fit checks for a second‑order model.
7. **`7_statistical_tests.py`** — additional statistical tests/diagnostics, save to csv.
8. **`8_demonstration.py`** — simple usage/demo script showing how to load and use the trained models.

## Outputs:

/region_splits contains

- A trained model for each geographical region e.g. E12000007 for London
- A plot of training and testing data
- A plot of Test data true values and values predicted using the trained model
- A straight line and second order polynomial fit to the residuals, showing that the residuals are consistent with a flat line
- An example prediction 'expected_excess....png'. This takes the median historical weather for June and December and predicts the effect of various temperature changes. These temperature changes are simply added linearly to the historical medians. More usefully, a real weather forecast could be used to predict excess deaths in the near future.

## Region codes:

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