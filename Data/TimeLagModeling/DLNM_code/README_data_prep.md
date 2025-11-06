# Python Preprocessing Workflow

This document outlines the step-by-step process used by the Python script to prepare the raw data for our R-based DLNM mortality model.

The main goal is to **merge** two separate, potentially misaligned datasets (weather and mortality) into a single, clean, and perfectly aligned CSV file that the R model can read.

---

## Workflow Steps

1.  **Load Raw Data**
    * The script begins by loading two different CSV files into memory:
        1.  The complete **weather data file** (which includes temperature).
        2.  The complete **mortality data file** (which includes deaths for London).

2.  **Isolate Key Columns**
    * From the two large files, the script isolates *only* the two columns we care about: the `temp` column and the `London` (deaths) column.

3.  **Align Data Lengths**
    * It checks if the two columns have the same number of rows.
    * If they don't (e.g., one file has more days of data than the other), it **truncates (cuts) both columns** to the length of the *shorter* one. This ensures we only analyze days for which we have *both* temperature and death data.

4.  **Create a Clean Date Index**
    * The script discards any existing dates from the raw files (which might be formatted incorrectly or have gaps).
    * It creates a new, perfect, gap-free `date_range` starting from **January 1, 1985**, and assigns this clean index to both the temperature and deaths columns.

5.  **Combine and Clean the Data**
    * The two aligned columns (temp and deaths) are merged into a single new table.
    * This new table is then cleaned:
        * Any day with a **missing death count** is **dropped** entirely.
        * Any day with a **missing temperature** is **interpolated** (i.e., the value is estimated based on the temperatures of the days immediately before and after it).

6.  **Engineer Confounder Variables**
    * The script creates the three time-based "confounder" columns that the R model needs to control for:
        1.  `time`: A simple day count (0, 1, 2, 3...) to track long-term trends.
        2.  `doy` (Day of Year): A number from 1-365 to track seasonality.
        3.  `dow` (Day of Week): A number from 0-6 to track weekly patterns (like weekend reporting delays).

7.  **Export Final "Model-Ready" File**
    * Finally, the script saves this new, clean, and complete table as a single CSV file (`London1_data.csv`).

    * This file is now perfectly formatted and ready to be loaded directly into R for the DLNM analysis.
