# Nairobi Motorcycle Transit Comparison

This repository contains Python scripts for processing, analyzing, and visualizing data from a study comparing the performance of fuel and electric motorcycles in Nairobi's transit system. The dataset is publicly available on Mendeley Data (DOI: 10.17632/nv3rkn24zv.1) and should be downloaded from there. Once downloaded, the dataset will be in the correct format and just needs to be placed in the appropriate directory for the scripts to function properly.

## Purpose

The scripts in this repository are designed to:

- **Clean**: Process raw CSV data by handling missing values, removing outliers, and calculating additional metrics.
- **Visualize**: Create plots to explore daily and trip-level metrics, including comparisons between fuel and electric motorcycles.
- **Summarize**: Generate statistical summaries and comparisons across groups in JSON format.

## Dataset Overview

The dataset should be downloaded from Mendeley Data and will include the following files:

- **Daily Data**: Aggregated daily metrics for each motorcycle.
  - `baseline-fuel-motorcycle-daily-data.csv`
  - `transition-control-fuel-motorcycle-daily-data.csv`
  - `transition-treatment-electric-motorcycle-daily-data.csv`

- **Trip Data**: Detailed trip-level data for each motorcycle.
  - `baseline-fuel-motorcycle-trip-data.csv`
  - `transition-control-fuel-motorcycle-trip-data.csv`
  - `transition-treatment-electric-motorcycle-trip-data.csv`

These files will already be structured correctly upon download and should be placed in the following directory:

```
Nairobi Motorcycle Transit Comparison Dataset Fuel vs. Electric Vehicle Performance Tracking (2023)
```

with the subdirectories:
- `daily_data`
- `trip_data`

The scripts assume this structure for data loading. Metrics in these files may include distance traveled (km), duration (minutes), fuel consumption (liters) or energy usage (kWh), revenue (KES), costs (e.g., fuel, battery swaps, maintenance), and more.

## Directory Structure

After running the scripts, your repository will generate outputs in an `output` directory. Here’s the expected structure (assuming the dataset is present):

```
.
├── Nairobi Motorcycle Transit Comparison Dataset Fuel vs. Electric Vehicle Performance Tracking (2023)
│   ├── daily_data
│   │   ├── baseline-fuel-motorcycle-daily-data.csv
│   │   ├── transition-control-fuel-motorcycle-daily-data.csv
│   │   └── transition-treatment-electric-motorcycle-daily-data.csv
│   └── trip_data
│       ├── baseline-fuel-motorcycle-trip-data.csv
│       ├── transition-control-fuel-motorcycle-trip-data.csv
│       └── transition-treatment-electric-motorcycle-trip-data.csv
├── output
│   ├── cleaned_data
│   │   ├── baseline_fuel-daily-cleaned.csv
│   │   ├── baseline_fuel-trip-cleaned.csv
│   │   ├── control_fuel-daily-cleaned.csv
│   │   ├── control_fuel-trip-cleaned.csv
│   │   ├── treatment_electric-daily-cleaned.csv
│   │   └── treatment_electric-trip-cleaned.csv
│   ├── figures
│   │   ├── comparison/
│   │   ├── daily/
│   │   └── trip/
│   └── reports
│       ├── baseline_fuel_daily_summary.json
│       ├── baseline_fuel_trip_summary.json
│       ├── control_fuel_daily_summary.json
│       ├── control_fuel_trip_summary.json
│       ├── group_comparison.json
│       ├── overall_summary.json
│       ├── treatment_electric_daily_summary.json
│       └── treatment_electric_trip_summary.json
├── src
│   ├── clean.py
│   ├── plot.py
│   ├── summarise.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Scripts Description

- **`clean.py`**: Cleans raw daily and trip data by removing negative values, capping outliers, and adding derived metrics (e.g., fuel efficiency, energy consumption). Outputs cleaned CSV files to `output/cleaned_data/`.
- **`plot.py`**: Generates visualizations such as boxplots, histograms, and scatter plots for daily and trip metrics, including group comparisons. Saves PNG files to `output/figures/` in subdirectories `daily`, `trip`, and `comparison`.
- **`summarise.py`**: Produces detailed statistical summaries for each group and comparisons across groups. Saves JSON reports to `output/reports/`.
- **`utils.py`**: Provides helper functions and classes, including `DataLoader` for loading data, and metric calculation functions (`calculate_energy_metrics`, `calculate_economics`, `calculate_trip_metrics`).

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HallyStrats/DiB_Nairobi_Dataset_Helper_Scripts.git
   cd DiB_Nairobi_Dataset_Helper_Scripts
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the Dataset**:
   Download the dataset from Mendeley Data (DOI: 10.17632/nv3rkn24zv.1) and ensure it is placed in the correct directory structure as described above.

## Usage Instructions

Run each script from the root directory of the repository. Ensure the dataset is in place before executing.

- **Clean the Data**:
  ```bash
  python src/clean.py
  ```
  Generates cleaned CSV files in `output/cleaned_data/`.

- **Generate Plots**:
  ```bash
  python src/plot.py
  ```
  Creates PNG files in `output/figures/` subdirectories.

- **Create Summary Reports**:
  ```bash
  python src/summarise.py
  ```
  Produces JSON files in `output/reports/`.

## Output

After running the scripts, expect the following outputs:

- **Cleaned Data**: Processed CSV files in `output/cleaned_data/` with outlier flags and derived metrics.
- **Figures**: PNG plots in `output/figures/` visualizing metrics like distance, speed, costs, and efficiency.
- **Reports**: JSON files in `output/reports/` with summary statistics, group comparisons, and key findings.

## Dependencies

See `requirements.txt` for the list of required Python packages. These include libraries for data manipulation (`pandas`, `numpy`), plotting (`matplotlib`, `seaborn`), and statistical analysis (`scipy`).