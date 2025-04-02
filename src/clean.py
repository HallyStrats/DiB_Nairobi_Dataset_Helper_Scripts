import pandas as pd
import numpy as np
from utils import DataLoader, ensure_dir, calculate_energy_metrics, calculate_economics, calculate_trip_metrics
import os

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a DataFrame column
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    column : str
        Column name to check for outliers
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    Series of boolean values (True for outliers)
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return abs(z_scores) > threshold
    
    else:
        raise ValueError("Method must be one of ['iqr', 'zscore']")

def clean_daily_data(df, group, output_dir='output/cleaned_data'):
    """
    Clean daily data for a specific group
    
    Parameters:
    -----------
    df : DataFrame
        Daily data for a specific group
    group : str
        One of 'baseline_fuel', 'control_fuel', 'treatment_electric'
    output_dir : str
        Directory to save cleaned data
        
    Returns:
    --------
    DataFrame with cleaned data
    """
    # Make a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Convert date to datetime
    if 'date' in cleaned.columns:
        cleaned['date'] = pd.to_datetime(cleaned['date'])
    
    # Clean distance values
    if 'distance_km' in cleaned.columns:
        # Remove negative distances
        cleaned = cleaned[cleaned['distance_km'] >= 0]
        
        # Handle extreme values
        outliers = detect_outliers(cleaned, 'distance_km', method='iqr')
        cleaned['distance_km_outlier'] = outliers
        
        # Cap extremely high values rather than removing them
        upper_limit = cleaned['distance_km'].quantile(0.99)
        cleaned.loc[cleaned['distance_km'] > upper_limit, 'distance_km'] = upper_limit
    
    # Clean duration values
    if 'duration_min' in cleaned.columns:
        # Remove negative durations
        cleaned = cleaned[cleaned['duration_min'] >= 0]
        
        # Flag extreme values
        outliers = detect_outliers(cleaned, 'duration_min', method='iqr')
        cleaned['duration_min_outlier'] = outliers
        
        # Cap extremely high values
        upper_limit = cleaned['duration_min'].quantile(0.99)
        cleaned.loc[cleaned['duration_min'] > upper_limit, 'duration_min'] = upper_limit
    
    # Clean idle time values
    if 'idle_time_min' in cleaned.columns:
        # Remove negative idle times
        cleaned = cleaned[cleaned['idle_time_min'] >= 0]
        
        # Ensure idle time doesn't exceed duration
        if 'duration_min' in cleaned.columns:
            cleaned.loc[cleaned['idle_time_min'] > cleaned['duration_min'], 'idle_time_min'] = cleaned['duration_min']
    
    # Clean fuel-specific columns
    if 'fuel' in group:
        if 'fuel_amount_l' in cleaned.columns:
            # Remove negative fuel amounts
            cleaned = cleaned[cleaned['fuel_amount_l'].isna() | (cleaned['fuel_amount_l'] >= 0)]
            
            # Flag outliers
            mask = cleaned['fuel_amount_l'].notna()
            if mask.sum() > 10:  # Need enough data points
                outliers = detect_outliers(cleaned[mask], 'fuel_amount_l', method='iqr')
                cleaned.loc[mask, 'fuel_amount_l_outlier'] = outliers.values
            
            # Check for unrealistic fuel consumption
            if 'distance_km' in cleaned.columns:
                mask = (cleaned['fuel_amount_l'] > 0) & (cleaned['distance_km'] > 0)
                if mask.sum() > 0:
                    cleaned.loc[mask, 'km_per_l'] = cleaned.loc[mask, 'distance_km'] / cleaned.loc[mask, 'fuel_amount_l']
                    
                    # Flag unrealistic fuel efficiency
                    low_limit = 5   # Very low km/L for a motorcycle
                    high_limit = 50  # Very high km/L for a motorcycle
                    cleaned.loc[mask, 'fuel_efficiency_outlier'] = (
                        (cleaned.loc[mask, 'km_per_l'] < low_limit) | 
                        (cleaned.loc[mask, 'km_per_l'] > high_limit)
                    )
    
    # Clean electric-specific columns
    if 'electric' in group:
        if 'kwh_charging' in cleaned.columns:
            # Remove negative kWh values
            cleaned = cleaned[cleaned['kwh_charging'].isna() | (cleaned['kwh_charging'] >= 0)]
            
            # Flag outliers if we have enough data
            mask = cleaned['kwh_charging'].notna()
            if mask.sum() > 10:
                outliers = detect_outliers(cleaned[mask], 'kwh_charging', method='iqr')
                cleaned.loc[mask, 'kwh_charging_outlier'] = outliers.values
    
    # Calculate additional metrics
    cleaned = calculate_energy_metrics(cleaned, group)
    cleaned = calculate_economics(cleaned)
    
    # Save cleaned data
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, f"{group}-daily-cleaned.csv")
    cleaned.to_csv(output_file, index=False)
    
    return cleaned

def clean_trip_data(df, group, output_dir='output/cleaned_data'):
    """
    Clean trip data for a specific group
    
    Parameters:
    -----------
    df : DataFrame
        Trip data for a specific group
    group : str
        One of 'baseline_fuel', 'control_fuel', 'treatment_electric'
    output_dir : str
        Directory to save cleaned data
        
    Returns:
    --------
    DataFrame with cleaned data
    """
    # Make a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Convert date columns to datetime
    for col in ['start_date', 'end_date']:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col])
    
    # Clean distance values
    if 'distance_km' in cleaned.columns:
        # Remove negative distances
        cleaned = cleaned[cleaned['distance_km'] >= 0]
        
        # Flag outliers
        outliers = detect_outliers(cleaned, 'distance_km', method='iqr')
        cleaned['distance_km_outlier'] = outliers
    
    # Clean duration values
    if 'duration_min' in cleaned.columns:
        # Remove negative durations
        cleaned = cleaned[cleaned['duration_min'] >= 0]
        
        # Flag outliers
        outliers = detect_outliers(cleaned, 'duration_min', method='iqr')
        cleaned['duration_min_outlier'] = outliers
    
    # Clean idle time values
    if 'idle_time_min' in cleaned.columns:
        # Remove negative idle times
        cleaned = cleaned[cleaned['idle_time_min'] >= 0]
        
        # Ensure idle time doesn't exceed duration
        if 'duration_min' in cleaned.columns:
            cleaned.loc[cleaned['idle_time_min'] > cleaned['duration_min'], 'idle_time_min'] = cleaned['duration_min']
    
    # Clean speed values
    for speed_col in ['avg_speed_kmh', 'avg_moving_speed_kmh', 'max_speed_kmh']:
        if speed_col in cleaned.columns:
            # Remove negative speeds
            cleaned = cleaned[cleaned[speed_col].isna() | (cleaned[speed_col] >= 0)]
            
            # Flag extreme values
            mask = cleaned[speed_col].notna()
            if mask.sum() > 10:
                outliers = detect_outliers(cleaned[mask], speed_col, method='iqr', threshold=2.0)
                cleaned.loc[mask, f'{speed_col}_outlier'] = outliers.values
    
    # Check for consistent start/end times
    if all(col in cleaned.columns for col in ['start_date', 'start_time', 'end_date', 'end_time']):
        cleaned['start_datetime'] = pd.to_datetime(
            cleaned['start_date'].astype(str) + ' ' + cleaned['start_time'].astype(str)
        )
        cleaned['end_datetime'] = pd.to_datetime(
            cleaned['end_date'].astype(str) + ' ' + cleaned['end_time'].astype(str)
        )
        
        # Flag trips where end time is before start time
        cleaned['time_inconsistency'] = cleaned['end_datetime'] < cleaned['start_datetime']
    
    # Calculate additional metrics
    cleaned = calculate_trip_metrics(cleaned)
    
    # Save cleaned data
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, f"{group}-trip-cleaned.csv")
    cleaned.to_csv(output_file, index=False)
    
    return cleaned

def main():
    """Main function to clean all data"""
    loader = DataLoader()
    
    # Clean daily data
    print("Cleaning daily data...")
    daily_data = {}
    for group, df in loader.load_daily_data().items():
        print(f"  Processing {group}...")
        daily_data[group] = clean_daily_data(df, group)
    
    # Clean trip data
    print("Cleaning trip data...")
    trip_data = {}
    for group, df in loader.load_trip_data().items():
        print(f"  Processing {group}...")
        trip_data[group] = clean_trip_data(df, group)
    
    print("Done!")

if __name__ == "__main__":
    main()