import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the visual style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

class DataLoader:
    """Helper class to load and manipulate the motorcycle transit dataset"""
    
    def __init__(self, base_dir='Nairobi Motorcycle Transit Comparison Dataset Fuel vs. Electric Vehicle Performance Tracking (2023)'):
        """Initialize with the base directory of the dataset"""
        self.base_dir = Path(base_dir)
        self.daily_dir = self.base_dir / 'daily_data'
        self.trip_dir = self.base_dir / 'trip_data'
        
        # File paths
        self.daily_files = {
            'baseline_fuel': self.daily_dir / 'baseline-fuel-motorcycle-daily-data.csv',
            'control_fuel': self.daily_dir / 'transition-control-fuel-motorcycle-daily-data.csv',
            'treatment_electric': self.daily_dir / 'transition-treatment-electric-motorcycle-daily-data.csv'
        }
        
        self.trip_files = {
            'baseline_fuel': self.trip_dir / 'baseline-fuel-motorcycle-trip-data.csv',
            'control_fuel': self.trip_dir / 'transition-control-fuel-motorcycle-trip-data.csv',
            'treatment_electric': self.trip_dir / 'transition-treatment-electric-motorcycle-trip-data.csv'
        }
    
    def load_daily_data(self, group=None):
        """
        Load daily data for specified group or all groups
        
        Parameters:
        -----------
        group : str or None
            One of 'baseline_fuel', 'control_fuel', 'treatment_electric', or None for all
            
        Returns:
        --------
        DataFrame or dict of DataFrames
        """
        if group is not None:
            if group not in self.daily_files:
                raise ValueError(f"Group must be one of {list(self.daily_files.keys())}")
            df = pd.read_csv(self.daily_files[group])
            df['group'] = group
            return df
        
        # Load all groups
        dfs = {}
        for name, file_path in self.daily_files.items():
            df = pd.read_csv(file_path)
            df['group'] = name
            dfs[name] = df
            
        return dfs
    
    def load_trip_data(self, group=None):
        """
        Load trip data for specified group or all groups
        
        Parameters:
        -----------
        group : str or None
            One of 'baseline_fuel', 'control_fuel', 'treatment_electric', or None for all
            
        Returns:
        --------
        DataFrame or dict of DataFrames
        """
        if group is not None:
            if group not in self.trip_files:
                raise ValueError(f"Group must be one of {list(self.trip_files.keys())}")
            df = pd.read_csv(self.trip_files[group])
            df['group'] = group
            return df
        
        # Load all groups
        dfs = {}
        for name, file_path in self.trip_files.items():
            df = pd.read_csv(file_path)
            df['group'] = name
            dfs[name] = df
            
        return dfs
    
    def merge_daily_data(self):
        """Merge all daily data into a single DataFrame with a group identifier"""
        dfs = []
        for name, file_path in self.daily_files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['group'] = name
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError("No daily data files found")
        
        return pd.concat(dfs, ignore_index=True)
    
    def merge_trip_data(self):
        """Merge all trip data into a single DataFrame with a group identifier"""
        dfs = []
        for name, file_path in self.trip_files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['group'] = name
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError("No trip data files found")
        
        return pd.concat(dfs, ignore_index=True)

def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary"""
    os.makedirs(directory, exist_ok=True)
    
def calculate_energy_metrics(df, group):
    """
    Calculate energy consumption metrics based on group type
    
    Parameters:
    -----------
    df : DataFrame
        Daily data for a specific group
    group : str
        One of 'baseline_fuel', 'control_fuel', 'treatment_electric'
        
    Returns:
    --------
    DataFrame with additional energy metrics
    """
    df = df.copy()
    
    if 'fuel' in group:
        # For fuel motorcycles
        if 'fuel_amount_l' in df.columns and 'distance_km' in df.columns:
            # Calculate fuel efficiency (km/L)
            mask = (df['fuel_amount_l'].notna()) & (df['fuel_amount_l'] > 0)
            df.loc[mask, 'km_per_l'] = df.loc[mask, 'distance_km'] / df.loc[mask, 'fuel_amount_l']
            
            # Convert to energy metrics (assuming gasoline energy density of 34.2 MJ/L)
            # 1 kWh = 3.6 MJ
            df.loc[mask, 'energy_consumption_kwh'] = df.loc[mask, 'fuel_amount_l'] * 34.2 / 3.6
            df.loc[mask, 'energy_efficiency_km_per_kwh'] = df.loc[mask, 'distance_km'] / df.loc[mask, 'energy_consumption_kwh']
            df.loc[mask, 'energy_cost_per_km'] = df.loc[mask, 'fuel_cost_kes'] / df.loc[mask, 'distance_km']
    
    elif 'electric' in group:
        # For electric motorcycles
        if 'kwh_charging' in df.columns and df['kwh_charging'].notna().any():
            # If we have kWh data
            mask = (df['kwh_charging'].notna()) & (df['kwh_charging'] > 0)
            if mask.any():
                df.loc[mask, 'energy_consumption_kwh'] = df.loc[mask, 'kwh_charging']
                df.loc[mask, 'energy_efficiency_km_per_kwh'] = df.loc[mask, 'distance_km'] / df.loc[mask, 'energy_consumption_kwh']
        
        # Use battery swap cost as a proxy for energy if kWh not available
        if 'battery_swap_cost_kes' in df.columns:
            mask = (df['battery_swap_cost_kes'].notna()) & (df['battery_swap_cost_kes'] > 0) & (df['distance_km'] > 0)
            if mask.any():
                df.loc[mask, 'energy_cost_per_km'] = df.loc[mask, 'battery_swap_cost_kes'] / df.loc[mask, 'distance_km']
    
    return df

def calculate_economics(df):
    """Calculate economic metrics for daily data"""
    df = df.copy()
    
    # Calculate revenue per km
    mask = (df['revenue_kes'].notna()) & (df['revenue_kes'] > 0) & (df['distance_km'] > 0)
    if mask.any():
        df.loc[mask, 'revenue_per_km'] = df.loc[mask, 'revenue_kes'] / df.loc[mask, 'distance_km']
    
    # Calculate maintenance cost per km
    mask = (df['maintenance_kes'].notna()) & (df['distance_km'] > 0)
    if mask.any():
        df.loc[mask, 'maintenance_per_km'] = df.loc[mask, 'maintenance_kes'] / df.loc[mask, 'distance_km']
    
    # Calculate total operating costs
    df['total_operating_cost_kes'] = 0
    
    # For fuel vehicles
    if 'fuel_cost_kes' in df.columns:
        df['total_operating_cost_kes'] += df['fuel_cost_kes'].fillna(0)
    
    # For electric vehicles
    if 'battery_swap_cost_kes' in df.columns:
        df['total_operating_cost_kes'] += df['battery_swap_cost_kes'].fillna(0)
    
    # Add maintenance for all vehicles
    if 'maintenance_kes' in df.columns:
        df['total_operating_cost_kes'] += df['maintenance_kes'].fillna(0)
    
    # Calculate profit
    if 'revenue_kes' in df.columns:
        df['profit_kes'] = df['revenue_kes'].fillna(0) - df['total_operating_cost_kes']
        
        # Calculate profit per km
        mask = (df['profit_kes'].notna()) & (df['distance_km'] > 0)
        if mask.any():
            df.loc[mask, 'profit_per_km'] = df.loc[mask, 'profit_kes'] / df.loc[mask, 'distance_km']
    
    return df

def calculate_trip_metrics(df):
    """Calculate additional metrics for trip data"""
    df = df.copy()
    
    # Convert datetime columns
    for col in ['start_date', 'end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%H:%M:%S').dt.time
    
    # Calculate moving time
    df['moving_time_min'] = df['duration_min'] - df['idle_time_min']
    
    # Calculate time of day category
    if 'start_time' in df.columns:
        df['start_time_full'] = pd.to_datetime(df['start_date'].astype(str) + ' ' + df['start_time'].astype(str))
        df['hour'] = df['start_time_full'].dt.hour
        
        # Create time of day categories
        conditions = [
            (df['hour'] >= 5) & (df['hour'] < 12),
            (df['hour'] >= 12) & (df['hour'] < 17),
            (df['hour'] >= 17) & (df['hour'] < 22),
            (df['hour'] >= 22) | (df['hour'] < 5)
        ]
        choices = ['Morning', 'Afternoon', 'Evening', 'Night']
        df['time_of_day'] = np.select(conditions, choices, default='Unknown')
    
    return df