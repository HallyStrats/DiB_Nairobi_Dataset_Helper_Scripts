import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from utils import DataLoader, ensure_dir, calculate_energy_metrics, calculate_economics, calculate_trip_metrics

def set_plot_style():
    """Set the style for all plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

def plot_daily_metrics(df, output_dir='output/figures/daily'):
    """
    Create plots of daily metrics
    
    Parameters:
    -----------
    df : DataFrame
        Combined daily data with 'group' column
    output_dir : str
        Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    # Ensure date is datetime
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Plot 1: Daily distance by group
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='group', y='distance_km', data=df, showfliers=False)
    sns.stripplot(x='group', y='distance_km', data=df, color='black', alpha=0.5, size=4)
    plt.title('Daily Distance by Group')
    plt.ylabel('Distance (km)')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_distance_by_group.png'), dpi=300)
    plt.close()
    
    # Plot 2: Distance over time by group
    plt.figure(figsize=(14, 8))
    for name, group_df in df.groupby('group'):
        plt.plot(group_df['date'], group_df['distance_km'], 'o-', label=name, alpha=0.7)
    plt.title('Daily Distance Over Time')
    plt.ylabel('Distance (km)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_distance_over_time.png'), dpi=300)
    plt.close()
    
    # Plot 3: Duration vs. Distance scatter
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='distance_km', y='duration_min', hue='group', data=df, alpha=0.7, s=100)
    plt.title('Daily Duration vs. Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Duration (min)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_vs_distance.png'), dpi=300)
    plt.close()
    
    # Plot 4: Idle time percentage
    df['idle_percentage'] = df['idle_time_min'] / df['duration_min'] * 100
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='group', y='idle_percentage', data=df, showfliers=False)
    sns.stripplot(x='group', y='idle_percentage', data=df, color='black', alpha=0.5, size=4)
    plt.title('Idle Time Percentage by Group')
    plt.ylabel('Idle Time (%)')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'idle_time_percentage.png'), dpi=300)
    plt.close()
    
    # Plot 5: Economic comparison (if available)
    econ_cols = ['revenue_kes', 'maintenance_kes', 'fuel_cost_kes', 'battery_swap_cost_kes', 
                 'revenue_per_km', 'profit_per_km', 'energy_cost_per_km']
    
    available_cols = [col for col in econ_cols if col in df.columns and df[col].notna().sum() > 0]
    
    for col in available_cols:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='group', y=col, data=df, showfliers=False)
        sns.stripplot(x='group', y=col, data=df, color='black', alpha=0.5, size=4)
        plt.title(f'{col.replace("_", " ").title()} by Group')
        plt.ylabel(col.replace('_', ' ').title())
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_by_group.png'), dpi=300)
        plt.close()
    
    # Plot 6: Energy efficiency comparison (if available)
    if 'km_per_l' in df.columns or 'energy_efficiency_km_per_kwh' in df.columns:
        plt.figure(figsize=(14, 8))
        
        if 'km_per_l' in df.columns and df['km_per_l'].notna().sum() > 0:
            fuel_data = df[df['km_per_l'].notna()]
            sns.boxplot(x='group', y='km_per_l', data=fuel_data, showfliers=False)
            sns.stripplot(x='group', y='km_per_l', data=fuel_data, color='black', alpha=0.5, size=4)
            plt.title('Fuel Efficiency (km/L) by Group')
            plt.ylabel('km/L')
            plt.xlabel('')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fuel_efficiency.png'), dpi=300)
            plt.close()
        
        if 'energy_efficiency_km_per_kwh' in df.columns and df['energy_efficiency_km_per_kwh'].notna().sum() > 0:
            plt.figure(figsize=(14, 8))
            energy_data = df[df['energy_efficiency_km_per_kwh'].notna()]
            sns.boxplot(x='group', y='energy_efficiency_km_per_kwh', data=energy_data, showfliers=False)
            sns.stripplot(x='group', y='energy_efficiency_km_per_kwh', data=energy_data, color='black', alpha=0.5, size=4)
            plt.title('Energy Efficiency (km/kWh) by Group')
            plt.ylabel('km/kWh')
            plt.xlabel('')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'energy_efficiency.png'), dpi=300)
            plt.close()

def plot_trip_metrics(df, output_dir='output/figures/trip'):
    """
    Create plots of trip metrics
    
    Parameters:
    -----------
    df : DataFrame
        Combined trip data with 'group' column
    output_dir : str
        Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    # Plot 1: Trip distance distribution
    plt.figure(figsize=(14, 8))
    sns.histplot(data=df, x='distance_km', hue='group', bins=20, kde=True, element='step')
    plt.title('Trip Distance Distribution')
    plt.xlabel('Distance (km)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trip_distance_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 2: Trip duration distribution
    plt.figure(figsize=(14, 8))
    sns.histplot(data=df, x='duration_min', hue='group', bins=20, kde=True, element='step')
    plt.title('Trip Duration Distribution')
    plt.xlabel('Duration (min)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trip_duration_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 3: Average speed by group
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='group', y='avg_speed_kmh', data=df, showfliers=False)
    sns.stripplot(x='group', y='avg_speed_kmh', data=df, color='black', alpha=0.5, size=4)
    plt.title('Average Speed by Group')
    plt.ylabel('Average Speed (km/h)')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_speed_by_group.png'), dpi=300)
    plt.close()
    
    # Plot 4: Moving vs Idle time
    if 'moving_time_min' in df.columns:
        plot_data = df.copy()
        plot_data['moving_time_pct'] = plot_data['moving_time_min'] / plot_data['duration_min'] * 100
        plot_data['idle_time_pct'] = plot_data['idle_time_min'] / plot_data['duration_min'] * 100
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='group', y='moving_time_pct', data=plot_data, showfliers=False)
        plt.title('Moving Time Percentage by Group')
        plt.ylabel('Moving Time (%)')
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'moving_time_percentage.png'), dpi=300)
        plt.close()
    
    # Plot 5: Time of day distribution (if available)
    if 'time_of_day' in df.columns:
        plt.figure(figsize=(14, 8))
        time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
        time_counts = df.groupby(['group', 'time_of_day']).size().unstack(fill_value=0)
        
        # Make sure all time periods are present
        for time in time_order:
            if time not in time_counts.columns:
                time_counts[time] = 0
        
        time_counts = time_counts[time_order]
        time_counts.plot(kind='bar', stacked=False)
        plt.title('Trip Count by Time of Day')
        plt.xlabel('')
        plt.ylabel('Number of Trips')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trips_by_time_of_day.png'), dpi=300)
        plt.close()
    
    # Plot 6: Max speed by group
    if 'max_speed_kmh' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='group', y='max_speed_kmh', data=df, showfliers=False)
        sns.stripplot(x='group', y='max_speed_kmh', data=df, color='black', alpha=0.5, size=4)
        plt.title('Maximum Speed by Group')
        plt.ylabel('Maximum Speed (km/h)')
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'max_speed_by_group.png'), dpi=300)
        plt.close()
    
    # Plot 7: Trip distance vs duration scatter
    plt.figure(figsize=(14, 8))
    for name, group_df in df.groupby('group'):
        plt.scatter(group_df['distance_km'], group_df['duration_min'], label=name, alpha=0.5)
    
    plt.title('Trip Distance vs. Duration')
    plt.xlabel('Distance (km)')
    plt.ylabel('Duration (min)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trip_distance_vs_duration.png'), dpi=300)
    plt.close()
    
    # Plot 8: Trip heatmap by location (if coordinates available)
    if all(col in df.columns for col in ['start_lat', 'start_lon', 'end_lat', 'end_lon']):
        # Check if we have enough valid coordinates
        valid_coords = df.dropna(subset=['start_lat', 'start_lon', 'end_lat', 'end_lon'])
        
        if len(valid_coords) > 10:
            # Create separate plots for each group
            for name, group_df in valid_coords.groupby('group'):
                plt.figure(figsize=(12, 12))
                
                # Plot start points
                plt.scatter(group_df['start_lon'], group_df['start_lat'], 
                         c='green', alpha=0.5, label='Start Points')
                
                # Plot end points
                plt.scatter(group_df['end_lon'], group_df['end_lat'], 
                         c='red', alpha=0.5, label='End Points')
                
                plt.title(f'Trip Start and End Locations - {name}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'trip_locations_{name}.png'), dpi=300)
                plt.close()

def plot_comparison_metrics(daily_data, trip_data, output_dir='output/figures/comparison'):
    """
    Create comparative plots between fuel and electric vehicles
    
    Parameters:
    -----------
    daily_data : DataFrame
        Combined daily data with 'group' column
    trip_data : DataFrame
        Combined trip data with 'group' column
    output_dir : str
        Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    # Extract treatment and control groups for comparison
    control_daily = daily_data[daily_data['group'] == 'control_fuel'].copy()
    treatment_daily = daily_data[daily_data['group'] == 'treatment_electric'].copy()
    
    control_trip = trip_data[trip_data['group'] == 'control_fuel'].copy()
    treatment_trip = trip_data[trip_data['group'] == 'treatment_electric'].copy()
    
    # Plot 1: Daily distance comparison
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    control_daily_stats = control_daily.groupby('date')['distance_km'].mean().reset_index()
    treatment_daily_stats = treatment_daily.groupby('date')['distance_km'].mean().reset_index()
    
    plt.plot(control_daily_stats['date'], control_daily_stats['distance_km'], 
             'o-', label='Control (Fuel)', alpha=0.7, color='blue')
    plt.plot(treatment_daily_stats['date'], treatment_daily_stats['distance_km'], 
             'o-', label='Treatment (Electric)', alpha=0.7, color='green')
    
    plt.title('Average Daily Distance: Fuel vs. Electric')
    plt.ylabel('Average Distance (km)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_distance_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: Operating costs comparison (if available)
    cost_cols = {
        'control_fuel': 'fuel_cost_kes',
        'treatment_electric': 'battery_swap_cost_kes'
    }
    
    # Check if we have cost data
    has_control_cost = 'fuel_cost_kes' in control_daily.columns and control_daily['fuel_cost_kes'].notna().sum() > 0
    has_treatment_cost = 'battery_swap_cost_kes' in treatment_daily.columns and treatment_daily['battery_swap_cost_kes'].notna().sum() > 0
    
    if has_control_cost and has_treatment_cost:
        plt.figure(figsize=(14, 8))
        
        # Calculate cost per km
        control_daily['cost_per_km'] = control_daily['fuel_cost_kes'] / control_daily['distance_km']
        treatment_daily['cost_per_km'] = treatment_daily['battery_swap_cost_kes'] / treatment_daily['distance_km']
        
        # Plot boxplots
        data = [
            control_daily['cost_per_km'].dropna(),
            treatment_daily['cost_per_km'].dropna()
        ]
        
        plt.boxplot(data, labels=['Fuel', 'Electric'], showfliers=False)
        plt.title('Energy Cost per Kilometer: Fuel vs. Electric')
        plt.ylabel('Cost per km (KES)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cost_per_km_comparison.png'), dpi=300)
        plt.close()
    
    # Plot 3: Profit comparison (if available)
    if 'profit_per_km' in control_daily.columns and 'profit_per_km' in treatment_daily.columns:
        has_control_profit = control_daily['profit_per_km'].notna().sum() > 0
        has_treatment_profit = treatment_daily['profit_per_km'].notna().sum() > 0
        
        if has_control_profit and has_treatment_profit:
            plt.figure(figsize=(14, 8))
            
            data = [
                control_daily['profit_per_km'].dropna(),
                treatment_daily['profit_per_km'].dropna()
            ]
            
            plt.boxplot(data, labels=['Fuel', 'Electric'], showfliers=False)
            plt.title('Profit per Kilometer: Fuel vs. Electric')
            plt.ylabel('Profit per km (KES)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'profit_per_km_comparison.png'), dpi=300)
            plt.close()
    
    # Plot 4: Speed comparison from trip data
    plt.figure(figsize=(14, 8))
    
    data = [
        control_trip['avg_speed_kmh'].dropna(),
        treatment_trip['avg_speed_kmh'].dropna()
    ]
    
    plt.boxplot(data, labels=['Fuel', 'Electric'], showfliers=False)
    plt.title('Average Speed: Fuel vs. Electric')
    plt.ylabel('Average Speed (km/h)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_speed_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 5: Distance distribution comparison
    plt.figure(figsize=(14, 8))
    
    bins = np.linspace(0, max(control_trip['distance_km'].max(), treatment_trip['distance_km'].max()), 20)
    
    plt.hist(control_trip['distance_km'], bins=bins, alpha=0.5, label='Fuel', density=True)
    plt.hist(treatment_trip['distance_km'], bins=bins, alpha=0.5, label='Electric', density=True)
    
    plt.title('Trip Distance Distribution: Fuel vs. Electric')
    plt.xlabel('Distance (km)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trip_distance_distribution_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main function to create all plots"""
    loader = DataLoader()
    
    print("Loading data...")
    # Load daily data
    daily_dfs = loader.load_daily_data()
    merged_daily = loader.merge_daily_data()
    
    # Load trip data
    trip_dfs = loader.load_trip_data()
    merged_trip = loader.merge_trip_data()
    
    # Process daily data with additional metrics
    for group, df in daily_dfs.items():
        daily_dfs[group] = calculate_energy_metrics(df, group)
        daily_dfs[group] = calculate_economics(daily_dfs[group])
    
    # Process trip data with additional metrics
    for group, df in trip_dfs.items():
        trip_dfs[group] = calculate_trip_metrics(df)
    
    # Update merged datasets
    merged_daily = pd.concat([df for df in daily_dfs.values()], ignore_index=True)
    merged_trip = pd.concat([df for df in trip_dfs.values()], ignore_index=True)
    
    print("Creating daily metrics plots...")
    plot_daily_metrics(merged_daily)
    
    print("Creating trip metrics plots...")
    plot_trip_metrics(merged_trip)
    
    print("Creating comparison plots...")
    plot_comparison_metrics(merged_daily, merged_trip)
    
    print("Done!")

if __name__ == "__main__":
    main()