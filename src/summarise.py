import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from scipy import stats
from utils import DataLoader, ensure_dir, calculate_energy_metrics, calculate_economics, calculate_trip_metrics

def summarize_daily_data(df, group, output_dir='output/reports'):
    """
    Create summary statistics for daily data
    
    Parameters:
    -----------
    df : DataFrame
        Daily data for a specific group
    group : str
        One of 'baseline_fuel', 'control_fuel', 'treatment_electric'
    output_dir : str
        Directory to save summary
        
    Returns:
    --------
    dict of summary statistics
    """
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Prepare output directory
    ensure_dir(output_dir)
    
    # Basic summary statistics
    summary = {
        'group': group,
        'data_type': 'daily',
        'num_records': len(df),
        'date_range': {},
        'unique_users': len(df['user_id'].unique()) if 'user_id' in df.columns else None
    }
    
    # Populate date_range only with valid dates
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        if pd.notna(min_date):
            summary['date_range']['start'] = min_date.strftime('%Y-%m-%d')
        if pd.notna(max_date):
            summary['date_range']['end'] = max_date.strftime('%Y-%m-%d')
    
    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict = {}
    
    for col in numeric_cols:
        # Skip if column is mostly NaN
        if df[col].isna().sum() > 0.9 * len(df):
            continue
            
        col_stats = {
            'count': int(df[col].count()),
            'mean': float(df[col].mean()) if df[col].count() > 0 else None,
            'std': float(df[col].std()) if df[col].count() > 1 else None,
            'min': float(df[col].min()) if df[col].count() > 0 else None,
            'q1': float(df[col].quantile(0.25)) if df[col].count() > 0 else None,
            'median': float(df[col].median()) if df[col].count() > 0 else None,
            'q3': float(df[col].quantile(0.75)) if df[col].count() > 0 else None,
            'max': float(df[col].max()) if df[col].count() > 0 else None
        }
        
        stats_dict[col] = col_stats
    
    summary['statistics'] = stats_dict
    
    # Add specific metrics for energy and economics
    if 'fuel' in group:
        # Fuel-specific metrics
        fuel_metrics = {}
        
        if 'km_per_l' in df.columns and df['km_per_l'].notna().sum() > 0:
            fuel_metrics['avg_fuel_efficiency_km_per_l'] = float(df['km_per_l'].mean())
            fuel_metrics['median_fuel_efficiency_km_per_l'] = float(df['km_per_l'].median())
        
        if 'fuel_cost_kes' in df.columns and 'distance_km' in df.columns:
            mask = (df['fuel_cost_kes'].notna()) & (df['distance_km'] > 0)
            if mask.sum() > 0:
                fuel_metrics['avg_fuel_cost_per_km'] = float((df.loc[mask, 'fuel_cost_kes'] / df.loc[mask, 'distance_km']).mean())
        
        summary['fuel_metrics'] = fuel_metrics
    
    elif 'electric' in group:
        # Electric-specific metrics
        electric_metrics = {}
        
        if 'energy_efficiency_km_per_kwh' in df.columns and df['energy_efficiency_km_per_kwh'].notna().sum() > 0:
            electric_metrics['avg_energy_efficiency_km_per_kwh'] = float(df['energy_efficiency_km_per_kwh'].mean())
            electric_metrics['median_energy_efficiency_km_per_kwh'] = float(df['energy_efficiency_km_per_kwh'].median())
        
        if 'battery_swap_cost_kes' in df.columns and 'distance_km' in df.columns:
            mask = (df['battery_swap_cost_kes'].notna()) & (df['distance_km'] > 0)
            if mask.sum() > 0:
                electric_metrics['avg_battery_swap_cost_per_km'] = float((df.loc[mask, 'battery_swap_cost_kes'] / df.loc[mask, 'distance_km']).mean())
        
        summary['electric_metrics'] = electric_metrics
    
    # Economic summary
    economic_metrics = {}
    
    # Revenue metrics
    if 'revenue_kes' in df.columns and df['revenue_kes'].notna().sum() > 0:
        economic_metrics['avg_daily_revenue'] = float(df['revenue_kes'].mean())
        
        if 'distance_km' in df.columns:
            mask = (df['revenue_kes'].notna()) & (df['distance_km'] > 0)
            if mask.sum() > 0:
                economic_metrics['avg_revenue_per_km'] = float((df.loc[mask, 'revenue_kes'] / df.loc[mask, 'distance_km']).mean())
    
    # Profit metrics
    if 'profit_kes' in df.columns and df['profit_kes'].notna().sum() > 0:
        economic_metrics['avg_daily_profit'] = float(df['profit_kes'].mean())
        
        if 'distance_km' in df.columns:
            mask = (df['profit_kes'].notna()) & (df['distance_km'] > 0)
            if mask.sum() > 0:
                economic_metrics['avg_profit_per_km'] = float((df.loc[mask, 'profit_kes'] / df.loc[mask, 'distance_km']).mean())
    
    summary['economic_metrics'] = economic_metrics
    
    # Save to file
    output_file = os.path.join(output_dir, f"{group}_daily_summary.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def summarize_trip_data(df, group, output_dir='output/reports'):
    """
    Create summary statistics for trip data
    
    Parameters:
    -----------
    df : DataFrame
        Trip data for a specific group
    group : str
        One of 'baseline_fuel', 'control_fuel', 'treatment_electric'
    output_dir : str
        Directory to save summary
        
    Returns:
    --------
    dict of summary statistics
    """
    # Convert date columns to datetime if they exist
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    
    # Prepare output directory
    ensure_dir(output_dir)
    
    # Basic summary statistics
    summary = {
        'group': group,
        'data_type': 'trip',
        'num_trips': len(df),
        'date_range': {},
        'unique_users': len(df['user_id'].unique()) if 'user_id' in df.columns else None
    }
    
    # Populate date_range only with valid dates
    if 'start_date' in df.columns:
        min_start = df['start_date'].min()
        if pd.notna(min_start):
            summary['date_range']['start'] = min_start.strftime('%Y-%m-%d')
    if 'end_date' in df.columns:
        max_end = df['end_date'].max()
        if pd.notna(max_end):
            summary['date_range']['end'] = max_end.strftime('%Y-%m-%d')
    
    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict = {}
    
    for col in numeric_cols:
        # Skip if column is mostly NaN
        if df[col].isna().sum() > 0.9 * len(df):
            continue
            
        col_stats = {
            'count': int(df[col].count()),
            'mean': float(df[col].mean()) if df[col].count() > 0 else None,
            'std': float(df[col].std()) if df[col].count() > 1 else None,
            'min': float(df[col].min()) if df[col].count() > 0 else None,
            'q1': float(df[col].quantile(0.25)) if df[col].count() > 0 else None,
            'median': float(df[col].median()) if df[col].count() > 0 else None,
            'q3': float(df[col].quantile(0.75)) if df[col].count() > 0 else None,
            'max': float(df[col].max()) if df[col].count() > 0 else None
        }
        
        stats_dict[col] = col_stats
    
    summary['statistics'] = stats_dict
    
    # Trip characteristics
    trip_metrics = {}
    
    # Distance distribution
    if 'distance_km' in df.columns:
        trip_metrics['total_distance_km'] = float(df['distance_km'].sum())
        trip_metrics['avg_trip_distance_km'] = float(df['distance_km'].mean())
        trip_metrics['median_trip_distance_km'] = float(df['distance_km'].median())
        
        # Distance categorization
        trip_metrics['short_trips_pct'] = float((df['distance_km'] < 5).mean() * 100)  # Less than 5km
        trip_metrics['medium_trips_pct'] = float(((df['distance_km'] >= 5) & (df['distance_km'] < 15)).mean() * 100)  # 5-15km
        trip_metrics['long_trips_pct'] = float((df['distance_km'] >= 15).mean() * 100)  # 15km+
    
    # Duration metrics
    if 'duration_min' in df.columns:
        trip_metrics['avg_trip_duration_min'] = float(df['duration_min'].mean())
        trip_metrics['median_trip_duration_min'] = float(df['duration_min'].median())
        
        if 'distance_km' in df.columns:
            # Calculate minutes per km (inverse of speed) to avoid division by zero
            mask = df['distance_km'] > 0
            if mask.sum() > 0:
                min_per_km = df.loc[mask, 'duration_min'] / df.loc[mask, 'distance_km']
                trip_metrics['avg_min_per_km'] = float(min_per_km.mean())
                trip_metrics['median_min_per_km'] = float(min_per_km.median())
    
    # Speed metrics
    if 'avg_speed_kmh' in df.columns:
        trip_metrics['avg_speed_kmh'] = float(df['avg_speed_kmh'].mean())
        trip_metrics['median_speed_kmh'] = float(df['avg_speed_kmh'].median())
    
    if 'max_speed_kmh' in df.columns:
        trip_metrics['avg_max_speed_kmh'] = float(df['max_speed_kmh'].mean())
        trip_metrics['median_max_speed_kmh'] = float(df['max_speed_kmh'].median())
    
    # Idle time analysis
    if 'idle_time_min' in df.columns and 'duration_min' in df.columns:
        mask = df['duration_min'] > 0
        if mask.sum() > 0:
            idle_pct = df.loc[mask, 'idle_time_min'] / df.loc[mask, 'duration_min'] * 100
            trip_metrics['avg_idle_time_pct'] = float(idle_pct.mean())
            trip_metrics['median_idle_time_pct'] = float(idle_pct.median())
    
    # Time of day distribution
    if 'start_time' in df.columns:
        # Convert start_time to datetime if it's not already
        if isinstance(df['start_time'].iloc[0], str):
            try:
                df['start_hour'] = pd.to_datetime(df['start_time']).dt.hour
            except:
                # If it fails, try parsing as time string
                df['start_hour'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.hour
        else:
            # Assume it's already a datetime
            df['start_hour'] = df['start_time'].dt.hour
        
        # Categorize trips by time of day
        morning_mask = (df['start_hour'] >= 6) & (df['start_hour'] < 12)
        afternoon_mask = (df['start_hour'] >= 12) & (df['start_hour'] < 18)
        evening_mask = (df['start_hour'] >= 18) & (df['start_hour'] < 22)
        night_mask = (df['start_hour'] >= 22) | (df['start_hour'] < 6)
        
        trip_metrics['morning_trips_pct'] = float(morning_mask.mean() * 100)
        trip_metrics['afternoon_trips_pct'] = float(afternoon_mask.mean() * 100)
        trip_metrics['evening_trips_pct'] = float(evening_mask.mean() * 100)
        trip_metrics['night_trips_pct'] = float(night_mask.mean() * 100)
    
    summary['trip_metrics'] = trip_metrics
    
    # Save to file
    output_file = os.path.join(output_dir, f"{group}_trip_summary.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def compare_groups(summaries, output_dir='output/reports'):
    """
    Compare statistics between different groups
    
    Parameters:
    -----------
    summaries : list of dict
        List of summary dictionaries from summarize_daily_data or summarize_trip_data
    output_dir : str
        Directory to save comparison results
        
    Returns:
    --------
    dict of comparison results
    """
    ensure_dir(output_dir)
    
    # Group summaries by data type
    daily_summaries = [s for s in summaries if s.get('data_type') == 'daily']
    trip_summaries = [s for s in summaries if s.get('data_type') == 'trip']
    
    comparisons = {
        'daily': compare_daily_summaries(daily_summaries) if daily_summaries else None,
        'trip': compare_trip_summaries(trip_summaries) if trip_summaries else None
    }
    
    # Save to file
    output_file = os.path.join(output_dir, "group_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    return comparisons

def compare_daily_summaries(summaries):
    """Compare daily data summaries between groups"""
    # Extract group names
    groups = [s['group'] for s in summaries]
    comparison = {'groups': groups}
    
    # Extract fuel vs electric groups
    fuel_groups = [s for s in summaries if 'fuel' in s['group']]
    electric_groups = [s for s in summaries if 'electric' in s['group']]
    
    # Compare basic metrics
    basic_metrics = {}
    
    # Distance comparison
    if all('statistics' in s and 'distance_km' in s['statistics'] for s in summaries):
        distance_data = {
            s['group']: {
                'avg': s['statistics']['distance_km']['mean'],
                'median': s['statistics']['distance_km']['median']
            } for s in summaries
        }
        basic_metrics['distance_km'] = distance_data
    
    # Duration comparison
    if all('statistics' in s and 'duration_min' in s['statistics'] for s in summaries):
        duration_data = {
            s['group']: {
                'avg': s['statistics']['duration_min']['mean'],
                'median': s['statistics']['duration_min']['median']
            } for s in summaries
        }
        basic_metrics['duration_min'] = duration_data
    
    # Revenue comparison
    if all('statistics' in s and 'revenue_kes' in s['statistics'] for s in summaries):
        revenue_data = {
            s['group']: {
                'avg': s['statistics']['revenue_kes']['mean'],
                'median': s['statistics']['revenue_kes']['median']
            } for s in summaries
        }
        basic_metrics['revenue_kes'] = revenue_data
    
    comparison['basic_metrics'] = basic_metrics
    
    # Energy efficiency comparison (fuel vs electric)
    energy_comparison = {}
    
    # Collect fuel efficiency data
    fuel_efficiency = {}
    for s in fuel_groups:
        if 'fuel_metrics' in s and 'avg_fuel_efficiency_km_per_l' in s['fuel_metrics']:
            fuel_efficiency[s['group']] = {
                'avg_km_per_l': s['fuel_metrics']['avg_fuel_efficiency_km_per_l'],
                'avg_cost_per_km': s['fuel_metrics'].get('avg_fuel_cost_per_km')
            }
    
    # Collect electric efficiency data
    electric_efficiency = {}
    for s in electric_groups:
        if 'electric_metrics' in s and 'avg_energy_efficiency_km_per_kwh' in s['electric_metrics']:
            electric_efficiency[s['group']] = {
                'avg_km_per_kwh': s['electric_metrics']['avg_energy_efficiency_km_per_kwh'],
                'avg_cost_per_km': s['electric_metrics'].get('avg_battery_swap_cost_per_km')
            }
    
    energy_comparison['fuel'] = fuel_efficiency
    energy_comparison['electric'] = electric_efficiency
    
    # Calculate cost savings (if data available)
    if fuel_efficiency and electric_efficiency:
        # Take the first available group from each category
        fuel_group = list(fuel_efficiency.keys())[0]
        electric_group = list(electric_efficiency.keys())[0]
        
        fuel_cost_per_km = fuel_efficiency[fuel_group].get('avg_cost_per_km')
        electric_cost_per_km = electric_efficiency[electric_group].get('avg_cost_per_km')
        
        if fuel_cost_per_km is not None and electric_cost_per_km is not None:
            savings_per_km = fuel_cost_per_km - electric_cost_per_km
            savings_pct = (savings_per_km / fuel_cost_per_km) * 100 if fuel_cost_per_km > 0 else None
            
            energy_comparison['cost_comparison'] = {
                'savings_per_km': float(savings_per_km),
                'savings_percentage': float(savings_pct) if savings_pct is not None else None
            }
    
    comparison['energy_comparison'] = energy_comparison
    
    # Economic comparison
    economic_comparison = {}
    
    for s in summaries:
        if 'economic_metrics' in s:
            group_economics = {}
            metrics = s['economic_metrics']
            
            if 'avg_daily_revenue' in metrics:
                group_economics['avg_daily_revenue'] = metrics['avg_daily_revenue']
            
            if 'avg_daily_profit' in metrics:
                group_economics['avg_daily_profit'] = metrics['avg_daily_profit']
            
            if 'avg_revenue_per_km' in metrics:
                group_economics['avg_revenue_per_km'] = metrics['avg_revenue_per_km']
            
            if 'avg_profit_per_km' in metrics:
                group_economics['avg_profit_per_km'] = metrics['avg_profit_per_km']
            
            economic_comparison[s['group']] = group_economics
    
    comparison['economic_comparison'] = economic_comparison
    
    return comparison

def compare_trip_summaries(summaries):
    """Compare trip data summaries between groups"""
    # Extract group names
    groups = [s['group'] for s in summaries]
    comparison = {'groups': groups}
    
    # Compare trip metrics
    trip_comparison = {}
    
    # Distance comparison
    if all('trip_metrics' in s and 'avg_trip_distance_km' in s['trip_metrics'] for s in summaries):
        distance_data = {
            s['group']: {
                'avg_trip_distance_km': s['trip_metrics']['avg_trip_distance_km'],
                'median_trip_distance_km': s['trip_metrics']['median_trip_distance_km'],
                'short_trips_pct': s['trip_metrics'].get('short_trips_pct'),
                'medium_trips_pct': s['trip_metrics'].get('medium_trips_pct'),
                'long_trips_pct': s['trip_metrics'].get('long_trips_pct')
            } for s in summaries
        }
        trip_comparison['distance'] = distance_data
    
    # Duration comparison
    if all('trip_metrics' in s and 'avg_trip_duration_min' in s['trip_metrics'] for s in summaries):
        duration_data = {
            s['group']: {
                'avg_trip_duration_min': s['trip_metrics']['avg_trip_duration_min'],
                'median_trip_duration_min': s['trip_metrics']['median_trip_duration_min']
            } for s in summaries
        }
        trip_comparison['duration'] = duration_data
    
    # Speed comparison
    if all('trip_metrics' in s and 'avg_speed_kmh' in s['trip_metrics'] for s in summaries):
        speed_data = {
            s['group']: {
                'avg_speed_kmh': s['trip_metrics']['avg_speed_kmh'],
                'median_speed_kmh': s['trip_metrics']['median_speed_kmh']
            } for s in summaries
        }
        trip_comparison['speed'] = speed_data
    
    # Time of day distribution comparison
    time_metrics = ['morning_trips_pct', 'afternoon_trips_pct', 'evening_trips_pct', 'night_trips_pct']
    if all('trip_metrics' in s and all(m in s['trip_metrics'] for m in time_metrics) for s in summaries):
        time_data = {
            s['group']: {
                metric: s['trip_metrics'][metric] for metric in time_metrics
            } for s in summaries
        }
        trip_comparison['time_of_day'] = time_data
    
    comparison['trip_comparison'] = trip_comparison
    
    return comparison

def run_summary_analysis(data_loader, output_dir='output/reports'):
    """
    Run summary analysis on all data groups
    
    Parameters:
    -----------
    data_loader : DataLoader
        Initialized DataLoader object
    output_dir : str
        Directory to save summary reports
        
    Returns:
    --------
    dict with all summary results
    """
    ensure_dir(output_dir)
    summaries = {'daily': [], 'trip': []}
    
    # Process daily data for each group
    for group in ['baseline_fuel', 'control_fuel', 'treatment_electric']:
        # Get daily data
        daily_df = data_loader.load_daily_data(group)
        
        if daily_df is not None and not daily_df.empty:
            # Summarize daily data
            daily_summary = summarize_daily_data(daily_df, group, output_dir)
            summaries['daily'].append(daily_summary)
    
    # Process trip data for each group
    for group in ['baseline_fuel', 'control_fuel', 'treatment_electric']:
        # Get trip data
        trip_df = data_loader.load_trip_data(group)
        
        if trip_df is not None and not trip_df.empty:
            # Summarize trip data
            trip_summary = summarize_trip_data(trip_df, group, output_dir)
            summaries['trip'].append(trip_summary)
    
    # Compare groups
    daily_comparison = compare_groups(summaries['daily'], output_dir) if summaries['daily'] else None
    trip_comparison = compare_groups(summaries['trip'], output_dir) if summaries['trip'] else None
    
    # Create overall summary report
    overall_summary = {
        'summary_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'groups_analyzed': ['baseline_fuel', 'control_fuel', 'treatment_electric'],
        'daily_data': {s['group']: {'records': s['num_records'], 'users': s['unique_users']} for s in summaries['daily']},
        'trip_data': {s['group']: {'trips': s['num_trips'], 'users': s['unique_users']} for s in summaries['trip']},
        'key_findings': extract_key_findings(summaries)
    }
    
    # Save overall summary
    output_file = os.path.join(output_dir, "overall_summary.json")
    with open(output_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    return {
        'daily_summaries': summaries['daily'],
        'trip_summaries': summaries['trip'],
        'overall_summary': overall_summary
    }

def extract_key_findings(summaries):
    """Extract key findings from summaries"""
    findings = {}
    
    # Get daily summaries
    daily_summaries = summaries['daily']
    
    # Extract fuel vs electric groups
    fuel_groups = [s for s in daily_summaries if 'fuel' in s['group']]
    electric_groups = [s for s in daily_summaries if 'electric' in s['group']]
    
    # Energy efficiency comparison
    if fuel_groups and electric_groups and 'fuel_metrics' in fuel_groups[0] and 'electric_metrics' in electric_groups[0]:
        # Use the first available groups
        fuel_group = fuel_groups[0]
        electric_group = electric_groups[0]
        
        # Check if we have cost data for comparison
        fuel_cost_per_km = None
        if 'fuel_metrics' in fuel_group and 'avg_fuel_cost_per_km' in fuel_group['fuel_metrics']:
            fuel_cost_per_km = fuel_group['fuel_metrics']['avg_fuel_cost_per_km']
        
        electric_cost_per_km = None
        if 'electric_metrics' in electric_group and 'avg_battery_swap_cost_per_km' in electric_group['electric_metrics']:
            electric_cost_per_km = electric_group['electric_metrics']['avg_battery_swap_cost_per_km']
        
        if fuel_cost_per_km is not None and electric_cost_per_km is not None:
            savings_per_km = fuel_cost_per_km - electric_cost_per_km
            savings_pct = (savings_per_km / fuel_cost_per_km) * 100 if fuel_cost_per_km > 0 else None
            
            findings['cost_savings'] = {
                'fuel_cost_per_km': float(fuel_cost_per_km),
                'electric_cost_per_km': float(electric_cost_per_km),
                'savings_per_km': float(savings_per_km),
                'savings_percentage': float(savings_pct) if savings_pct is not None else None,
                'interpretation': 'Electric motorcycles are cheaper to operate' if savings_per_km > 0 else 'Fuel motorcycles are cheaper to operate'
            }
    
    # Distance comparison
    daily_distance = {}
    for s in daily_summaries:
        if 'statistics' in s and 'distance_km' in s['statistics']:
            daily_distance[s['group']] = s['statistics']['distance_km']['mean']
    
    if daily_distance:
        findings['daily_distance_km'] = daily_distance
    
    # Trip summaries analysis
    trip_summaries = summaries['trip']
    
    # Speed comparison
    trip_speed = {}
    for s in trip_summaries:
        if 'trip_metrics' in s and 'avg_speed_kmh' in s['trip_metrics']:
            trip_speed[s['group']] = s['trip_metrics']['avg_speed_kmh']
    
    if trip_speed:
        findings['avg_speed_kmh'] = trip_speed
    
    return findings

def main():
    """Main function to run summary analysis"""
    # Initialize data loader
    data_loader = DataLoader()
    
    # Run summary analysis
    summaries = run_summary_analysis(data_loader)
    
    print("Summary analysis completed. Results saved to output/reports/")
    
    return summaries

if __name__ == "__main__":
    main()