#!/usr/bin/env python3
"""
15-Minute Daily Temperature Pattern Analysis
==========================================
Analyzes temperature patterns in 15-minute chunks throughout the day
to identify consistent daily behaviors and specific time-based patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 100

def load_and_prepare_data():
    """Load the combined Plus dataset and prepare for analysis"""
    print("Loading Plus dataset for 15-minute pattern analysis...")
    
    # Load data
    df = pd.read_csv('data/combined_plus_sensor_data.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    print(f"Data loaded: {len(df):,} records")
    print(f"Date range: {df['date_time'].min()} to {df['date_time'].max()}")
    
    return df

def create_15min_time_chunks(df):
    """Create 15-minute time chunks for analysis"""
    print("\nCreating 15-minute time chunks...")
    
    # Extract time components
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    df['date'] = df['date_time'].dt.date
    
    # Create 15-minute chunks (0, 15, 30, 45)
    df['minute_chunk'] = (df['minute'] // 15) * 15
    
    # Create time_of_day in 15-minute intervals (0-95 representing 00:00 to 23:45)
    df['time_chunk'] = df['hour'] * 4 + (df['minute'] // 15)
    
    # Create readable time labels
    time_labels = []
    for i in range(96):  # 24 hours * 4 (15-min intervals)
        hour = i // 4
        minute = (i % 4) * 15
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    print(f"Created {len(time_labels)} 15-minute time chunks (96 per day)")
    
    return df, time_labels

def analyze_daily_patterns(df, time_labels):
    """Analyze temperature patterns across 15-minute chunks"""
    print("\nAnalyzing 15-minute daily temperature patterns...")
    
    # Group by time chunk across all days
    daily_pattern = df.groupby('time_chunk')['temperature'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(4)
    
    daily_pattern['time_label'] = [time_labels[i] for i in daily_pattern.index]
    
    # Calculate temperature changes between consecutive 15-min periods
    daily_pattern['temp_change'] = daily_pattern['mean'].diff()
    daily_pattern['temp_change_abs'] = daily_pattern['temp_change'].abs()
    
    print(f"Daily pattern analysis complete for {len(daily_pattern)} time chunks")
    
    return daily_pattern

def find_consistent_patterns(df, daily_pattern):
    """Find specific consistent patterns in the data"""
    print("\nIdentifying specific daily temperature patterns...")
    
    patterns_found = {}
    
    # 1. Find daily temperature minimum and maximum times
    min_temp_time = daily_pattern.loc[daily_pattern['mean'].idxmin(), 'time_label']
    max_temp_time = daily_pattern.loc[daily_pattern['mean'].idxmax(), 'time_label']
    min_temp_value = daily_pattern['mean'].min()
    max_temp_value = daily_pattern['mean'].max()
    
    patterns_found['daily_extremes'] = {
        'min_time': min_temp_time,
        'min_value': min_temp_value,
        'max_time': max_temp_time,
        'max_value': max_temp_value,
        'daily_amplitude': max_temp_value - min_temp_value
    }
    
    # 2. Find largest temperature rises and falls
    max_rise_idx = daily_pattern['temp_change'].idxmax()
    max_fall_idx = daily_pattern['temp_change'].idxmin()
    
    patterns_found['temperature_changes'] = {
        'largest_rise_time': daily_pattern.loc[max_rise_idx, 'time_label'],
        'largest_rise_amount': daily_pattern.loc[max_rise_idx, 'temp_change'],
        'largest_fall_time': daily_pattern.loc[max_fall_idx, 'time_label'],
        'largest_fall_amount': daily_pattern.loc[max_fall_idx, 'temp_change']
    }
    
    # 3. Find periods of stability (low temperature change)
    stable_periods = daily_pattern[daily_pattern['temp_change_abs'] < 0.1]
    
    # 4. Find periods of high activity (large temperature changes)
    active_periods = daily_pattern[daily_pattern['temp_change_abs'] > 0.5]
    
    patterns_found['stability_analysis'] = {
        'stable_periods_count': len(stable_periods),
        'active_periods_count': len(active_periods),
        'most_stable_time': stable_periods.loc[stable_periods['temp_change_abs'].idxmin(), 'time_label'] if len(stable_periods) > 0 else 'None',
        'most_active_time': active_periods.loc[active_periods['temp_change_abs'].idxmax(), 'time_label'] if len(active_periods) > 0 else 'None'
    }
    
    # 5. Identify heating and cooling periods
    heating_period = daily_pattern[daily_pattern['temp_change'] > 0]
    cooling_period = daily_pattern[daily_pattern['temp_change'] < 0]
    
    patterns_found['thermal_periods'] = {
        'heating_periods': len(heating_period),
        'cooling_periods': len(cooling_period),
        'avg_heating_rate': heating_period['temp_change'].mean() if len(heating_period) > 0 else 0,
        'avg_cooling_rate': cooling_period['temp_change'].mean() if len(cooling_period) > 0 else 0
    }
    
    return patterns_found

def analyze_consistency_across_days(df):
    """Analyze how consistent these patterns are across different days"""
    print("\nAnalyzing pattern consistency across days...")
    
    # Create daily summaries
    daily_summary = df.groupby('date').agg({
        'temperature': ['min', 'max', 'mean']
    }).round(4)
    
    daily_summary.columns = ['daily_min', 'daily_max', 'daily_mean']
    daily_summary['daily_amplitude'] = daily_summary['daily_max'] - daily_summary['daily_min']
    
    # Find time of daily min and max for each day
    daily_extremes = df.groupby('date').apply(
        lambda x: pd.Series({
            'min_temp_time': x.loc[x['temperature'].idxmin(), 'time_chunk'],
            'max_temp_time': x.loc[x['temperature'].idxmax(), 'time_chunk']
        })
    )
    
    daily_summary = daily_summary.join(daily_extremes)
    
    # Calculate consistency metrics
    consistency_metrics = {
        'min_time_std': daily_summary['min_temp_time'].std(),
        'max_time_std': daily_summary['max_temp_time'].std(),
        'amplitude_std': daily_summary['daily_amplitude'].std(),
        'avg_amplitude': daily_summary['daily_amplitude'].mean(),
        'most_common_min_time': daily_summary['min_temp_time'].mode().iloc[0] if len(daily_summary['min_temp_time'].mode()) > 0 else 'Variable',
        'most_common_max_time': daily_summary['max_temp_time'].mode().iloc[0] if len(daily_summary['max_temp_time'].mode()) > 0 else 'Variable'
    }
    
    return daily_summary, consistency_metrics

def create_visualizations(daily_pattern, patterns_found, time_labels):
    """Create comprehensive visualizations of 15-minute patterns"""
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main daily temperature pattern
    plt.subplot(3, 2, 1)
    x_ticks = range(0, 96, 8)  # Every 2 hours
    plt.plot(daily_pattern.index, daily_pattern['mean'], linewidth=2, color='blue', label='Average Temperature')
    plt.fill_between(daily_pattern.index, 
                     daily_pattern['mean'] - daily_pattern['std'],
                     daily_pattern['mean'] + daily_pattern['std'],
                     alpha=0.3, color='blue', label='±1 Standard Deviation')
    
    plt.xlabel('Time of Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Daily Temperature Pattern (15-minute intervals)', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, [time_labels[i] for i in x_ticks], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Temperature changes (rate of change)
    plt.subplot(3, 2, 2)
    colors = ['red' if x < 0 else 'green' for x in daily_pattern['temp_change']]
    plt.bar(daily_pattern.index, daily_pattern['temp_change'], color=colors, alpha=0.7, width=0.8)
    plt.xlabel('Time of Day')
    plt.ylabel('Temperature Change (°C/15min)')
    plt.title('Temperature Rate of Change (15-minute intervals)', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, [time_labels[i] for i in x_ticks], rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # 3. Data availability heatmap
    plt.subplot(3, 2, 3)
    plt.bar(daily_pattern.index, daily_pattern['count'], alpha=0.7, color='purple')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Measurements')
    plt.title('Data Availability by Time of Day', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, [time_labels[i] for i in x_ticks], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Temperature variability
    plt.subplot(3, 2, 4)
    plt.plot(daily_pattern.index, daily_pattern['std'], linewidth=2, color='orange', marker='o', markersize=2)
    plt.xlabel('Time of Day')
    plt.ylabel('Temperature Standard Deviation (°C)')
    plt.title('Temperature Variability Throughout Day', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, [time_labels[i] for i in x_ticks], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Min-Max range
    plt.subplot(3, 2, 5)
    plt.fill_between(daily_pattern.index, daily_pattern['min'], daily_pattern['max'], 
                     alpha=0.4, color='gray', label='Min-Max Range')
    plt.plot(daily_pattern.index, daily_pattern['median'], linewidth=2, color='red', label='Median')
    plt.xlabel('Time of Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Range (Min-Max) by Time of Day', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, [time_labels[i] for i in x_ticks], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Highlight key pattern times
    plt.subplot(3, 2, 6)
    plt.plot(daily_pattern.index, daily_pattern['mean'], linewidth=2, color='blue', alpha=0.7)
    
    # Mark important points
    min_idx = daily_pattern['mean'].idxmin()
    max_idx = daily_pattern['mean'].idxmax()
    
    plt.scatter(min_idx, daily_pattern.loc[min_idx, 'mean'], 
               color='blue', s=100, marker='v', label=f'Daily Min ({time_labels[min_idx]})', zorder=5)
    plt.scatter(max_idx, daily_pattern.loc[max_idx, 'mean'], 
               color='red', s=100, marker='^', label=f'Daily Max ({time_labels[max_idx]})', zorder=5)
    
    # Mark largest temperature changes
    max_rise_idx = daily_pattern['temp_change'].idxmax()
    max_fall_idx = daily_pattern['temp_change'].idxmin()
    
    plt.scatter(max_rise_idx, daily_pattern.loc[max_rise_idx, 'mean'], 
               color='green', s=80, marker='>', label=f'Largest Rise ({time_labels[max_rise_idx]})', zorder=5)
    plt.scatter(max_fall_idx, daily_pattern.loc[max_fall_idx, 'mean'], 
               color='purple', s=80, marker='<', label=f'Largest Fall ({time_labels[max_fall_idx]})', zorder=5)
    
    plt.xlabel('Time of Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Key Daily Temperature Events', fontsize=14, fontweight='bold')
    plt.xticks(x_ticks, [time_labels[i] for i in x_ticks], rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('15min_daily_temperature_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved visualization: 15min_daily_temperature_patterns.png")
    plt.show()

def print_pattern_summary(patterns_found, consistency_metrics, daily_pattern, time_labels):
    """Print a comprehensive summary of discovered patterns"""
    print("\n" + "="*80)
    print("15-MINUTE DAILY TEMPERATURE PATTERN ANALYSIS RESULTS")
    print("="*80)
    
    # Daily extremes
    extremes = patterns_found['daily_extremes']
    print(f"\n📊 DAILY TEMPERATURE EXTREMES:")
    print(f"🌡️  Coldest time of day: {extremes['min_time']} ({extremes['min_value']:.2f}°C)")
    print(f"🌡️  Warmest time of day: {extremes['max_time']} ({extremes['max_value']:.2f}°C)")
    print(f"🌡️  Daily temperature amplitude: {extremes['daily_amplitude']:.2f}°C")
    
    # Temperature changes
    changes = patterns_found['temperature_changes']
    print(f"\n🔄 TEMPERATURE CHANGE PATTERNS:")
    print(f"⬆️  Largest temperature rise: {changes['largest_rise_amount']:.3f}°C at {changes['largest_rise_time']}")
    print(f"⬇️  Largest temperature drop: {changes['largest_fall_amount']:.3f}°C at {changes['largest_fall_time']}")
    
    # Thermal periods
    thermal = patterns_found['thermal_periods']
    print(f"\n🌡️  HEATING & COOLING PERIODS:")
    print(f"🔥 Heating periods: {thermal['heating_periods']} (avg rate: {thermal['avg_heating_rate']:.3f}°C/15min)")
    print(f"❄️  Cooling periods: {thermal['cooling_periods']} (avg rate: {thermal['avg_cooling_rate']:.3f}°C/15min)")
    
    # Stability analysis
    stability = patterns_found['stability_analysis']
    print(f"\n⚖️  TEMPERATURE STABILITY:")
    print(f"📍 Most stable time: {stability['most_stable_time']}")
    print(f"🌪️  Most variable time: {stability['most_active_time']}")
    print(f"📊 Stable periods (change < 0.1°C): {stability['stable_periods_count']}")
    print(f"📊 Active periods (change > 0.5°C): {stability['active_periods_count']}")
    
    # Consistency metrics
    print(f"\n🔄 PATTERN CONSISTENCY ACROSS DAYS:")
    print(f"📅 Average daily amplitude: {consistency_metrics['avg_amplitude']:.2f}°C")
    print(f"📅 Amplitude variability: {consistency_metrics['amplitude_std']:.2f}°C")
    print(f"🕐 Min temperature time consistency: ±{consistency_metrics['min_time_std']:.1f} time chunks")
    print(f"🕐 Max temperature time consistency: ±{consistency_metrics['max_time_std']:.1f} time chunks")
    
    # Specific pattern insights
    print(f"\n🎯 KEY PATTERN INSIGHTS:")
    
    # Find periods of rapid change
    rapid_changes = daily_pattern[daily_pattern['temp_change_abs'] > 0.3]
    if len(rapid_changes) > 0:
        print(f"⚡ Rapid temperature changes occur at: {', '.join([f'{time_labels[i]}' for i in rapid_changes.index[:5]])}")
    
    # Find stable periods
    stable_periods = daily_pattern[daily_pattern['temp_change_abs'] < 0.05]
    if len(stable_periods) > 0:
        print(f"🎯 Most stable temperatures occur at: {', '.join([f'{time_labels[i]}' for i in stable_periods.index[:5]])}")
    
    # Temperature gradient analysis
    morning_warming = daily_pattern[(daily_pattern.index >= 20) & (daily_pattern.index <= 40)]['temp_change'].mean()  # 5AM-10AM
    evening_cooling = daily_pattern[(daily_pattern.index >= 68) & (daily_pattern.index <= 88)]['temp_change'].mean()  # 5PM-10PM
    
    print(f"🌅 Morning warming rate (5-10 AM): {morning_warming:.3f}°C/15min")
    print(f"🌆 Evening cooling rate (5-10 PM): {evening_cooling:.3f}°C/15min")
    
    print(f"\n💡 PRACTICAL INSIGHTS:")
    print(f"• Most predictable temperature time: {extremes['min_time']} (daily minimum)")
    print(f"• Highest temperature variability: Around {time_labels[daily_pattern['std'].idxmax()]}")
    print(f"• Best time for temperature predictions: Stable periods around {stability['most_stable_time']}")
    
    print("="*80)

def main():
    """Main analysis function"""
    print("Starting 15-Minute Daily Temperature Pattern Analysis")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create 15-minute time chunks
    df, time_labels = create_15min_time_chunks(df)
    
    # Analyze daily patterns
    daily_pattern = analyze_daily_patterns(df, time_labels)
    
    # Find specific patterns
    patterns_found = find_consistent_patterns(df, daily_pattern)
    
    # Analyze consistency across days
    daily_summary, consistency_metrics = analyze_consistency_across_days(df)
    
    # Create visualizations
    create_visualizations(daily_pattern, patterns_found, time_labels)
    
    # Print comprehensive summary
    print_pattern_summary(patterns_found, consistency_metrics, daily_pattern, time_labels)
    
    # Save detailed results
    daily_pattern.to_csv('15min_daily_patterns.csv', index=True)
    print(f"\n💾 Saved detailed results to: 15min_daily_patterns.csv")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📊 Discovered patterns in {len(daily_pattern)} 15-minute time chunks")
    print(f"📈 Analyzed {len(df):,} temperature measurements")
    
    return daily_pattern, patterns_found, consistency_metrics

if __name__ == "__main__":
    daily_pattern, patterns_found, consistency_metrics = main()
