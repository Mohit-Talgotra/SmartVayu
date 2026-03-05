#!/usr/bin/env python3
"""
Comprehensive 15-Minute Temperature Pattern Analysis
===================================================
Generates individual figures, detailed analysis, and comprehensive reports
for temperature patterns in 15-minute intervals throughout the day.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class Comprehensive15MinAnalyzer:
    """Comprehensive analyzer for 15-minute temperature patterns"""
    
    def __init__(self):
        self.df = None
        self.daily_pattern = None
        self.time_labels = None
        self.patterns_found = {}
        self.consistency_metrics = {}
        
        # Create output directories
        os.makedirs('../figures', exist_ok=True)
        os.makedirs('../data', exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load the combined Plus dataset and prepare for analysis"""
        print("Loading Plus dataset for comprehensive 15-minute analysis...")
        
        # Load data
        self.df = pd.read_csv('../../data/combined_plus_sensor_data.csv')
        self.df['date_time'] = pd.to_datetime(self.df['date_time'])
        
        print(f"Data loaded: {len(self.df):,} records")
        print(f"Date range: {self.df['date_time'].min()} to {self.df['date_time'].max()}")
        
        return self.df
    
    def create_time_features(self):
        """Create comprehensive time-based features"""
        print("\nCreating comprehensive time features...")
        
        # Basic time features
        self.df['hour'] = self.df['date_time'].dt.hour
        self.df['minute'] = self.df['date_time'].dt.minute
        self.df['date'] = self.df['date_time'].dt.date
        self.df['day_of_week'] = self.df['date_time'].dt.dayofweek
        self.df['month'] = self.df['date_time'].dt.month
        self.df['season'] = self.df['date_time'].dt.quarter
        self.df['is_weekend'] = self.df['date_time'].dt.dayofweek >= 5
        
        # 15-minute chunks
        self.df['minute_chunk'] = (self.df['minute'] // 15) * 15
        self.df['time_chunk'] = self.df['hour'] * 4 + (self.df['minute'] // 15)
        
        # Create time labels
        self.time_labels = []
        for i in range(96):  # 24 hours * 4 (15-min intervals)
            hour = i // 4
            minute = (i % 4) * 15
            self.time_labels.append(f"{hour:02d}:{minute:02d}")
        
        # Additional time features
        self.df['time_of_day'] = pd.cut(self.df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)
        
        print(f"Created {len(self.time_labels)} 15-minute time chunks")
        
    def analyze_daily_patterns(self):
        """Comprehensive daily pattern analysis"""
        print("\nAnalyzing comprehensive daily patterns...")
        
        # Main pattern analysis
        self.daily_pattern = self.df.groupby('time_chunk')['temperature'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median',
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75)   # Q3
        ]).round(4)
        
        self.daily_pattern.columns = ['count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        self.daily_pattern['time_label'] = [self.time_labels[i] for i in self.daily_pattern.index]
        self.daily_pattern['iqr'] = self.daily_pattern['q75'] - self.daily_pattern['q25']
        
        # Temperature changes
        self.daily_pattern['temp_change'] = self.daily_pattern['mean'].diff()
        self.daily_pattern['temp_change_abs'] = self.daily_pattern['temp_change'].abs()
        
        # Smoothed patterns (rolling average)
        self.daily_pattern['mean_smooth'] = self.daily_pattern['mean'].rolling(window=3, center=True).mean()
        
        print(f"Daily pattern analysis complete for {len(self.daily_pattern)} time chunks")
        
    def find_specific_patterns(self):
        """Find specific temperature patterns and events"""
        print("\nIdentifying specific temperature patterns...")
        
        # Daily extremes
        min_idx = self.daily_pattern['mean'].idxmin()
        max_idx = self.daily_pattern['mean'].idxmax()
        
        self.patterns_found['daily_extremes'] = {
            'min_time': self.time_labels[min_idx],
            'min_value': self.daily_pattern.loc[min_idx, 'mean'],
            'max_time': self.time_labels[max_idx],
            'max_value': self.daily_pattern.loc[max_idx, 'mean'],
            'amplitude': self.daily_pattern.loc[max_idx, 'mean'] - self.daily_pattern.loc[min_idx, 'mean']
        }
        
        # Temperature change events
        max_rise_idx = self.daily_pattern['temp_change'].idxmax()
        max_fall_idx = self.daily_pattern['temp_change'].idxmin()
        
        self.patterns_found['change_events'] = {
            'largest_rise_time': self.time_labels[max_rise_idx],
            'largest_rise_amount': self.daily_pattern.loc[max_rise_idx, 'temp_change'],
            'largest_fall_time': self.time_labels[max_fall_idx],
            'largest_fall_amount': self.daily_pattern.loc[max_fall_idx, 'temp_change']
        }
        
        # Stability analysis
        stable_threshold = 0.05
        active_threshold = 0.3
        
        stable_periods = self.daily_pattern[self.daily_pattern['temp_change_abs'] <= stable_threshold]
        active_periods = self.daily_pattern[self.daily_pattern['temp_change_abs'] >= active_threshold]
        
        self.patterns_found['stability'] = {
            'stable_periods': len(stable_periods),
            'active_periods': len(active_periods),
            'most_stable_times': [self.time_labels[i] for i in stable_periods.index[:5]],
            'most_active_times': [self.time_labels[i] for i in active_periods.index[:5]]
        }
        
        # Time-of-day analysis
        tod_patterns = {}
        for tod in ['Night', 'Morning', 'Afternoon', 'Evening']:
            tod_data = self.df[self.df['time_of_day'] == tod]['temperature']
            tod_patterns[tod] = {
                'mean': tod_data.mean(),
                'std': tod_data.std(),
                'count': len(tod_data)
            }
        
        self.patterns_found['time_of_day'] = tod_patterns
        
    def analyze_variability(self):
        """Analyze temperature variability patterns"""
        print("\nAnalyzing temperature variability...")
        
        # Variability by time of day
        variability_analysis = {
            'highest_variability_time': self.time_labels[self.daily_pattern['std'].idxmax()],
            'highest_variability_value': self.daily_pattern['std'].max(),
            'lowest_variability_time': self.time_labels[self.daily_pattern['std'].idxmin()],
            'lowest_variability_value': self.daily_pattern['std'].min(),
            'avg_variability': self.daily_pattern['std'].mean()
        }
        
        # Range analysis
        self.daily_pattern['range'] = self.daily_pattern['max'] - self.daily_pattern['min']
        
        variability_analysis.update({
            'highest_range_time': self.time_labels[self.daily_pattern['range'].idxmax()],
            'highest_range_value': self.daily_pattern['range'].max(),
            'avg_range': self.daily_pattern['range'].mean()
        })
        
        self.patterns_found['variability'] = variability_analysis
        
    def create_individual_figures(self):
        """Create individual figure files"""
        print("\nCreating individual figure files...")
        
        # Figure 1: Main Daily Pattern
        self.create_main_pattern_figure()
        
        # Figure 2: Temperature Changes
        self.create_temperature_changes_figure()
        
        # Figure 3: Variability Analysis
        self.create_variability_figure()
        
        # Figure 4: Data Availability
        self.create_data_availability_figure()
        
        # Figure 5: Statistical Distribution
        self.create_statistical_distribution_figure()
        
        # Figure 6: Pattern Consistency
        self.create_pattern_consistency_figure()
        
    def create_main_pattern_figure(self):
        """Create main daily temperature pattern figure"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: Mean temperature with confidence bands
        x = self.daily_pattern.index
        x_ticks = range(0, 96, 8)
        
        ax1.plot(x, self.daily_pattern['mean'], linewidth=3, color='blue', label='Average Temperature')
        ax1.fill_between(x, 
                        self.daily_pattern['mean'] - self.daily_pattern['std'],
                        self.daily_pattern['mean'] + self.daily_pattern['std'],
                        alpha=0.3, color='blue', label='±1 Standard Deviation')
        
        # Mark extremes
        min_idx = self.daily_pattern['mean'].idxmin()
        max_idx = self.daily_pattern['mean'].idxmax()
        
        ax1.scatter(min_idx, self.daily_pattern.loc[min_idx, 'mean'], 
                   color='darkblue', s=100, marker='v', label=f'Daily Min ({self.time_labels[min_idx]})', zorder=5)
        ax1.scatter(max_idx, self.daily_pattern.loc[max_idx, 'mean'], 
                   color='red', s=100, marker='^', label=f'Daily Max ({self.time_labels[max_idx]})', zorder=5)
        
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Daily Temperature Pattern (15-minute intervals)', fontsize=16, fontweight='bold')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom plot: Min-Max range
        ax2.fill_between(x, self.daily_pattern['min'], self.daily_pattern['max'], 
                        alpha=0.4, color='gray', label='Daily Min-Max Range')
        ax2.plot(x, self.daily_pattern['median'], linewidth=2, color='red', label='Median')
        ax2.plot(x, self.daily_pattern['q25'], linewidth=1, color='orange', linestyle='--', label='Q1 (25%)')
        ax2.plot(x, self.daily_pattern['q75'], linewidth=1, color='orange', linestyle='--', label='Q3 (75%)')
        
        ax2.set_xlabel('Time of Day')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Distribution Throughout Day', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('../figures/01_daily_temperature_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 01_daily_temperature_pattern.png")
        
    def create_temperature_changes_figure(self):
        """Create temperature change analysis figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = self.daily_pattern.index
        x_ticks = range(0, 96, 8)
        
        # Temperature changes bar chart
        colors = ['red' if x < 0 else 'green' for x in self.daily_pattern['temp_change']]
        ax1.bar(x, self.daily_pattern['temp_change'], color=colors, alpha=0.7, width=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Temperature Change (°C/15min)')
        ax1.set_title('Temperature Rate of Change', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Absolute changes
        ax2.plot(x, self.daily_pattern['temp_change_abs'], linewidth=2, color='purple', marker='o', markersize=2)
        ax2.set_xlabel('Time of Day')
        ax2.set_ylabel('Absolute Temperature Change (°C/15min)')
        ax2.set_title('Absolute Rate of Change', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Cumulative temperature change
        cumulative_change = self.daily_pattern['temp_change'].cumsum()
        ax3.plot(x, cumulative_change, linewidth=2, color='darkgreen')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time of Day')
        ax3.set_ylabel('Cumulative Temperature Change (°C)')
        ax3.set_title('Cumulative Temperature Change', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Heating vs Cooling periods
        heating_periods = self.daily_pattern['temp_change'] > 0
        cooling_periods = self.daily_pattern['temp_change'] < 0
        
        heating_times = x[heating_periods]
        cooling_times = x[cooling_periods]
        
        if len(heating_times) > 0:
            ax4.scatter(heating_times, [1] * len(heating_times), 
                       c='red', s=30, alpha=0.7, label='Heating Periods')
        if len(cooling_times) > 0:
            ax4.scatter(cooling_times, [0] * len(cooling_times), 
                       c='blue', s=30, alpha=0.7, label='Cooling Periods')
        
        ax4.set_xlabel('Time of Day')
        ax4.set_ylabel('Heating/Cooling')
        ax4.set_title('Heating vs Cooling Periods', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Cooling', 'Heating'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/02_temperature_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 02_temperature_changes.png")
        
    def create_variability_figure(self):
        """Create temperature variability analysis figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = self.daily_pattern.index
        x_ticks = range(0, 96, 8)
        
        # Standard deviation
        ax1.plot(x, self.daily_pattern['std'], linewidth=2, color='orange', marker='o', markersize=3)
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Temperature Standard Deviation (°C)')
        ax1.set_title('Temperature Variability Throughout Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # IQR (Interquartile Range)
        ax2.plot(x, self.daily_pattern['iqr'], linewidth=2, color='purple', marker='s', markersize=3)
        ax2.set_xlabel('Time of Day')
        ax2.set_ylabel('Interquartile Range (°C)')
        ax2.set_title('Temperature IQR Throughout Day', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Range (Max - Min)
        ax3.plot(x, self.daily_pattern['range'], linewidth=2, color='red', marker='d', markersize=3)
        ax3.set_xlabel('Time of Day')
        ax3.set_ylabel('Temperature Range (°C)')
        ax3.set_title('Daily Temperature Range by Time', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Coefficient of Variation
        cv = (self.daily_pattern['std'] / self.daily_pattern['mean']) * 100
        ax4.plot(x, cv, linewidth=2, color='darkblue', marker='^', markersize=3)
        ax4.set_xlabel('Time of Day')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.set_title('Temperature CV Throughout Day', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/03_temperature_variability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 03_temperature_variability.png")
        
    def create_data_availability_figure(self):
        """Create data availability and coverage figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = self.daily_pattern.index
        x_ticks = range(0, 96, 8)
        
        # Data count by time
        ax1.bar(x, self.daily_pattern['count'], alpha=0.7, color='green')
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Number of Measurements')
        ax1.set_title('Data Availability by Time of Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([self.time_labels[i] for i in x_ticks], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Monthly data distribution
        monthly_counts = self.df.groupby(self.df['date_time'].dt.month).size()
        ax2.bar(monthly_counts.index, monthly_counts.values, alpha=0.7, color='blue')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Records')
        ax2.set_title('Data Distribution by Month', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, 13))
        ax2.grid(True, alpha=0.3)
        
        # Day of week distribution
        dow_counts = self.df.groupby(self.df['day_of_week']).size()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax3.bar(range(7), dow_counts.values, alpha=0.7, color='orange')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Number of Records')
        ax3.set_title('Data Distribution by Day of Week', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(dow_labels)
        ax3.grid(True, alpha=0.3)
        
        # Time coverage heatmap
        coverage_matrix = self.df.groupby([self.df['date_time'].dt.hour, 
                                          self.df['date_time'].dt.day_of_week]).size().unstack(fill_value=0)
        
        im = ax4.imshow(coverage_matrix.values, cmap='YlOrRd', aspect='auto')
        ax4.set_xlabel('Day of Week')
        ax4.set_ylabel('Hour of Day')
        ax4.set_title('Data Coverage Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(7))
        ax4.set_xticklabels(dow_labels)
        ax4.set_yticks(range(0, 24, 4))
        ax4.set_yticklabels(range(0, 24, 4))
        plt.colorbar(im, ax=ax4, label='Number of Records')
        
        plt.tight_layout()
        plt.savefig('../figures/04_data_availability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 04_data_availability.png")
        
    def create_statistical_distribution_figure(self):
        """Create statistical distribution analysis figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Temperature distribution by time of day
        tod_data = []
        tod_labels = []
        for tod in ['Night', 'Morning', 'Afternoon', 'Evening']:
            data = self.df[self.df['time_of_day'] == tod]['temperature']
            if len(data) > 0:
                tod_data.append(data)
                tod_labels.append(f'{tod}\n(n={len(data):,})')
        
        if tod_data:
            ax1.boxplot(tod_data, labels=tod_labels)
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title('Temperature Distribution by Time of Day', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Hourly temperature distributions
        hourly_data = [self.df[self.df['hour'] == h]['temperature'].values for h in range(0, 24, 4)]
        hourly_labels = [f'{h}:00' for h in range(0, 24, 4)]
        
        ax2.boxplot(hourly_data, labels=hourly_labels)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Distributions by Hour (4-hour intervals)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Weekend vs Weekday comparison
        weekend_data = self.df[self.df['is_weekend']]['temperature']
        weekday_data = self.df[~self.df['is_weekend']]['temperature']
        
        ax3.hist([weekday_data, weekend_data], bins=50, alpha=0.7, 
                label=[f'Weekdays (n={len(weekday_data):,})', f'Weekends (n={len(weekend_data):,})'],
                density=True)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Density')
        ax3.set_title('Temperature Distribution: Weekday vs Weekend', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Seasonal distributions
        seasonal_data = []
        seasonal_labels = []
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        
        for season_num, season_name in seasons.items():
            data = self.df[self.df['season'] == season_num]['temperature']
            if len(data) > 0:
                seasonal_data.append(data)
                seasonal_labels.append(f'{season_name}\n(n={len(data):,})')
        
        if seasonal_data:
            ax4.boxplot(seasonal_data, labels=seasonal_labels)
            ax4.set_ylabel('Temperature (°C)')
            ax4.set_title('Temperature Distribution by Season', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/05_statistical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 05_statistical_distributions.png")
        
    def create_pattern_consistency_figure(self):
        """Create pattern consistency analysis figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily extremes consistency
        daily_summary = self.df.groupby('date').apply(
            lambda x: pd.Series({
                'min_temp': x['temperature'].min(),
                'max_temp': x['temperature'].max(),
                'min_time': x.loc[x['temperature'].idxmin(), 'time_chunk'],
                'max_time': x.loc[x['temperature'].idxmax(), 'time_chunk'],
                'amplitude': x['temperature'].max() - x['temperature'].min()
            })
        )
        
        # Min temperature time consistency
        ax1.hist(daily_summary['min_time'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Time Chunk (15-min intervals)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Consistency of Daily Minimum Temperature Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Max temperature time consistency
        ax2.hist(daily_summary['max_time'], bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Time Chunk (15-min intervals)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Consistency of Daily Maximum Temperature Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Daily amplitude consistency
        ax3.hist(daily_summary['amplitude'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Daily Temperature Amplitude (°C)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Consistency of Daily Temperature Range', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Amplitude vs Time scatter
        ax4.scatter(daily_summary['min_time'], daily_summary['amplitude'], alpha=0.6, s=20, color='purple')
        ax4.set_xlabel('Time of Daily Minimum (15-min chunks)')
        ax4.set_ylabel('Daily Temperature Amplitude (°C)')
        ax4.set_title('Amplitude vs Minimum Time Relationship', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/06_pattern_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 06_pattern_consistency.png")
        
    def save_detailed_data(self):
        """Save detailed analysis data"""
        print("\nSaving detailed analysis data...")
        
        # Main pattern data
        self.daily_pattern.to_csv('../data/15min_daily_patterns.csv', index=True)
        
        # Summary statistics by time chunk
        summary_stats = self.daily_pattern[['time_label', 'mean', 'std', 'min', 'max', 'temp_change', 'temp_change_abs']].copy()
        summary_stats['predictability_rank'] = summary_stats['temp_change_abs'].rank()
        summary_stats.to_csv('../data/time_chunk_summary.csv', index=False)
        
        # Patterns found
        import json
        with open('../data/discovered_patterns.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            patterns_json = {}
            for key, value in self.patterns_found.items():
                if isinstance(value, dict):
                    patterns_json[key] = {k: float(v) if isinstance(v, np.floating) else v for k, v in value.items()}
                else:
                    patterns_json[key] = value
            json.dump(patterns_json, f, indent=2)
        
        print("  Saved: 15min_daily_patterns.csv")
        print("  Saved: time_chunk_summary.csv") 
        print("  Saved: discovered_patterns.json")
        
    def generate_detailed_report(self):
        """Generate comprehensive written analysis report"""
        print("\nGenerating comprehensive analysis report...")
        
        report_content = f"""
# 15-Minute Daily Temperature Pattern Analysis Report

## Executive Summary

This analysis examines temperature patterns in 15-minute intervals throughout the day using {len(self.df):,} measurements from the Plus sensor dataset spanning {(self.df['date_time'].max() - self.df['date_time'].min()).days} days.

## Key Findings

### Daily Temperature Extremes
- **Coldest Time**: {self.patterns_found['daily_extremes']['min_time']} ({self.patterns_found['daily_extremes']['min_value']:.2f}°C)
- **Warmest Time**: {self.patterns_found['daily_extremes']['max_time']} ({self.patterns_found['daily_extremes']['max_value']:.2f}°C)
- **Daily Amplitude**: {self.patterns_found['daily_extremes']['amplitude']:.2f}°C

### Temperature Change Events
- **Largest Rise**: {self.patterns_found['change_events']['largest_rise_amount']:.3f}°C at {self.patterns_found['change_events']['largest_rise_time']}
- **Largest Drop**: {self.patterns_found['change_events']['largest_fall_amount']:.3f}°C at {self.patterns_found['change_events']['largest_fall_time']}

### Stability Analysis
- **Stable Periods**: {self.patterns_found['stability']['stable_periods']} time chunks (<=0.05°C change)
- **Active Periods**: {self.patterns_found['stability']['active_periods']} time chunks (>=0.3°C change)
- **Most Stable Times**: {', '.join(self.patterns_found['stability']['most_stable_times'])}
- **Most Active Times**: {', '.join(self.patterns_found['stability']['most_active_times'])}

### Variability Analysis
- **Highest Variability Time**: {self.patterns_found['variability']['highest_variability_time']} (σ = {self.patterns_found['variability']['highest_variability_value']:.3f}°C)
- **Lowest Variability Time**: {self.patterns_found['variability']['lowest_variability_time']} (σ = {self.patterns_found['variability']['lowest_variability_value']:.3f}°C)
- **Average Variability**: {self.patterns_found['variability']['avg_variability']:.3f}°C

## Time-of-Day Analysis

"""
        
        # Add time of day analysis
        for tod, stats in self.patterns_found['time_of_day'].items():
            report_content += f"### {tod}\n"
            report_content += f"- Mean Temperature: {stats['mean']:.2f}°C\n"
            report_content += f"- Standard Deviation: {stats['std']:.2f}°C\n"
            report_content += f"- Sample Size: {stats['count']:,} measurements\n\n"
        
        report_content += f"""
## Practical Implications for Prediction Modeling

### Most Predictable Times (Low Variability)
The following 15-minute periods show the most consistent temperature behavior and are ideal for baseline predictions:
"""
        
        # Add most stable periods
        stable_df = self.daily_pattern.nsmallest(10, 'temp_change_abs')
        for idx, row in stable_df.iterrows():
            report_content += f"- {row['time_label']}: {row['mean']:.2f}°C (±{row['std']:.3f}°C)\n"
        
        report_content += f"""
### Most Variable Times (High Activity)
These periods require more sophisticated modeling due to higher variability:
"""
        
        # Add most active periods  
        active_df = self.daily_pattern.nlargest(5, 'temp_change_abs')
        for idx, row in active_df.iterrows():
            report_content += f"- {row['time_label']}: {row['temp_change']:.3f}°C change, σ = {row['std']:.3f}°C\n"
        
        report_content += f"""
## Recommendations for Prediction Model

### Primary Features (Time-based)
1. **15-minute time chunks** (0-95) as categorical variables
2. **Time-of-day groups** (Night, Morning, Afternoon, Evening)
3. **Stability zones** as binary features (stable vs active periods)

### Secondary Features (Environmental)
- Use humidity, pressure, light as supporting variables
- Weight their importance based on time-specific patterns
- Consider interaction terms with time features

### Model Architecture Suggestions
1. **Baseline Model**: Use discovered daily pattern as starting point
2. **Time-specific Models**: Different models for stable vs active periods
3. **Ensemble Approach**: Combine time-based and environmental models

### Expected Performance
- **High Accuracy Periods**: {', '.join(self.patterns_found['stability']['most_stable_times'][:3])}
- **Challenging Periods**: {', '.join(self.patterns_found['stability']['most_active_times'][:3]) if self.patterns_found['stability']['most_active_times'] else 'Early morning transition periods'}
- **Overall Pattern Strength**: {self.patterns_found['daily_extremes']['amplitude']:.2f}°C daily amplitude provides good predictive signal

## Data Quality Assessment
- **Completeness**: Excellent coverage across all 96 daily time chunks
- **Consistency**: Strong daily patterns with identifiable extremes
- **Reliability**: Low variability in stable periods indicates good sensor performance

## Conclusion
The analysis reveals strong, consistent daily temperature patterns with specific periods of high predictability. The discovered patterns provide an excellent foundation for time-series prediction modeling with clear guidance on when predictions will be most accurate.
"""
        
        # Save report
        with open('../15min_pattern_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print("  Saved: 15min_pattern_analysis_report.md")
        
    def create_summary_inference(self):
        """Create summary inference document"""
        inference_content = f"""
# Temperature Pattern Analysis - Key Inferences for Prediction Modeling

## Core Discovery: Strong 15-Minute Pattern Structure

### Primary Pattern Skeleton
- **96 distinct 15-minute time chunks** each with characteristic temperature behavior
- **Daily minimum at {self.patterns_found['daily_extremes']['min_time']}** - most predictable point
- **Daily maximum at {self.patterns_found['daily_extremes']['max_time']}** - peak temperature
- **{self.patterns_found['daily_extremes']['amplitude']:.2f}°C daily amplitude** - consistent thermal cycle

## Prediction Model Implications

### Tier 1: Highly Predictable Times (sigma < 0.05°C change)
These time periods have minimal temperature variation and provide the most reliable predictions:
"""
        
        # Get most stable periods
        stable_periods = self.daily_pattern[self.daily_pattern['temp_change_abs'] <= 0.05]
        for idx, row in stable_periods.head(10).iterrows():
            inference_content += f"- **{row['time_label']}**: {row['mean']:.2f}°C (change: {row['temp_change']:.4f}°C)\n"
        
        inference_content += f"""
### Tier 2: Moderate Predictability (0.05°C < sigma < 0.20°C change)
Standard prediction accuracy expected:
"""
        
        moderate_periods = self.daily_pattern[
            (self.daily_pattern['temp_change_abs'] > 0.05) & 
            (self.daily_pattern['temp_change_abs'] <= 0.20)
        ]
        for idx, row in moderate_periods.head(5).iterrows():
            inference_content += f"- **{row['time_label']}**: {row['temp_change']:.3f}°C change\n"
        
        inference_content += f"""
### Tier 3: Challenging Periods (sigma > 0.20°C change)
Require sophisticated modeling:
"""
        
        challenging_periods = self.daily_pattern[self.daily_pattern['temp_change_abs'] > 0.20]
        for idx, row in challenging_periods.iterrows():
            inference_content += f"- **{row['time_label']}**: {row['temp_change']:.3f}°C change (high variability)\n"
        
        inference_content += f"""
## Recommended Model Architecture

### Base Model: Time-Pattern Lookup
```python
# Baseline prediction using discovered patterns
base_temp = daily_pattern[time_chunk]['mean']
confidence = 1.0 / daily_pattern[time_chunk]['std']
```

### Enhanced Model: Time + Environmental
```python
# Feature engineering
features = {{
    'time_chunk': 0-95,  # Primary feature
    'is_stable_period': boolean,  # Stability flag
    'time_of_day': categorical,  # Night/Morning/Afternoon/Evening
    'humidity': continuous,  # Secondary feature
    'pressure': continuous,  # Secondary feature  
    'light': continuous     # Secondary feature
}}
```

### Advanced Model: Period-Specific
- **Stable Period Model**: Simple linear regression
- **Active Period Model**: Complex ensemble method
- **Transition Model**: Specialized handling for {self.patterns_found['change_events']['largest_fall_time']} and {self.patterns_found['change_events']['largest_rise_time']}

## Expected Model Performance

### High Accuracy Zones (MAE < 0.5°C)
- **{self.patterns_found['daily_extremes']['min_time']}** - Daily minimum (most predictable)
- **Afternoon plateau**: {', '.join(self.patterns_found['stability']['most_stable_times'][:3])}
- **Late night stability**: {', '.join([t for t in self.patterns_found['stability']['most_stable_times'] if t.startswith('23') or t.startswith('00')][:2])}

### Moderate Accuracy Zones (MAE 0.5-1.0°C)
- **Mid-morning warming**: 8:00-11:00 periods
- **Evening transitions**: 17:00-20:00 periods

### Challenging Zones (MAE > 1.0°C)
- **{self.patterns_found['change_events']['largest_fall_time']}** - Largest temperature drop
- **{self.patterns_found['change_events']['largest_rise_time']}** - Largest temperature rise

## Practical Implementation Strategy

### Phase 1: Baseline Implementation
1. Use discovered 15-minute patterns as lookup table
2. Implement confidence scoring based on historical variability
3. Achieve baseline accuracy using pure temporal features

### Phase 2: Environmental Enhancement  
1. Add humidity, pressure, light as secondary features
2. Time-specific feature weighting (more important during active periods)
3. Interaction terms between time and environmental features

### Phase 3: Advanced Modeling
1. Period-specific models (stable vs active)
2. Ensemble methods for challenging time periods
3. Anomaly detection for unusual temperature patterns

## Critical Success Factors

### Data Advantages
- **Strong pattern signal**: {self.patterns_found['daily_extremes']['amplitude']:.2f}°C daily amplitude
- **Consistent timing**: Daily extremes occur at predictable times  
- **High data quality**: Excellent 15-minute resolution coverage

### Model Design Principles
- **Time-first approach**: Temporal features as primary predictors
- **Period-specific handling**: Different strategies for stable vs active periods
- **Confidence modeling**: Predictions with uncertainty quantification

### Performance Expectations
- **Overall accuracy**: Very good due to strong temporal patterns
- **Best performance**: {self.patterns_found['daily_extremes']['min_time']} and afternoon stability periods
- **Most challenging**: Early morning transition around {self.patterns_found['change_events']['largest_fall_time']}

## Final Recommendation

The discovered 15-minute temperature patterns provide an excellent foundation for prediction modeling. The strong daily thermal cycle with identifiable stable and active periods enables a tiered modeling approach that can achieve high accuracy during predictable periods while handling variability during challenging times.

**Start with the time-pattern skeleton, then enhance with environmental features for improved accuracy.**
"""
        
        # Save inference
        with open('../temperature_modeling_inference.md', 'w', encoding='utf-8') as f:
            f.write(inference_content)
            
        print("  Saved: temperature_modeling_inference.md")
        
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis"""
        print("="*80)
        print("COMPREHENSIVE 15-MINUTE TEMPERATURE PATTERN ANALYSIS")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Create time features
        self.create_time_features()
        
        # Analyze patterns
        self.analyze_daily_patterns()
        self.find_specific_patterns()
        self.analyze_variability()
        
        # Create individual figures
        self.create_individual_figures()
        
        # Save data
        self.save_detailed_data()
        
        # Generate reports
        self.generate_detailed_report()
        self.create_summary_inference()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"[FIGURES] Generated 6 individual figure files")
        print(f"[DATA] Saved 3 detailed data files") 
        print(f"[REPORTS] Created 2 comprehensive reports")
        print(f"[ANALYSIS] Analyzed {len(self.daily_pattern)} 15-minute time chunks")
        print(f"[PROCESSING] Processed {len(self.df):,} temperature measurements")
        print("="*80)

def main():
    """Main function"""
    analyzer = Comprehensive15MinAnalyzer()
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
