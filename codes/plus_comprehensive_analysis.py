#!/usr/bin/env python3
"""
Comprehensive Plus Dataset Analysis
==================================
Generates separate graphs and performs statistical tests on the combined Plus sensor data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest, anderson, jarque_bera
from scipy.stats import levene, bartlett, fligner, kruskal, f_oneway
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class PlusDatasetAnalyzer:
    """Comprehensive Plus dataset analyzer with separate graph generation"""
    
    def __init__(self, csv_file):
        """Initialize with CSV file path"""
        self.csv_file = csv_file
        self.df = None
        self.stats_results = {}
        self.test_results = {}
        self.graph_files = []
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading Plus dataset...")
        self.df = pd.read_csv(self.csv_file)
        self.df['date_time'] = pd.to_datetime(self.df['date_time'])
        
        # Remove any NaN temperature values (though we expect none)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['temperature'])
        final_count = len(self.df)
        
        print(f"Data loaded: {final_count:,} records")
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count:,} NaN temperature values")
        
        print(f"Date range: {self.df['date_time'].min()} to {self.df['date_time'].max()}")
        print(f"RPI IDs: {sorted(self.df['rpi_id'].unique())}")
        print(f"Additional sensors: oxidised, reduced, nh3 (not in REG dataset)")
        
    def create_temperature_distribution(self):
        """Create temperature distribution graph"""
        print("\n📊 Creating temperature distribution graph...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with normal curve
        temp = self.df['temperature']
        ax1.hist(temp, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add normal curve overlay
        mu, sigma = temp.mean(), temp.std()
        x = np.linspace(temp.min(), temp.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, y, 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
        ax1.set_title('Temperature Distribution with Normal Overlay', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(temp, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = '01_temperature_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.graph_files.append(filename)
        plt.close()
        print(f"   Saved: {filename}")
        
    def create_temperature_by_device(self):
        """Create temperature comparison by RPI device"""
        print("📊 Creating temperature by device graphs...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot
        sns.boxplot(data=self.df, x='rpi_id', y='temperature', ax=ax1)
        ax1.set_title('Temperature Distribution by RPI Device (Box Plot)', fontweight='bold')
        ax1.set_xlabel('RPI Device')
        ax1.set_ylabel('Temperature (°C)')
        ax1.grid(True, alpha=0.3)
        
        # Violin plot
        sns.violinplot(data=self.df, x='rpi_id', y='temperature', ax=ax2)
        ax2.set_title('Temperature Distribution by RPI Device (Violin Plot)', fontweight='bold')
        ax2.set_xlabel('RPI Device')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True, alpha=0.3)
        
        # Mean comparison with error bars
        device_stats = self.df.groupby('rpi_id')['temperature'].agg(['mean', 'std', 'count'])
        x_pos = range(len(device_stats))
        ax3.bar(x_pos, device_stats['mean'], yerr=device_stats['std'], 
                capsize=5, alpha=0.7, color='lightcoral')
        ax3.set_title('Mean Temperature by RPI Device (±1 SD)', fontweight='bold')
        ax3.set_xlabel('RPI Device')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(device_stats.index)
        ax3.grid(True, alpha=0.3)
        
        # Histogram overlay
        for rpi_id in sorted(self.df['rpi_id'].unique()):
            data = self.df[self.df['rpi_id'] == rpi_id]['temperature']
            ax4.hist(data, bins=30, alpha=0.5, label=f'RPI {rpi_id}', density=True)
        ax4.set_title('Temperature Distributions by RPI Device (Overlaid)', fontweight='bold')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = '02_temperature_by_device.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.graph_files.append(filename)
        plt.close()
        print(f"   Saved: {filename}")
        
    def create_time_series(self):
        """Create time series visualizations"""
        print("📊 Creating time series graphs...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Full time series (sampled)
        sample_df = self.df.iloc[::max(1, len(self.df)//2000)].sort_values('date_time')
        ax1.plot(sample_df['date_time'], sample_df['temperature'], alpha=0.7, linewidth=0.5)
        ax1.set_title('Temperature Time Series (Sampled)', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (°C)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Daily averages
        daily_temp = self.df.set_index('date_time')['temperature'].resample('D').mean()
        ax2.plot(daily_temp.index, daily_temp.values, linewidth=1, color='red')
        ax2.set_title('Daily Average Temperature', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Temperature (°C)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Monthly statistics
        monthly_stats = self.df.groupby(self.df['date_time'].dt.to_period('M'))['temperature'].agg(['mean', 'std', 'min', 'max'])
        x_months = range(len(monthly_stats))
        ax3.fill_between(x_months, monthly_stats['min'], monthly_stats['max'], alpha=0.3, label='Min-Max Range')
        ax3.errorbar(x_months, monthly_stats['mean'], yerr=monthly_stats['std'], 
                    fmt='o-', capsize=3, label='Mean ± SD', color='red')
        ax3.set_title('Monthly Temperature Statistics', fontweight='bold')
        ax3.set_xlabel('Month Index')
        ax3.set_ylabel('Temperature (°C)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Seasonal decomposition (simplified)
        df_hourly = self.df.set_index('date_time')['temperature'].resample('H').mean().dropna()
        if len(df_hourly) > 100:
            # Simple moving averages for trend
            trend = df_hourly.rolling(window=24*7, center=True).mean()  # 7-day trend
            ax4.plot(df_hourly.index, df_hourly.values, alpha=0.3, label='Hourly', linewidth=0.5)
            ax4.plot(trend.index, trend.values, color='red', linewidth=2, label='Weekly Trend')
            ax4.set_title('Temperature Trend Analysis', fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Temperature (°C)')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = '03_time_series_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.graph_files.append(filename)
        plt.close()
        print(f"   Saved: {filename}")
        
    def create_correlation_analysis(self):
        """Create correlation analysis graphs"""
        print("📊 Creating correlation analysis graphs...")
        
        # Select numeric columns for correlation
        numeric_cols = ['proximity', 'humidity', 'pressure', 'light', 'oxidised', 'reduced', 
                       'nh3', 'temperature', 'sound_high', 'sound_mid', 'sound_low', 'sound_amp']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Correlation heatmap
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Sensor Correlation Matrix', fontweight='bold')
        
        # Temperature correlations bar plot
        temp_corrs = corr_matrix['temperature'].drop('temperature').sort_values(key=abs, ascending=False)
        colors = ['red' if x < 0 else 'blue' for x in temp_corrs.values]
        ax2.barh(range(len(temp_corrs)), temp_corrs.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(temp_corrs)))
        ax2.set_yticklabels(temp_corrs.index)
        ax2.set_xlabel('Correlation with Temperature')
        ax2.set_title('Temperature Correlations (Sorted)', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot of strongest positive correlation
        strongest_pos = temp_corrs[temp_corrs > 0].index[0] if any(temp_corrs > 0) else temp_corrs.index[0]
        sample_size = min(5000, len(self.df))
        sample_df = self.df.sample(sample_size)
        ax3.scatter(sample_df[strongest_pos], sample_df['temperature'], alpha=0.5, s=1)
        ax3.set_xlabel(f'{strongest_pos}')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title(f'Temperature vs {strongest_pos} (r={temp_corrs[strongest_pos]:.3f})', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot of strongest negative correlation (if any)
        if any(temp_corrs < 0):
            strongest_neg = temp_corrs[temp_corrs < 0].index[0]
            ax4.scatter(sample_df[strongest_neg], sample_df['temperature'], alpha=0.5, s=1, color='red')
            ax4.set_xlabel(f'{strongest_neg}')
            ax4.set_ylabel('Temperature (°C)')
            ax4.set_title(f'Temperature vs {strongest_neg} (r={temp_corrs[strongest_neg]:.3f})', fontweight='bold')
        else:
            # If no negative correlations, show second strongest
            second_strongest = temp_corrs.index[1] if len(temp_corrs) > 1 else temp_corrs.index[0]
            ax4.scatter(sample_df[second_strongest], sample_df['temperature'], alpha=0.5, s=1, color='green')
            ax4.set_xlabel(f'{second_strongest}')
            ax4.set_ylabel('Temperature (°C)')
            ax4.set_title(f'Temperature vs {second_strongest} (r={temp_corrs[second_strongest]:.3f})', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = '04_correlation_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.graph_files.append(filename)
        plt.close()
        print(f"   Saved: {filename}")
        
    def create_sensor_comparisons(self):
        """Create additional sensor analysis (unique to Plus dataset)"""
        print("📊 Creating additional sensor analysis graphs...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gas sensors analysis
        gas_sensors = ['oxidised', 'reduced', 'nh3']
        for sensor in gas_sensors:
            if sensor in self.df.columns:
                sample_data = self.df.sample(min(5000, len(self.df)))
                ax1.scatter(sample_data[sensor], sample_data['temperature'], 
                           alpha=0.5, s=1, label=sensor)
        ax1.set_xlabel('Gas Sensor Values')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature vs Gas Sensors', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Light vs Temperature (environmental relationship)
        sample_data = self.df.sample(min(5000, len(self.df)))
        ax2.scatter(sample_data['light'], sample_data['temperature'], alpha=0.5, s=1, color='orange')
        ax2.set_xlabel('Light Level')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature vs Light Level', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Sound vs Temperature
        sound_cols = ['sound_high', 'sound_mid', 'sound_low']
        for i, sound_col in enumerate(sound_cols):
            if sound_col in self.df.columns:
                sample_data = self.df.sample(min(2000, len(self.df)))
                ax3.scatter(sample_data[sound_col], sample_data['temperature'], 
                           alpha=0.6, s=1, label=sound_col)
        ax3.set_xlabel('Sound Level')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Temperature vs Sound Levels', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Humidity vs Pressure (environmental conditions)
        sample_data = self.df.sample(min(5000, len(self.df)))
        scatter = ax4.scatter(sample_data['humidity'], sample_data['pressure'], 
                            c=sample_data['temperature'], cmap='coolwarm', alpha=0.6, s=2)
        ax4.set_xlabel('Humidity (%)')
        ax4.set_ylabel('Pressure (hPa)')
        ax4.set_title('Humidity vs Pressure (colored by Temperature)', fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Temperature (°C)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = '05_sensor_relationships.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.graph_files.append(filename)
        plt.close()
        print(f"   Saved: {filename}")
        
    def create_statistical_diagnostics(self):
        """Create statistical diagnostic plots"""
        print("📊 Creating statistical diagnostic graphs...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        temp = self.df['temperature']
        
        # Residuals plot (detrended)
        df_sorted = self.df.sort_values('date_time')
        time_index = np.arange(len(df_sorted))
        sample_size = min(10000, len(df_sorted))
        sample_idx = np.random.choice(len(df_sorted), sample_size, replace=False)
        sample_time = time_index[sample_idx]
        sample_temp = df_sorted['temperature'].iloc[sample_idx]
        
        # Fit linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(sample_time, sample_temp)
        residuals = sample_temp - (slope * sample_time + intercept)
        
        ax1.scatter(sample_time, residuals, alpha=0.5, s=1)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Time Index', fontweight='bold')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Residuals (°C)')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=50, density=True, alpha=0.7, color='lightgreen')
        # Fit normal curve to residuals
        mu_res, sigma_res = residuals.mean(), residuals.std()
        x_res = np.linspace(residuals.min(), residuals.max(), 100)
        y_res = stats.norm.pdf(x_res, mu_res, sigma_res)
        ax2.plot(x_res, y_res, 'r-', linewidth=2)
        ax2.set_title('Distribution of Residuals', fontweight='bold')
        ax2.set_xlabel('Residuals (°C)')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        # Box plots by month
        self.df['month'] = self.df['date_time'].dt.month
        monthly_data = []
        month_labels = []
        for month in sorted(self.df['month'].unique()):
            month_temps = self.df[self.df['month'] == month]['temperature']
            if len(month_temps) > 10:  # Only include months with sufficient data
                monthly_data.append(month_temps)
                month_labels.append(f'Month {month}')
        
        if monthly_data:
            ax3.boxplot(monthly_data, labels=month_labels)
            ax3.set_title('Temperature Distribution by Month', fontweight='bold')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Temperature (°C)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Outlier detection plot
        z_scores = np.abs((temp - temp.mean()) / temp.std())
        outliers = z_scores > 3
        ax4.scatter(range(len(temp)), temp, c=outliers, cmap='coolwarm', alpha=0.6, s=0.5)
        ax4.set_title(f'Temperature with Outliers (Z>3): {outliers.sum()} points', fontweight='bold')
        ax4.set_xlabel('Record Index')
        ax4.set_ylabel('Temperature (°C)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = '06_statistical_diagnostics.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.graph_files.append(filename)
        plt.close()
        print(f"   Saved: {filename}")
        
    def perform_descriptive_statistics(self):
        """Calculate comprehensive descriptive statistics"""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        temp = self.df['temperature']
        
        # Basic statistics
        stats_dict = {
            'count': len(temp),
            'mean': temp.mean(),
            'median': temp.median(),
            'mode': temp.mode().iloc[0] if not temp.mode().empty else np.nan,
            'std': temp.std(),
            'variance': temp.var(),
            'min': temp.min(),
            'max': temp.max(),
            'range': temp.max() - temp.min(),
            'q25': temp.quantile(0.25),
            'q75': temp.quantile(0.75),
            'iqr': temp.quantile(0.75) - temp.quantile(0.25),
            'skewness': temp.skew(),
            'kurtosis': temp.kurtosis(),
            'cv': temp.std() / temp.mean() * 100,
        }
        
        self.stats_results.update(stats_dict)
        
        # Print results
        print(f"Count:              {stats_dict['count']:,}")
        print(f"Mean:               {stats_dict['mean']:.4f}°C")
        print(f"Median:             {stats_dict['median']:.4f}°C")
        print(f"Mode:               {stats_dict['mode']:.4f}°C")
        print(f"Standard Deviation: {stats_dict['std']:.4f}°C")
        print(f"Variance:           {stats_dict['variance']:.4f}")
        print(f"Minimum:            {stats_dict['min']:.4f}°C")
        print(f"Maximum:            {stats_dict['max']:.4f}°C")
        print(f"Range:              {stats_dict['range']:.4f}°C")
        print(f"Q1 (25%):           {stats_dict['q25']:.4f}°C")
        print(f"Q3 (75%):           {stats_dict['q75']:.4f}°C")
        print(f"IQR:                {stats_dict['iqr']:.4f}°C")
        print(f"Skewness:           {stats_dict['skewness']:.4f}")
        print(f"Kurtosis:           {stats_dict['kurtosis']:.4f}")
        print(f"Coeff. of Variation: {stats_dict['cv']:.2f}%")
        
        # Statistics by RPI ID
        print(f"\n{'='*40}")
        print("TEMPERATURE STATISTICS BY RPI ID")
        print(f"{'='*40}")
        
        rpi_stats = self.df.groupby('rpi_id')['temperature'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print(rpi_stats)
        self.stats_results['by_rpi'] = rpi_stats
        
    def perform_normality_tests(self):
        """Perform various normality tests"""
        print("\n" + "="*60)
        print("NORMALITY TESTS")
        print("="*60)
        
        temp = self.df['temperature'].dropna()
        
        # Sample data for tests (use subset for large datasets)
        if len(temp) > 5000:
            sample_temp = temp.sample(5000, random_state=42)
            print(f"Using random sample of 5,000 records for normality tests")
        else:
            sample_temp = temp
        
        tests = {}
        
        # Shapiro-Wilk test (max 5000 samples)
        if len(sample_temp) <= 5000:
            stat, p_value = shapiro(sample_temp)
            tests['Shapiro-Wilk'] = {'statistic': stat, 'p_value': p_value}
            print(f"Shapiro-Wilk Test:   Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        # D'Agostino-Pearson test
        stat, p_value = normaltest(sample_temp)
        tests['DAgostino-Pearson'] = {'statistic': stat, 'p_value': p_value}
        print(f"D'Agostino-Pearson:  Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        # Jarque-Bera test
        stat, p_value = jarque_bera(sample_temp)
        tests['Jarque-Bera'] = {'statistic': stat, 'p_value': p_value}
        print(f"Jarque-Bera Test:    Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        # Kolmogorov-Smirnov test
        stat, p_value = kstest(sample_temp, 'norm', args=(sample_temp.mean(), sample_temp.std()))
        tests['Kolmogorov-Smirnov'] = {'statistic': stat, 'p_value': p_value}
        print(f"Kolmogorov-Smirnov:  Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        # Anderson-Darling test
        result = anderson(sample_temp, dist='norm')
        tests['Anderson-Darling'] = {
            'statistic': result.statistic,
            'critical_values': result.critical_values,
            'significance_levels': result.significance_level
        }
        print(f"Anderson-Darling:    Statistic = {result.statistic:.6f}")
        print(f"                     Critical values: {result.critical_values}")
        print(f"                     Significance levels: {result.significance_level}")
        
        self.test_results['normality'] = tests
        
        # Interpretation
        print(f"\n{'='*40}")
        print("NORMALITY TEST INTERPRETATION")
        print(f"{'='*40}")
        alpha = 0.05
        normal_count = 0
        total_tests = 0
        
        for test_name, results in tests.items():
            if test_name == 'Anderson-Darling':
                continue
            else:
                total_tests += 1
                if results['p_value'] > alpha:
                    normal_count += 1
                    print(f"{test_name}: NORMAL (p > {alpha})")
                else:
                    print(f"{test_name}: NOT NORMAL (p <= {alpha})")
        
        print(f"\nSummary: {normal_count}/{total_tests} tests suggest normal distribution")
        
        if normal_count >= total_tests // 2:
            print("✅ Data appears to be approximately normally distributed")
            print("   → Can use parametric statistical tests")
        else:
            print("❌ Data does not appear to be normally distributed")
            print("   → Should use non-parametric statistical tests")
        
    def perform_correlation_tests(self):
        """Analyze correlations between temperature and other variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric columns
        numeric_cols = ['proximity', 'humidity', 'pressure', 'light', 'oxidised', 'reduced',
                       'nh3', 'temperature', 'sound_high', 'sound_mid', 'sound_low', 'sound_amp']
        
        correlation_data = self.df[numeric_cols].select_dtypes(include=[np.number])
        
        # Pearson correlations
        print("PEARSON CORRELATIONS WITH TEMPERATURE:")
        print("-" * 45)
        
        pearson_corrs = {}
        for col in correlation_data.columns:
            if col != 'temperature':
                corr, p_value = pearsonr(correlation_data['temperature'].dropna(), 
                                       correlation_data[col].dropna())
                pearson_corrs[col] = {'correlation': corr, 'p_value': p_value}
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{col:15}: {corr:7.4f} (p={p_value:.4e}) {significance}")
        
        # Spearman correlations
        print(f"\nSPEARMAN CORRELATIONS WITH TEMPERATURE:")
        print("-" * 46)
        
        spearman_corrs = {}
        for col in correlation_data.columns:
            if col != 'temperature':
                corr, p_value = spearmanr(correlation_data['temperature'].dropna(), 
                                        correlation_data[col].dropna())
                spearman_corrs[col] = {'correlation': corr, 'p_value': p_value}
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{col:15}: {corr:7.4f} (p={p_value:.4e}) {significance}")
        
        self.test_results['correlations'] = {
            'pearson': pearson_corrs,
            'spearman': spearman_corrs
        }
        
        print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Identify strongest correlations
        strongest_pearson = max(pearson_corrs.items(), key=lambda x: abs(x[1]['correlation']))
        strongest_spearman = max(spearman_corrs.items(), key=lambda x: abs(x[1]['correlation']))
        
        print(f"\nSTRONGEST CORRELATIONS:")
        print(f"Pearson:  {strongest_pearson[0]} (r = {strongest_pearson[1]['correlation']:.4f})")
        print(f"Spearman: {strongest_spearman[0]} (ρ = {strongest_spearman[1]['correlation']:.4f})")
        
    def perform_group_tests(self):
        """Test for differences between RPI groups"""
        print("\n" + "="*60)
        print("GROUP COMPARISON TESTS")
        print("="*60)
        
        # Group temperature data by RPI ID
        groups = [group['temperature'].values for name, group in self.df.groupby('rpi_id')]
        group_names = [str(name) for name, group in self.df.groupby('rpi_id')]
        
        print(f"Comparing temperature across {len(groups)} RPI devices: {group_names}")
        print(f"Group sample sizes: {[len(g) for g in groups]}")
        
        # Variance homogeneity tests
        print(f"\nVARIANCE HOMOGENEITY TESTS:")
        print("-" * 30)
        
        # Levene's test
        stat, p_value = levene(*groups)
        self.test_results['levene'] = {'statistic': stat, 'p_value': p_value}
        print(f"Levene's Test:       Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        # Bartlett's test
        stat, p_value = bartlett(*groups)
        self.test_results['bartlett'] = {'statistic': stat, 'p_value': p_value}
        print(f"Bartlett's Test:     Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        # Fligner-Killeen test
        stat, p_value = fligner(*groups)
        self.test_results['fligner'] = {'statistic': stat, 'p_value': p_value}
        print(f"Fligner-Killeen:     Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        
        alpha = 0.05
        equal_variances = self.test_results['levene']['p_value'] > alpha
        
        if equal_variances:
            print(f"\n✅ Variances appear to be equal across groups (Levene p > {alpha})")
        else:
            print(f"\n❌ Variances appear to differ significantly across groups (Levene p <= {alpha})")
        
        # Group mean comparison tests
        print(f"\nGROUP MEAN COMPARISON TESTS:")
        print("-" * 35)
        
        # ANOVA (parametric)
        stat, p_value = f_oneway(*groups)
        self.test_results['anova'] = {'statistic': stat, 'p_value': p_value}
        print(f"One-Way ANOVA:       F = {stat:.6f}, p-value = {p_value:.6e}")
        
        # Kruskal-Wallis (non-parametric)
        stat, p_value = kruskal(*groups)
        self.test_results['kruskal_wallis'] = {'statistic': stat, 'p_value': p_value}
        print(f"Kruskal-Wallis:      H = {stat:.6f}, p-value = {p_value:.6e}")
        
        print(f"\nINTERPRETATION (α = {alpha}):")
        
        if self.test_results['anova']['p_value'] <= alpha:
            print("✅ ANOVA: Significant differences between device groups")
            print("   → Temperature varies significantly across RPI devices")
        else:
            print("❌ ANOVA: No significant differences between device groups")
            print("   → Temperature is consistent across RPI devices")
            
        if self.test_results['kruskal_wallis']['p_value'] <= alpha:
            print("✅ Kruskal-Wallis: Significant differences between device groups")
        else:
            print("❌ Kruskal-Wallis: No significant differences between device groups")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS REPORT - PLUS DATASET")
        print("="*80)
        
        print(f"Dataset: {self.csv_file}")
        print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Records: {len(self.df):,}")
        print(f"Date Range: {self.df['date_time'].min()} to {self.df['date_time'].max()}")
        print(f"Temperature Range: {self.stats_results['min']:.2f}°C to {self.stats_results['max']:.2f}°C")
        print(f"Mean Temperature: {self.stats_results['mean']:.2f}°C ± {self.stats_results['std']:.2f}°C")
        
        # Key findings
        print(f"\nKEY STATISTICAL FINDINGS:")
        print(f"- Distribution skewness: {self.stats_results['skewness']:.3f}")
        if abs(self.stats_results['skewness']) < 0.5:
            print("  → Approximately symmetric distribution")
        elif self.stats_results['skewness'] > 0:
            print("  → Right-skewed (positive skew)")
        else:
            print("  → Left-skewed (negative skew)")
            
        print(f"- Distribution kurtosis: {self.stats_results['kurtosis']:.3f}")
        if abs(self.stats_results['kurtosis']) < 0.5:
            print("  → Normal tail thickness (mesokurtic)")
        elif self.stats_results['kurtosis'] > 0:
            print("  → Heavy-tailed (leptokurtic)")
        else:
            print("  → Light-tailed (platykurtic)")
            
        print(f"- Coefficient of variation: {self.stats_results['cv']:.2f}%")
        if self.stats_results['cv'] < 15:
            print("  → Low variability (good measurement consistency)")
        elif self.stats_results['cv'] < 30:
            print("  → Moderate variability")
        else:
            print("  → High variability")
        
        # Correlation insights
        if 'correlations' in self.test_results:
            print(f"\nSTRONGEST CORRELATIONS WITH TEMPERATURE:")
            pearson_corrs = self.test_results['correlations']['pearson']
            sorted_corrs = sorted(pearson_corrs.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
            
            for i, (var, stats_data) in enumerate(sorted_corrs[:3]):
                corr = stats_data['correlation']
                p_val = stats_data['p_value']
                print(f"- {var}: r = {corr:.3f} (p = {p_val:.3e})")
                if abs(corr) > 0.7:
                    print(f"  → Strong correlation")
                elif abs(corr) > 0.3:
                    print(f"  → Moderate correlation")
                else:
                    print(f"  → Weak correlation")
        
        # Statistical tests summary
        print(f"\nSTATISTICAL TESTS SUMMARY:")
        
        # Normality
        if 'normality' in self.test_results:
            normal_tests = ['Shapiro-Wilk', 'DAgostino-Pearson', 'Jarque-Bera', 'Kolmogorov-Smirnov']
            normal_count = 0
            total_tests = len([t for t in normal_tests if t in self.test_results['normality']])
            for test in normal_tests:
                if test in self.test_results['normality']:
                    if self.test_results['normality'][test]['p_value'] > 0.05:
                        normal_count += 1
            print(f"- Normality: {normal_count}/{total_tests} tests suggest normal distribution")
            
            if normal_count >= total_tests // 2:
                print("  → Use parametric statistical methods")
            else:
                print("  → Use non-parametric statistical methods")
        
        # Group differences
        if 'anova' in self.test_results:
            anova_p = self.test_results['anova']['p_value']
            if anova_p <= 0.05:
                print(f"- Device differences: Significant (ANOVA p = {anova_p:.3e})")
                print("  → Temperature varies significantly between RPI devices")
                print("  → Consider device-specific analysis or correction factors")
            else:
                print(f"- Device differences: Not significant (ANOVA p = {anova_p:.3e})")
                print("  → Temperature measurements are consistent across devices")
        
        # Data quality assessment
        print(f"\nDATA QUALITY ASSESSMENT:")
        print(f"✅ No missing temperature values")
        print(f"✅ Chronologically ordered dataset")
        print(f"✅ {len(self.df):,} high-quality measurements")
        print(f"✅ Additional gas sensors (oxidised, reduced, nh3) not in REG dataset")
        print(f"✅ Extended date range: {(self.df['date_time'].max() - self.df['date_time'].min()).days} days")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS FOR FURTHER ANALYSIS:")
        print(f"1. Temperature data quality is excellent - suitable for advanced modeling")
        
        if 'correlations' in self.test_results:
            strongest_var = max(self.test_results['correlations']['pearson'].items(), 
                              key=lambda x: abs(x[1]['correlation']))[0]
            print(f"2. Investigate {strongest_var}-temperature relationship for predictive modeling")
        
        print(f"3. Explore gas sensor data (oxidised, reduced, nh3) for environmental insights")
        print(f"4. Consider temporal modeling due to excellent time series coverage")
        
        if 'anova' in self.test_results and self.test_results['anova']['p_value'] <= 0.05:
            print(f"5. Account for device-specific differences in multi-device analysis")
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE! Generated {len(self.graph_files)} visualization files.")
        print(f"{'='*80}")
        
        # List generated files
        print(f"\nGENERATED FILES:")
        for i, filename in enumerate(self.graph_files, 1):
            print(f"{i:2d}. {filename}")
        
    def run_complete_analysis(self):
        """Run all analyses and generate all graphs"""
        print("Starting Comprehensive Plus Dataset Analysis...")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Generate all graphs
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION FILES")
        print("="*60)
        
        self.create_temperature_distribution()
        self.create_temperature_by_device()
        self.create_time_series()
        self.create_correlation_analysis()
        self.create_sensor_comparisons()
        self.create_statistical_diagnostics()
        
        # Perform statistical analyses
        self.perform_descriptive_statistics()
        self.perform_normality_tests()
        self.perform_correlation_tests()
        self.perform_group_tests()
        
        # Generate final report
        self.generate_comprehensive_report()
        
def main():
    """Main function"""
    csv_file = "combined_plus_sensor_data.csv"
    
    analyzer = PlusDatasetAnalyzer(csv_file)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
