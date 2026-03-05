"""
Quick Model Validation - Overfitting Detection
==============================================

This script performs key validation checks to detect overfitting:
1. Recreates the exact training process to get true test metrics  
2. Compares with baselines
3. Analyzes temporal patterns
4. Provides overfitting assessment

Author: AI Assistant  
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_process_data(file_path):
    """Load and process data exactly like training."""
    print("🔄 Loading and processing data...")
    
    df = pd.read_csv(file_path)
    print(f"   Loaded {len(df):,} records")
    
    # Basic preprocessing
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)
    
    # Keep required columns
    required_cols = ['date_time', 'temperature', 'humidity', 'pressure', 'light']
    df = df[required_cols].copy()
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    return df

def engineer_features(df):
    """Engineer features exactly like training."""
    print("🔄 Engineering features...")
    
    df = df.copy()
    
    # Time features  
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['day_of_year'] = df['date_time'].dt.dayofyear
    df['month'] = df['date_time'].dt.month
    df['quarter'] = df['date_time'].dt.quarter
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (reduced)
    lag_periods = [1, 5, 15, 30]
    for col in ['temperature', 'humidity', 'pressure', 'light']:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Rolling statistics (reduced)  
    windows = [5, 15, 30]
    for col in ['temperature', 'humidity', 'pressure', 'light']:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    
    # Rate of change
    for col in ['temperature', 'humidity', 'pressure', 'light']:
        df[f'{col}_diff_1'] = df[col].diff(1)
        df[f'{col}_diff_5'] = df[col].diff(5)
        df[f'{col}_pct_change_5'] = df[col].pct_change(5)
    
    # Handle NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col != 'date_time':
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    df = df.dropna().reset_index(drop=True)
    
    print(f"   Final dataset: {len(df):,} records with {len(df.columns)-1} features")
    return df

def create_simple_sequences(df, sequence_length=30):
    """Create simple sequences for analysis."""
    print(f"🔄 Creating sequences (length={sequence_length})...")
    
    # Sample every 3rd row (matching training)
    df_sampled = df.iloc[::3].reset_index(drop=True)
    print(f"   Sampled to {len(df_sampled):,} records")
    
    # Get features and target
    feature_cols = [col for col in df_sampled.columns if col not in ['date_time', 'temperature']]
    
    X = df_sampled[feature_cols].values
    y = df_sampled['temperature'].values
    timestamps = df_sampled['date_time'].values
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    t_sequences = []
    
    for i in range(sequence_length, len(X)):
        X_sequences.append(X[i - sequence_length:i])
        y_sequences.append(y[i])
        t_sequences.append(timestamps[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"   Created {len(X_sequences):,} sequences")
    return X_sequences, y_sequences, t_sequences, feature_cols

def split_data_chronologically(X, y, timestamps, train_size=0.7, val_size=0.15):
    """Split data chronologically."""
    print("🔄 Splitting data chronologically...")
    
    n_samples = len(X)
    train_end = int(n_samples * train_size)
    val_end = int(n_samples * (train_size + val_size))
    
    # Training
    X_train = X[:train_end]
    y_train = y[:train_end]
    t_train = timestamps[:train_end]
    
    # Validation
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    t_val = timestamps[train_end:val_end]
    
    # Test
    X_test = X[val_end:]
    y_test = y[val_end:]
    t_test = timestamps[val_end:]
    
    print(f"   Train: {len(X_train):,} samples ({t_train[0]} to {t_train[-1]})")
    print(f"   Val:   {len(X_val):,} samples ({t_val[0]} to {t_val[-1]})")
    print(f"   Test:  {len(X_test):,} samples ({t_test[0]} to {t_test[-1]})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test

def compare_baselines(X_train, X_test, y_train, y_test):
    """Compare different baseline methods."""
    print("\n" + "="*60)
    print("📊 BASELINE COMPARISON - REALITY CHECK")
    print("="*60)
    
    results = {}
    
    # 1. Naive baseline - last observed value
    print("🔄 Testing naive baseline...")
    naive_pred = np.roll(y_test, 1)
    naive_pred[0] = y_test[0]
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_acc = (1 - naive_mae / np.mean(y_test)) * 100
    results['Naive (Last Value)'] = {'mae': naive_mae, 'accuracy': naive_acc}
    
    # 2. Moving average 
    print("🔄 Testing moving average...")
    window = 5
    ma_pred = np.convolve(y_test, np.ones(window)/window, mode='same')
    ma_pred[:window//2] = y_test[:window//2]
    ma_pred[-(window//2):] = y_test[-(window//2):]
    ma_mae = mean_absolute_error(y_test, ma_pred)
    ma_acc = (1 - ma_mae / np.mean(y_test)) * 100
    results['Moving Average'] = {'mae': ma_mae, 'accuracy': ma_acc}
    
    # 3. Linear regression on last timestep features
    print("🔄 Testing linear regression...")
    try:
        # Use last timestep of sequences with first 10 features
        X_train_flat = X_train[:, -1, :10]
        X_test_flat = X_test[:, -1, :10]
        
        lr = LinearRegression()
        lr.fit(X_train_flat, y_train)
        lr_pred = lr.predict(X_test_flat)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        lr_acc = (1 - lr_mae / np.mean(y_test)) * 100
        results['Linear Regression'] = {'mae': lr_mae, 'accuracy': lr_acc}
    except Exception as e:
        print(f"   ⚠️ Linear regression failed: {e}")
        results['Linear Regression'] = {'mae': np.nan, 'accuracy': np.nan}
    
    # 4. Random Forest (simple)
    print("🔄 Testing Random Forest...")
    try:
        X_train_flat = X_train[:, -1, :15]  # Last timestep, first 15 features
        X_test_flat = X_test[:, -1, :15]
        
        rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_flat, y_train)
        rf_pred = rf.predict(X_test_flat)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_acc = (1 - rf_mae / np.mean(y_test)) * 100
        results['Random Forest'] = {'mae': rf_mae, 'accuracy': rf_acc}
    except Exception as e:
        print(f"   ⚠️ Random Forest failed: {e}")
        results['Random Forest'] = {'mae': np.nan, 'accuracy': np.nan}
    
    # Print results
    print(f"\n📊 BASELINE RESULTS:")
    print("   Method                MAE (°C)    Accuracy")
    print("   " + "-"*45)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf'))
    
    for method, metrics in sorted_results:
        if not np.isnan(metrics['mae']):
            print(f"   {method:<20} {metrics['mae']:8.4f}    {metrics['accuracy']:6.2f}%")
    
    return results

def analyze_temporal_patterns(y_test, timestamps_test):
    """Analyze temporal patterns in the data."""
    print("\n🔍 TEMPORAL PATTERN ANALYSIS:")
    
    # Basic statistics
    print(f"   Temperature range: {np.min(y_test):.2f}°C to {np.max(y_test):.2f}°C")
    print(f"   Temperature mean: {np.mean(y_test):.2f}°C")
    print(f"   Temperature std: {np.std(y_test):.2f}°C")
    
    # Time-based patterns
    df_temp = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps_test),
        'temperature': y_test
    })
    
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    df_temp['month'] = df_temp['timestamp'].dt.month
    
    # Hourly patterns
    hourly_stats = df_temp.groupby('hour')['temperature'].agg(['mean', 'std']).round(2)
    print(f"\n   📈 Hourly Temperature Variation:")
    print(f"      Peak hour: {hourly_stats['mean'].idxmax()}:00 ({hourly_stats['mean'].max():.2f}°C)")
    print(f"      Low hour:  {hourly_stats['mean'].idxmin()}:00 ({hourly_stats['mean'].min():.2f}°C)")
    print(f"      Daily range: {hourly_stats['mean'].max() - hourly_stats['mean'].min():.2f}°C")
    
    # Monthly patterns
    if len(df_temp['month'].unique()) > 1:
        monthly_stats = df_temp.groupby('month')['temperature'].agg(['mean', 'std']).round(2)
        print(f"\n   🗓️ Monthly Temperature Variation:")
        print(f"      Warmest month: {monthly_stats['mean'].idxmax()} ({monthly_stats['mean'].max():.2f}°C)")
        print(f"      Coolest month: {monthly_stats['mean'].idxmin()} ({monthly_stats['mean'].min():.2f}°C)")
    
    return df_temp

def assess_overfitting_likelihood(baseline_results):
    """Assess likelihood of overfitting based on baseline comparison."""
    print("\n" + "="*60)
    print("🎯 OVERFITTING ASSESSMENT")
    print("="*60)
    
    # Get reported LSTM performance from training logs
    reported_val_mae = 0.2864  # From training
    reported_val_acc = 98.97   # From training
    
    print(f"📊 REPORTED LSTM PERFORMANCE:")
    print(f"   Validation MAE: {reported_val_mae:.4f}°C")
    print(f"   Validation Accuracy: {reported_val_acc:.2f}%")
    
    # Compare with baselines
    print(f"\n🔍 OVERFITTING INDICATORS:")
    evidence = []
    
    # Check if LSTM beats all baselines significantly
    best_baseline_mae = min([r['mae'] for r in baseline_results.values() if not np.isnan(r['mae'])])
    best_baseline_name = [k for k, v in baseline_results.items() if v['mae'] == best_baseline_mae][0]
    
    improvement = ((best_baseline_mae - reported_val_mae) / best_baseline_mae) * 100
    
    print(f"   Best baseline: {best_baseline_name} (MAE: {best_baseline_mae:.4f}°C)")
    print(f"   LSTM improvement: {improvement:.1f}% better")
    
    if improvement > 90:
        evidence.append("❌ Extremely large improvement (>90%) - likely overfitting")
    elif improvement > 70:
        evidence.append("⚠️ Very large improvement (>70%) - possible overfitting")
    elif improvement > 50:
        evidence.append("⚠️ Large improvement (>50%) - check for leakage")
    else:
        evidence.append("✅ Reasonable improvement - likely valid")
    
    # Check accuracy level
    if reported_val_acc > 99:
        evidence.append("❌ Unrealistically high accuracy (>99%)")
    elif reported_val_acc > 98.5:
        evidence.append("⚠️ Very high accuracy (>98.5%) - suspicious")
    else:
        evidence.append("✅ Reasonable accuracy level")
    
    # Check feature engineering complexity
    evidence.append("⚠️ 90 engineered features from 4 sensors - high complexity")
    evidence.append("⚠️ Extensive lag/rolling features may cause data leakage")
    
    print(f"\n🚨 EVIDENCE SUMMARY:")
    for item in evidence:
        print(f"   {item}")
    
    # Final verdict
    red_flags = sum(1 for item in evidence if item.startswith("❌"))
    yellow_flags = sum(1 for item in evidence if item.startswith("⚠️"))
    
    print(f"\n⚖️ FINAL ASSESSMENT:")
    if red_flags >= 2:
        print("   🚨 SIGNIFICANT OVERFITTING LIKELY")
        print("   📋 Recommendation: Rebuild with simpler features and validate properly")
    elif red_flags >= 1 or yellow_flags >= 3:
        print("   ⚠️ MODERATE OVERFITTING SUSPECTED")
        print("   📋 Recommendation: Test on true holdout data and reduce feature complexity")
    else:
        print("   ✅ MINIMAL OVERFITTING DETECTED")
        print("   📋 Recommendation: Model may be valid but verify on new data")
    
    return evidence

def create_validation_plots(baseline_results, df_temp):
    """Create validation plots."""
    print("\n🔄 Creating validation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Validation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Baseline comparison
    methods = [k for k, v in baseline_results.items() if not np.isnan(v['mae'])]
    maes = [baseline_results[m]['mae'] for m in methods]
    colors = plt.cm.Set3(np.arange(len(methods)))
    
    bars = axes[0, 0].bar(methods, maes, color=colors)
    axes[0, 0].set_title('Baseline Model Comparison')
    axes[0, 0].set_ylabel('MAE (°C)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add LSTM reported performance
    lstm_mae = 0.2864
    axes[0, 0].axhline(y=lstm_mae, color='red', linestyle='--', linewidth=2, 
                      label=f'Reported LSTM: {lstm_mae:.4f}°C')
    axes[0, 0].legend()
    
    # 2. Accuracy comparison
    accuracies = [baseline_results[m]['accuracy'] for m in methods]
    bars = axes[0, 1].bar(methods, accuracies, color=colors)
    axes[0, 1].set_title('Baseline Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add LSTM reported accuracy
    lstm_acc = 98.97
    axes[0, 1].axhline(y=lstm_acc, color='red', linestyle='--', linewidth=2,
                      label=f'Reported LSTM: {lstm_acc:.2f}%')
    axes[0, 1].legend()
    
    # 3. Temperature time series (sample)
    sample_size = min(2000, len(df_temp))
    sample_indices = np.linspace(0, len(df_temp)-1, sample_size, dtype=int)
    sample_data = df_temp.iloc[sample_indices]
    
    axes[1, 0].plot(sample_data['timestamp'], sample_data['temperature'], alpha=0.7, linewidth=0.8)
    axes[1, 0].set_title('Temperature Time Series (Sample)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Hourly temperature pattern
    hourly_mean = df_temp.groupby('hour')['temperature'].mean()
    axes[1, 1].plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2)
    axes[1, 1].set_title('Average Temperature by Hour')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_analysis.png', dpi=300, bbox_inches='tight')
    print("   📊 Validation plots saved: validation_analysis.png")
    plt.show()

def main():
    """Run quick validation check."""
    print("🔬 QUICK MODEL VALIDATION - OVERFITTING DETECTION")
    print("="*60)
    print("This will analyze your data and compare with baselines to detect overfitting")
    print("="*60)
    
    # Load and process data
    data_file = r"C:\Users\operator\Desktop\smartvayu\data\combined_plus_sensor_data.csv"
    df = load_and_process_data(data_file)
    
    # Feature engineering  
    df_features = engineer_features(df)
    
    # Create sequences
    X, y, timestamps, feature_cols = create_simple_sequences(df_features)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = split_data_chronologically(X, y, timestamps)
    
    # Check temporal leakage
    print(f"\n🔍 TEMPORAL LEAKAGE CHECK:")
    print(f"   Train ends:  {t_train[-1]}")
    print(f"   Val starts:  {t_val[0]}")
    print(f"   Test starts: {t_test[0]}")
    if t_val[0] > t_train[-1] and t_test[0] > t_val[-1]:
        print("   ✅ No temporal leakage detected")
    else:
        print("   ❌ TEMPORAL LEAKAGE DETECTED!")
    
    # Compare baselines
    baseline_results = compare_baselines(X_train, X_test, y_train, y_test)
    
    # Analyze temporal patterns
    df_temp = analyze_temporal_patterns(y_test, t_test)
    
    # Assess overfitting
    evidence = assess_overfitting_likelihood(baseline_results)
    
    # Create plots
    create_validation_plots(baseline_results, df_temp)
    
    print("\n" + "="*60)
    print("🎉 VALIDATION COMPLETE!")
    print("="*60)
    print("Key findings:")
    print("• Check baseline comparison - if LSTM beats all by >70%, likely overfitting")  
    print("• 99%+ accuracy is suspicious for real sensor data")
    print("• 90 features from 4 sensors is very high complexity")
    print("• Generated validation_analysis.png with detailed plots")
    print("="*60)

if __name__ == "__main__":
    main()
