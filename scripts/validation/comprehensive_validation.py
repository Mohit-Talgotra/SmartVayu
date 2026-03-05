"""
Comprehensive Model Validation & Overfitting Detection
=====================================================

This script performs extensive validation of the LSTM temperature prediction model:
1. Quick sanity checks (test set evaluation, residual analysis)
2. Deeper checks (autocorrelation, drift, seasonality)
3. Baseline comparisons
4. Overfitting detection

Author: AI Assistant
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep learning and ML imports
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

# Time series analysis
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats

import joblib
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ModelValidator:
    """
    Comprehensive validator for the LSTM temperature prediction model.
    Performs sanity checks, overfitting detection, and baseline comparisons.
    """
    
    def __init__(self, model_dir="temperature_lstm_model"):
        """Initialize validator with model artifacts."""
        self.model_dir = model_dir
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.config = None
        self.load_model_artifacts()
        
    def load_model_artifacts(self):
        """Load saved model and preprocessors."""
        print("🔄 Loading model artifacts...")
        
        try:
            # Load model with custom objects to handle TensorFlow compatibility
            custom_objects = {
                'mse': 'mean_squared_error',
                'mae': 'mean_absolute_error'
            }
            
            try:
                self.model = load_model(f"{self.model_dir}/lstm_model.h5", custom_objects=custom_objects)
            except:
                # Fallback: try loading without custom objects
                import keras.losses as losses
                import keras.metrics as metrics
                custom_objects = {
                    'mse': losses.MeanSquaredError(),
                    'mae': metrics.MeanAbsoluteError()
                }
                self.model = load_model(f"{self.model_dir}/lstm_model.h5", custom_objects=custom_objects)
            
            self.scaler_features = joblib.load(f"{self.model_dir}/scaler_features.pkl")
            self.scaler_target = joblib.load(f"{self.model_dir}/scaler_target.pkl")
            self.config = joblib.load(f"{self.model_dir}/model_config.pkl")
            print("   ✅ Model artifacts loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            print("   ⚠️  Attempting to rebuild model from config...")
            try:
                self.scaler_features = joblib.load(f"{self.model_dir}/scaler_features.pkl")
                self.scaler_target = joblib.load(f"{self.model_dir}/scaler_target.pkl")
                self.config = joblib.load(f"{self.model_dir}/model_config.pkl")
                self._rebuild_model_from_config()
                print("   ✅ Model rebuilt successfully")
            except Exception as e2:
                print(f"   ❌ Failed to rebuild model: {e2}")
                raise e2
    
    def _rebuild_model_from_config(self):
        """Rebuild LSTM model from configuration."""
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from keras.optimizers import Adam
        from keras.regularizers import l1_l2
        
        # Rebuild the same architecture as training
        model = Sequential([
            LSTM(64, return_sequences=True, 
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Try to load weights
        try:
            model.load_weights(f"{self.model_dir}/lstm_model.h5")
        except:
            print("   ⚠️  Could not load weights - model will be untrained")
        
        self.model = model
    
    def load_and_process_data(self, file_path):
        """Load and recreate the exact same data processing as training."""
        print("🔄 Loading and processing data...")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"   Loaded {len(df):,} records")
        
        # Basic preprocessing (matching training)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # Keep only required columns
        required_cols = ['date_time', 'temperature', 'humidity', 'pressure', 'light']
        df = df[required_cols].copy()
        
        # Drop NaN values
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def engineer_features(self, df):
        """Recreate exact feature engineering from training."""
        print("🔄 Engineering features (matching training process)...")
        
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
        
        # Lag features (reduced periods as per training)
        lag_periods = [1, 5, 15, 30]
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics (reduced windows as per training)
        windows = [5, 15, 30]
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Rate of change features
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_5'] = df[col].diff(5)
            df[f'{col}_pct_change_5'] = df[col].pct_change(5)
        
        # Handle NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if col != 'date_time':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        df = df.dropna().reset_index(drop=True)
        
        print(f"   Final dataset: {len(df):,} records with {len(df.columns)-1} features")
        return df
    
    def create_sequences(self, df):
        """Create sequences matching training process."""
        print(f"🔄 Creating sequences (length={self.config['sequence_length']})...")
        
        # Sample every 3rd row (matching training)
        df_sampled = df.iloc[::3].reset_index(drop=True)
        print(f"   Sampled to {len(df_sampled):,} records")
        
        # Separate features and target
        feature_cols = [col for col in df_sampled.columns if col not in ['date_time', 'temperature']]
        
        X = df_sampled[feature_cols].values
        y = df_sampled['temperature'].values
        
        # Create sequences
        sequence_length = self.config['sequence_length']
        X_sequences = []
        y_sequences = []
        timestamps = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i - sequence_length:i])
            y_sequences.append(y[i])
            timestamps.append(df_sampled['date_time'].iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"   Created {len(X_sequences):,} sequences")
        return X_sequences, y_sequences, timestamps, feature_cols
    
    def split_data_chronologically(self, X, y, timestamps, train_size=0.7, val_size=0.15):
        """Split data chronologically (matching training)."""
        print("🔄 Splitting data chronologically...")
        
        n_samples = len(X)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))
        
        # Training set
        X_train = X[:train_end]
        y_train = y[:train_end]
        t_train = timestamps[:train_end]
        
        # Validation set
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        t_val = timestamps[train_end:val_end]
        
        # Test set
        X_test = X[val_end:]
        y_test = y[val_end:]
        t_test = timestamps[val_end:]
        
        print(f"   Train: {len(X_train):,} samples ({t_train[0]} to {t_train[-1]})")
        print(f"   Val:   {len(X_val):,} samples ({t_val[0]} to {t_val[-1]})")
        print(f"   Test:  {len(X_test):,} samples ({t_test[0]} to {t_test[-1]})")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test)
    
    def scale_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Scale data using saved scalers."""
        print("🔄 Scaling data with saved scalers...")
        
        # Scale features
        n_train, seq_len, n_features = X_train.shape
        X_train_scaled = self.scaler_features.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
        X_val_scaled = self.scaler_features.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
        X_test_scaled = self.scaler_features.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], seq_len, n_features)
        
        # Scale targets
        y_train_scaled = self.scaler_target.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled

    def analyze_imbalance_with_smote(self, X_train, X_test, y_train, y_test):
        """Analyze target imbalance via binning and SMOTE on flattened features.

        Note: SMOTE natively supports classification. We approximate by binning
        continuous temperature into quantile-based classes, apply SMOTE to balance
        bins on the last-timestep features, and compare per-bin errors using a
        RandomForestRegressor trained on original vs SMOTE-augmented data.
        """
        print("\n" + "="*60)
        print("4️⃣  SMOTE IMBALANCE ANALYSIS (Quantile bin approximation)")
        print("="*60)

        # Flatten to last-timestep features for a tabular baseline
        X_train_flat = X_train[:, -1, :]
        X_test_flat = X_test[:, -1, :]

        # Bin targets into quartiles
        try:
            y_train_bins = pd.qcut(y_train, q=4, labels=False, duplicates='drop')
            y_test_bins = pd.qcut(y_test, q=4, labels=False, duplicates='drop')
        except Exception as e:
            print(f"   ❌ Binning failed: {e}")
            return

        # Show class distribution
        train_counts = pd.Series(y_train_bins).value_counts().sort_index()
        test_counts = pd.Series(y_test_bins).value_counts().sort_index()
        print("   Training bin counts:", train_counts.to_dict())
        print("   Test bin counts:", test_counts.to_dict())

        # Apply SMOTE to balance bins
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_res, y_res_bins = smote.fit_resample(X_train_flat, y_train_bins)
            # For regression target, align y via median per bin of original
            bin_to_value = pd.Series(y_train).groupby(y_train_bins).median()
            y_res = pd.Series(y_res_bins).map(bin_to_value).values
            print("   ✅ SMOTE applied. New bin counts:", pd.Series(y_res_bins).value_counts().sort_index().to_dict())
        except Exception as e:
            print(f"   ❌ SMOTE failed: {e}")
            return

        # Train baseline regressors
        rf_orig = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_res = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_orig.fit(X_train_flat, y_train)
        rf_res.fit(X_res, y_res)

        pred_orig = rf_orig.predict(X_test_flat)
        pred_res = rf_res.predict(X_test_flat)

        # Overall comparison
        mae_orig = mean_absolute_error(y_test, pred_orig)
        mae_res = mean_absolute_error(y_test, pred_res)
        print(f"\n📊 Overall MAE - Original: {mae_orig:.4f}°C | SMOTE-aug: {mae_res:.4f}°C")

        # Per-bin comparison
        print("\n📊 Per-Bin MAE (quartiles):")
        print("   Bin   Count   MAE_orig   MAE_smote")
        for b in sorted(pd.unique(y_test_bins)):
            mask = (y_test_bins == b)
            if np.any(mask):
                mae_b_orig = mean_absolute_error(y_test[mask], pred_orig[mask])
                mae_b_res = mean_absolute_error(y_test[mask], pred_res[mask])
                print(f"   {int(b):>3}   {int(mask.sum()):>5}   {mae_b_orig:8.4f}   {mae_b_res:9.4f}")
    
    def evaluate_test_set(self, X_test_scaled, y_test):
        """1. SANITY CHECK: Evaluate on held-out test set with exact metrics."""
        print("\n" + "="*60)
        print("1️⃣  SANITY CHECK: TEST SET EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate exact metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        # Calculate accuracy percentage
        avg_temp = np.mean(y_test)
        accuracy = (1 - mae / avg_temp) * 100
        
        print(f"📊 TEST SET PERFORMANCE (N={len(y_test):,}):")
        print(f"   MAE:  {mae:.4f}°C")
        print(f"   RMSE: {rmse:.4f}°C")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Avg Temperature: {avg_temp:.2f}°C")
        
        # Store for later analysis
        self.test_results = {
            'y_true': y_test,
            'y_pred': y_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy': accuracy
        }
        
        return self.test_results
    
    def analyze_residuals(self, t_test):
        """Analyze residuals for bias and patterns."""
        print("\n🔍 RESIDUAL ANALYSIS:")
        
        residuals = self.test_results['y_true'] - self.test_results['y_pred']
        
        # Basic statistics
        print(f"   Mean residual: {np.mean(residuals):.4f}°C (bias check)")
        print(f"   Std residual:  {np.std(residuals):.4f}°C")
        print(f"   Skewness:      {stats.skew(residuals):.4f}")
        print(f"   Kurtosis:      {stats.kurtosis(residuals):.4f}")
        
        # Check for bias
        if abs(np.mean(residuals)) > 0.1:
            print("   ⚠️  BIAS DETECTED: Mean residual > 0.1°C")
        else:
            print("   ✅ No significant bias detected")
        
        # Create residual plots
        self.plot_residuals(residuals, t_test)
        
        return residuals
    
    def plot_residuals(self, residuals, timestamps):
        """Plot residual analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Residual Analysis - Overfitting Detection', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Actual
        axes[0, 0].scatter(self.test_results['y_true'], self.test_results['y_pred'], alpha=0.6, s=1)
        axes[0, 0].plot([self.test_results['y_true'].min(), self.test_results['y_true'].max()], 
                       [self.test_results['y_true'].min(), self.test_results['y_true'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Temperature (°C)')
        axes[0, 0].set_ylabel('Predicted Temperature (°C)')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to plot
        axes[0, 0].text(0.05, 0.95, f'R² = {self.test_results["r2"]:.4f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Residuals vs Predictions
        axes[0, 1].scatter(self.test_results['y_pred'], residuals, alpha=0.6, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].axhline(y=np.mean(residuals), color='orange', linestyle='-', label=f'Mean: {np.mean(residuals):.3f}°C')
        axes[0, 1].set_xlabel('Predicted Temperature (°C)')
        axes[0, 1].set_ylabel('Residuals (°C)')
        axes[0, 1].set_title('Residuals vs Predictions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution
        axes[0, 2].hist(residuals, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[0, 2].axvline(x=0, color='r', linestyle='--')
        axes[0, 2].axvline(x=np.mean(residuals), color='orange', linestyle='-')
        axes[0, 2].set_xlabel('Residuals (°C)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Residual Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residuals over time
        sample_indices = np.linspace(0, len(residuals)-1, min(2000, len(residuals)), dtype=int)
        axes[1, 0].plot(sample_indices, residuals[sample_indices], alpha=0.7, linewidth=0.8)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].axhline(y=np.mean(residuals), color='orange', linestyle='-')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Residuals (°C)')
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Residuals by hour of day
        timestamps_arr = np.array(timestamps)
        timestamps_sample = timestamps_arr[sample_indices] if len(timestamps_arr) > len(sample_indices) else timestamps_arr
        hours = [t.hour for t in timestamps_sample]
        residuals_sample = residuals[sample_indices] if len(residuals) > len(sample_indices) else residuals
        
        hour_residuals = {}
        for h in range(24):
            hour_mask = np.array(hours) == h
            if np.any(hour_mask):
                hour_residuals[h] = residuals_sample[hour_mask]
        
        if hour_residuals:
            axes[1, 2].boxplot([hour_residuals[h] for h in sorted(hour_residuals.keys())], 
                              labels=sorted(hour_residuals.keys()))
            axes[1, 2].axhline(y=0, color='r', linestyle='--')
            axes[1, 2].set_xlabel('Hour of Day')
            axes[1, 2].set_ylabel('Residuals (°C)')
            axes[1, 2].set_title('Residuals by Hour of Day')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        print("   📊 Residual plots saved: residual_analysis.png")
        plt.show()
    
    def check_temporal_leakage(self, timestamps_train, timestamps_val, timestamps_test):
        """Check for temporal leakage in data splits."""
        print("\n🔍 TEMPORAL LEAKAGE CHECK:")
        
        max_train = max(timestamps_train)
        min_val = min(timestamps_val)
        max_val = max(timestamps_val)
        min_test = min(timestamps_test)
        
        print(f"   Train ends:   {max_train}")
        print(f"   Val starts:   {min_val}")
        print(f"   Val ends:     {max_val}")
        print(f"   Test starts:  {min_test}")
        
        # Check for overlaps
        if min_val > max_train:
            print("   ✅ No train-val temporal overlap")
        else:
            print("   ❌ TEMPORAL LEAKAGE: Train-val overlap detected!")
        
        if min_test > max_val:
            print("   ✅ No val-test temporal overlap")
        else:
            print("   ❌ TEMPORAL LEAKAGE: Val-test overlap detected!")
    
    def analyze_error_patterns(self):
        """Analyze error patterns by temperature bins and time."""
        print("\n🔍 ERROR PATTERN ANALYSIS:")
        
        y_true = self.test_results['y_true']
        y_pred = self.test_results['y_pred']
        errors = np.abs(y_true - y_pred)
        
        # Error by temperature bins
        temp_bins = np.percentile(y_true, [0, 25, 50, 75, 100])
        bin_labels = ['Low', 'Med-Low', 'Med-High', 'High']
        
        print(f"   📊 Error by Temperature Bins:")
        for i, label in enumerate(bin_labels):
            mask = (y_true >= temp_bins[i]) & (y_true < temp_bins[i+1])
            bin_mae = np.mean(errors[mask])
            bin_count = np.sum(mask)
            temp_range = f"{temp_bins[i]:.1f}-{temp_bins[i+1]:.1f}°C"
            print(f"      {label:8} ({temp_range:8}): MAE={bin_mae:.4f}°C (N={bin_count:,})")
        
        # Find problematic ranges
        worst_bin_idx = np.argmax([np.mean(errors[(y_true >= temp_bins[i]) & (y_true < temp_bins[i+1])]) 
                                  for i in range(len(bin_labels))])
        print(f"   ⚠️  Highest errors in {bin_labels[worst_bin_idx]} temperature range")
    
    def check_residual_autocorrelation(self):
        """2. DEEPER CHECK: Check residual autocorrelation."""
        print("\n" + "="*60)
        print("2️⃣  DEEPER CHECK: RESIDUAL AUTOCORRELATION")
        print("="*60)
        
        residuals = self.test_results['y_true'] - self.test_results['y_pred']
        
        # Calculate ACF and PACF
        try:
            lags = min(50, len(residuals)//4)
            acf_vals = acf(residuals, nlags=lags, fft=True)
            pacf_vals = pacf(residuals, nlags=lags)
            
            # Check for significant autocorrelation
            significant_lags = np.sum(np.abs(acf_vals[1:11]) > 0.1)  # First 10 lags
            
            print(f"   Autocorrelation at lag 1: {acf_vals[1]:.4f}")
            print(f"   Significant lags (|ACF| > 0.1): {significant_lags}/10")
            
            if significant_lags > 2:
                print("   ⚠️  STRONG AUTOCORRELATION: Model missing temporal structure")
            elif significant_lags > 0:
                print("   ⚠️  MODERATE AUTOCORRELATION: Some temporal patterns missed")
            else:
                print("   ✅ Low autocorrelation: Good temporal modeling")
            
            # Plot ACF/PACF
            self.plot_autocorrelation(acf_vals, pacf_vals)
            
        except Exception as e:
            print(f"   ❌ Error in autocorrelation analysis: {e}")
    
    def plot_autocorrelation(self, acf_vals, pacf_vals):
        """Plot ACF and PACF."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ACF plot
        lags = range(len(acf_vals))
        ax1.stem(lags, acf_vals, basefmt=" ")
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Autocorrelation Function (ACF)')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')
        ax1.grid(True, alpha=0.3)
        
        # PACF plot
        lags_pacf = range(len(pacf_vals))
        ax2.stem(lags_pacf, pacf_vals, basefmt=" ")
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('PACF')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
        print("   📊 Autocorrelation plots saved: autocorrelation_analysis.png")
        plt.show()
    
    def analyze_performance_drift(self, timestamps_test):
        """Check for performance drift over time."""
        print("\n🔍 PERFORMANCE DRIFT ANALYSIS:")
        
        y_true = self.test_results['y_true']
        y_pred = self.test_results['y_pred']
        
        # Calculate rolling MAE (7-day windows)
        window_size = min(1000, len(y_true)//10)  # Adaptive window size
        rolling_mae = []
        window_centers = []
        
        for i in range(window_size//2, len(y_true) - window_size//2):
            window_true = y_true[i-window_size//2:i+window_size//2]
            window_pred = y_pred[i-window_size//2:i+window_size//2]
            window_mae = mean_absolute_error(window_true, window_pred)
            rolling_mae.append(window_mae)
            window_centers.append(i)
        
        # Check for drift
        if len(rolling_mae) > 10:
            start_mae = np.mean(rolling_mae[:len(rolling_mae)//4])
            end_mae = np.mean(rolling_mae[-len(rolling_mae)//4:])
            drift_pct = ((end_mae - start_mae) / start_mae) * 100
            
            print(f"   Early period MAE: {start_mae:.4f}°C")
            print(f"   Late period MAE:  {end_mae:.4f}°C")
            print(f"   Performance drift: {drift_pct:+.1f}%")
            
            if abs(drift_pct) > 20:
                print("   ⚠️  SIGNIFICANT DRIFT: Model degrading over time")
            elif abs(drift_pct) > 10:
                print("   ⚠️  MODERATE DRIFT: Some performance change")
            else:
                print("   ✅ Stable performance over time")
            
            # Plot drift
            self.plot_performance_drift(rolling_mae, window_centers, timestamps_test)
    
    def plot_performance_drift(self, rolling_mae, window_centers, timestamps):
        """Plot performance drift over time."""
        plt.figure(figsize=(15, 6))
        
        # Convert indices to timestamps for plotting
        if len(timestamps) > max(window_centers):
            time_labels = [timestamps[i] for i in window_centers]
            plt.plot(time_labels, rolling_mae, linewidth=2)
            plt.xticks(rotation=45)
        else:
            plt.plot(window_centers, rolling_mae, linewidth=2)
        
        plt.axhline(y=np.mean(rolling_mae), color='r', linestyle='--', 
                   label=f'Mean MAE: {np.mean(rolling_mae):.4f}°C')
        plt.title('Performance Drift Analysis - Rolling MAE Over Time')
        plt.xlabel('Time')
        plt.ylabel('Rolling MAE (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('performance_drift.png', dpi=300, bbox_inches='tight')
        print("   📊 Performance drift plot saved: performance_drift.png")
        plt.show()
    
    def compare_with_baselines(self, X_train, X_test, y_train, y_test):
        """Compare with simple baselines on same test split."""
        print("\n" + "="*60)
        print("3️⃣  BASELINE COMPARISONS")
        print("="*60)
        
        baselines = {}
        
        # 1. Last observed value (naive)
        print("🔄 Testing naive baseline (last observed value)...")
        naive_pred = np.roll(y_test, 1)
        naive_pred[0] = y_test[0]  # Handle first prediction
        naive_mae = mean_absolute_error(y_test, naive_pred)
        baselines['Naive'] = naive_mae
        
        # 2. Linear Regression
        print("🔄 Testing linear regression baseline...")
        try:
            # Use first timestep of sequences as features for linear regression
            X_train_flat = X_train[:, -1, :10]  # Last timestep, first 10 features
            X_test_flat = X_test[:, -1, :10]
            
            lr = LinearRegression()
            lr.fit(X_train_flat, y_train)
            lr_pred = lr.predict(X_test_flat)
            lr_mae = mean_absolute_error(y_test, lr_pred)
            baselines['Linear Regression'] = lr_mae
        except Exception as e:
            print(f"   ⚠️  Linear regression failed: {e}")
            baselines['Linear Regression'] = np.nan
        
        # 3. Simple moving average
        print("🔄 Testing moving average baseline...")
        window = 5
        ma_pred = np.convolve(y_test, np.ones(window)/window, mode='same')
        # Fix edges
        ma_pred[:window//2] = y_test[:window//2]
        ma_pred[-(window//2):] = y_test[-(window//2):]
        ma_mae = mean_absolute_error(y_test, ma_pred)
        baselines['Moving Average'] = ma_mae
        
        # 4. Current model
        lstm_mae = self.test_results['mae']
        baselines['LSTM Model'] = lstm_mae
        
        # Print comparison
        print("\n📊 BASELINE COMPARISON:")
        print("   Method                    MAE (°C)    Improvement")
        print("   " + "-"*50)
        
        sorted_baselines = sorted(baselines.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
        best_mae = sorted_baselines[0][1]
        
        for method, mae in sorted_baselines:
            if not np.isnan(mae):
                if mae == best_mae:
                    improvement = "BEST"
                else:
                    improvement = f"{((mae - best_mae) / mae * 100):+.1f}%"
                print(f"   {method:<25} {mae:8.4f}    {improvement}")
        
        # Check if LSTM is actually best
        if baselines['LSTM Model'] != min(b for b in baselines.values() if not np.isnan(b)):
            print("\n   ⚠️  WARNING: LSTM is NOT the best model!")
            print("   This suggests potential overfitting or poor generalization.")
        else:
            improvement = (baselines['Naive'] - baselines['LSTM Model']) / baselines['Naive'] * 100
            print(f"\n   ✅ LSTM beats naive baseline by {improvement:.1f}%")
    
    def seasonal_analysis(self, y_test, timestamps_test):
        """Analyze performance by season."""
        print("\n🔍 SEASONAL ANALYSIS:")
        
        y_pred = self.test_results['y_pred']
        
        # Group by season
        seasonal_performance = {}
        for i, timestamp in enumerate(timestamps_test):
            month = timestamp.month
            if month in [12, 1, 2]:
                season = 'Winter'
            elif month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            else:
                season = 'Fall'
            
            if season not in seasonal_performance:
                seasonal_performance[season] = {'true': [], 'pred': []}
            seasonal_performance[season]['true'].append(y_test[i])
            seasonal_performance[season]['pred'].append(y_pred[i])
        
        print("   📊 Performance by Season:")
        print("   Season      MAE (°C)    Count")
        print("   " + "-"*30)
        
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            if season in seasonal_performance and len(seasonal_performance[season]['true']) > 0:
                season_mae = mean_absolute_error(
                    seasonal_performance[season]['true'],
                    seasonal_performance[season]['pred']
                )
                count = len(seasonal_performance[season]['true'])
                print(f"   {season:<10}  {season_mae:8.4f}    {count:,}")
    
    def generate_final_report(self):
        """Generate final overfitting assessment report."""
        print("\n" + "="*60)
        print("🎯 FINAL OVERFITTING ASSESSMENT REPORT")
        print("="*60)
        
        # Collect evidence
        evidence = []
        
        # Test performance vs reported validation
        val_mae_reported = 0.2864  # From training logs
        test_mae_actual = self.test_results['mae']
        
        performance_gap = ((test_mae_actual - val_mae_reported) / val_mae_reported) * 100
        
        print(f"📊 PERFORMANCE COMPARISON:")
        print(f"   Validation MAE (reported): {val_mae_reported:.4f}°C")
        print(f"   Test MAE (actual):         {test_mae_actual:.4f}°C")
        print(f"   Performance gap:           {performance_gap:+.1f}%")
        
        if performance_gap > 20:
            evidence.append("❌ Large test-validation gap (>20%)")
        elif performance_gap > 10:
            evidence.append("⚠️  Moderate test-validation gap (10-20%)")
        else:
            evidence.append("✅ Small test-validation gap (<10%)")
        
        # Accuracy assessment
        if self.test_results['accuracy'] > 99:
            evidence.append("❌ Unrealistically high accuracy (>99%)")
        elif self.test_results['accuracy'] > 98:
            evidence.append("⚠️  Very high accuracy (>98%) - check for leakage")
        else:
            evidence.append("✅ Reasonable accuracy level")
        
        print(f"\n🔍 OVERFITTING EVIDENCE:")
        for item in evidence:
            print(f"   {item}")
        
        # Final verdict
        red_flags = sum(1 for item in evidence if item.startswith("❌"))
        yellow_flags = sum(1 for item in evidence if item.startswith("⚠️"))
        
        print(f"\n⚖️  FINAL VERDICT:")
        if red_flags >= 2:
            print("   🚨 SIGNIFICANT OVERFITTING DETECTED")
            print("   Recommendation: Rebuild model with simpler features")
        elif red_flags == 1 or yellow_flags >= 2:
            print("   ⚠️  MODERATE OVERFITTING LIKELY")
            print("   Recommendation: Add regularization, reduce features")
        else:
            print("   ✅ MINIMAL OVERFITTING - Model appears robust")
            print("   Recommendation: Model ready for production")

def main():
    """Run comprehensive model validation."""
    print("🔬 COMPREHENSIVE MODEL VALIDATION & OVERFITTING DETECTION")
    print("="*80)
    print("This script will thoroughly test your LSTM model for overfitting")
    print("and provide detailed performance analysis.")
    print("="*80)
    
    # Initialize validator
    validator = ModelValidator()
    
    # Load and process data
    data_file = r"C:\Users\operator\Desktop\smartvayu\data\combined_plus_sensor_data.csv"
    df = validator.load_and_process_data(data_file)
    
    # Feature engineering
    df_features = validator.engineer_features(df)
    
    # Create sequences
    X, y, timestamps, feature_cols = validator.create_sequences(df_features)
    
    # Split data
    splits = validator.split_data_chronologically(X, y, timestamps)
    X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = splits
    
    # Scale data
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = validator.scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # 1. SANITY CHECKS
    test_results = validator.evaluate_test_set(X_test_s, y_test)
    residuals = validator.analyze_residuals(t_test)
    validator.check_temporal_leakage(t_train, t_val, t_test)
    validator.analyze_error_patterns()
    
    # 2. DEEPER CHECKS
    validator.check_residual_autocorrelation()
    validator.analyze_performance_drift(t_test)
    validator.seasonal_analysis(y_test, t_test)
    
    # 3. BASELINE COMPARISONS
    validator.compare_with_baselines(X_train, X_test, y_train, y_test)
    
    # 4. FINAL ASSESSMENT
    validator.generate_final_report()

    # 5. SMOTE IMBALANCE ANALYSIS
    validator.analyze_imbalance_with_smote(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*80)
    print("🎉 VALIDATION COMPLETE!")
    print("Check the generated plots and analysis above for detailed insights.")
    print("Generated files:")
    print("   📊 residual_analysis.png")
    print("   📊 autocorrelation_analysis.png") 
    print("   📊 performance_drift.png")
    print("="*80)

if __name__ == "__main__":
    main()
