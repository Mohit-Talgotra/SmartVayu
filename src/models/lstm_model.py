
"""
Temperature Prediction Model - Deep Learning Time Series Analysis
================================================================

This script implements a deep learning model to predict temperature (t+1) using:
- Target: temperature 
- Features: humidity, pressure, light + engineered time features
- Model: LSTM neural network for time series prediction
- Validation: Proper time-based splits to avoid data leakage

Dataset: combined_plus_sensor_data.csv (1.5M records, June 2021 - July 2022)
Author: AI Assistant
Date: 2025-09-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

# Sklearn imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import joblib
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TemperatureLSTMModel:
    """
    LSTM-based temperature prediction model for time series data.
    
    This class handles:
    - Time series data preprocessing 
    - Feature engineering (lags, rolling stats, time features)
    - LSTM model training with proper validation
    - Prediction and evaluation
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=1):
        """
        Initialize the temperature prediction model.
        
        Args:
            sequence_length (int): Number of time steps to look back (default: 60 minutes)
            prediction_horizon (int): Steps ahead to predict (default: 1 for t+1)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_columns = None
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the sensor data.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Preprocessed dataframe ready for modeling
        """
        print("🔄 Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"   Loaded {len(df):,} records from {df['date_time'].min()} to {df['date_time'].max()}")
        
        # Convert datetime and sort chronologically
        print("🔄 Processing datetime...")
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # Keep only required columns (ignore rpi_id as requested)
        required_cols = ['date_time', 'temperature', 'humidity', 'pressure', 'light']
        df = df[required_cols].copy()
        
        print(f"   Using features: {[col for col in required_cols if col != 'date_time']}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️  Missing values found: {missing[missing > 0].to_dict()}")
            df = df.dropna().reset_index(drop=True)
            print(f"   After dropping missing values: {len(df):,} records")
        
        return df
    
    def engineer_features(self, df):
        """
        Create time-based and lag features for time series modeling.
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime column
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("🔄 Engineering time series features...")
        
        df = df.copy()
        
        # === TIME FEATURES ===
        print("   Creating time-based features...")
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['day_of_year'] = df['date_time'].dt.dayofyear
        df['month'] = df['date_time'].dt.month
        df['quarter'] = df['date_time'].dt.quarter
        
        # Cyclical encoding for seasonality
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # === LAG FEATURES ===
        print("   Creating lag features...")
        lag_periods = [1, 5, 15, 30]  # Reduced lag periods for memory efficiency
        
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # === ROLLING STATISTICS ===
        print("   Creating rolling window features...")
        windows = [5, 15, 30]  # Reduced window sizes for memory efficiency
        
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # === RATE OF CHANGE FEATURES ===
        print("   Creating rate of change features...")
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_5'] = df[col].diff(5)
            df[f'{col}_pct_change_5'] = df[col].pct_change(5)
        
        # Handle NaN and infinite values from feature engineering
        initial_rows = len(df)
        
        # Replace infinite values with NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with forward fill, then backward fill
        for col in df.columns:
            if col != 'date_time':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        print(f"   Handled infinite/NaN values, dropped {initial_rows - len(df):,} rows")
        print(f"   Final dataset: {len(df):,} records with {len(df.columns)-1} features")
        
        return df
    
    def create_sequences(self, df):
        """
        Create sequences for LSTM training.
        
        Args:
            df (pd.DataFrame): Dataframe with features
            
        Returns:
            tuple: (X_sequences, y, feature_columns)
        """
        print(f"🔄 Creating sequences for LSTM (sequence_length={self.sequence_length})...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['date_time', 'temperature']]
        target_col = 'temperature'
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X) - self.prediction_horizon + 1):
            # Features: look back 'sequence_length' time steps
            X_sequences.append(X[i - self.sequence_length:i])
            # Target: predict 'prediction_horizon' steps ahead
            y_sequences.append(y[i + self.prediction_horizon - 1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"   Created {len(X_sequences):,} sequences")
        print(f"   Sequence shape: {X_sequences.shape}")
        print(f"   Target shape: {y_sequences.shape}")
        
        self.feature_columns = feature_cols
        return X_sequences, y_sequences
    
    def time_series_split(self, X, y, train_size=0.7, val_size=0.15):
        """
        Split data chronologically for time series (no shuffling).
        
        Args:
            X (np.array): Feature sequences
            y (np.array): Target values
            train_size (float): Proportion for training
            val_size (float): Proportion for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("🔄 Splitting data chronologically...")
        
        n_samples = len(X)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        print(f"   Train: {len(X_train):,} samples ({len(X_train)/n_samples:.1%})")
        print(f"   Val:   {len(X_val):,} samples ({len(X_val)/n_samples:.1%})")
        print(f"   Test:  {len(X_test):,} samples ({len(X_test)/n_samples:.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Scale features and target using training data statistics.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Target arrays
            
        Returns:
            tuple: Scaled arrays
        """
        print("🔄 Scaling features and target...")
        
        # Scale features (reshape for scaling, then back to sequence format)
        n_samples_train, seq_len, n_features = X_train.shape
        
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        # Use RobustScaler as data may have outliers (from analysis)
        self.scaler_features = RobustScaler()
        X_train_scaled = self.scaler_features.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler_features.transform(X_val_reshaped)
        X_test_scaled = self.scaler_features.transform(X_test_reshaped)
        
        # Reshape back to sequences
        X_train_scaled = X_train_scaled.reshape(n_samples_train, seq_len, n_features)
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], seq_len, n_features)
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], seq_len, n_features)
        
        # Scale target
        self.scaler_target = RobustScaler()
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"   Scaled features: {X_train_scaled.shape}")
        print(f"   Feature scaler: RobustScaler")
        print(f"   Target scaler: RobustScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input sequences
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        print("🔄 Building LSTM model...")
        
        model = Sequential([
            # First LSTM layer with dropout (reduced size)
            LSTM(64, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer (reduced size)
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers for final prediction
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')  # Linear for regression
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("   Model Architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the LSTM model with callbacks.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Training history
        """
        print("🔄 Training LSTM model...")
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history.history
    
    def evaluate_model(self, X_test, y_test, dataset_name="Test"):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            dataset_name: Name for printing results
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"🔄 Evaluating model on {dataset_name} set...")
        
        # Make predictions (scaled)
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform to original scale
        y_test_original = self.scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = self.scaler_target.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        
        # Calculate MAPE (avoiding division by zero)
        mask = y_test_original != 0
        mape = np.mean(np.abs((y_test_original[mask] - y_pred_original[mask]) / y_test_original[mask])) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        print(f"   📊 {dataset_name} Set Performance:")
        print(f"      MAE:  {mae:.4f}°C")
        print(f"      RMSE: {rmse:.4f}°C")
        print(f"      R²:   {r2:.4f}")
        print(f"      MAPE: {mape:.2f}%")
        
        return metrics, y_test_original, y_pred_original
    
    def plot_results(self, history, y_test, y_pred, save_path="model_results.png"):
        """
        Create comprehensive plots of model results.
        
        Args:
            history: Training history
            y_test: True test values
            y_pred: Predicted test values
            save_path: Path to save the plot
        """
        print("🔄 Creating result visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temperature Prediction Model Results', fontsize=16, fontweight='bold')
        
        # 1. Training History - Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
        axes[0, 0].set_title('Model Loss During Training')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training History - MAE
        axes[0, 1].plot(history['mae'], label='Training MAE', color='blue', alpha=0.7)
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='red', alpha=0.7)
        axes[0, 1].set_title('Model MAE During Training')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Predictions vs Actual (scatter plot)
        axes[0, 2].scatter(y_test, y_pred, alpha=0.5, s=1)
        axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 2].set_title('Predictions vs Actual')
        axes[0, 2].set_xlabel('Actual Temperature (°C)')
        axes[0, 2].set_ylabel('Predicted Temperature (°C)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Time Series Plot (sample of predictions)
        sample_size = min(2000, len(y_test))
        sample_indices = np.linspace(0, len(y_test)-1, sample_size, dtype=int)
        
        axes[1, 0].plot(y_test[sample_indices], label='Actual', alpha=0.7, linewidth=1)
        axes[1, 0].plot(y_pred[sample_indices], label='Predicted', alpha=0.7, linewidth=1)
        axes[1, 0].set_title(f'Temperature Predictions Over Time (Sample of {sample_size} points)')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residuals Plot
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residuals vs Predictions')
        axes[1, 1].set_xlabel('Predicted Temperature (°C)')
        axes[1, 1].set_ylabel('Residuals (°C)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Residuals Distribution
        axes[1, 2].hist(residuals, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[1, 2].set_title('Distribution of Residuals')
        axes[1, 2].set_xlabel('Residuals (°C)')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add statistics text
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}°C\nRMSE = {rmse:.4f}°C'
        axes[0, 2].text(0.05, 0.95, stats_text, transform=axes[0, 2].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Results plot saved: {save_path}")
        plt.show()
    
    def save_model_artifacts(self, model_dir="temperature_model"):
        """
        Save model and preprocessing artifacts.
        
        Args:
            model_dir: Directory to save artifacts
        """
        print("🔄 Saving model artifacts...")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save Keras model
        model_path = os.path.join(model_dir, "lstm_model.h5")
        self.model.save(model_path)
        print(f"   💾 Model saved: {model_path}")
        
        # Save scalers
        scaler_features_path = os.path.join(model_dir, "scaler_features.pkl")
        scaler_target_path = os.path.join(model_dir, "scaler_target.pkl")
        
        joblib.dump(self.scaler_features, scaler_features_path)
        joblib.dump(self.scaler_target, scaler_target_path)
        print(f"   💾 Feature scaler saved: {scaler_features_path}")
        print(f"   💾 Target scaler saved: {scaler_target_path}")
        
        # Save feature columns and model config
        config = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }
        
        config_path = os.path.join(model_dir, "model_config.pkl")
        joblib.dump(config, config_path)
        print(f"   💾 Model config saved: {config_path}")
        
        # Create prediction utility script
        self.create_prediction_script(model_dir)
        
        print(f"✅ All artifacts saved to: {model_dir}/")
    
    def create_prediction_script(self, model_dir):
        """
        Create a standalone prediction script.
        
        Args:
            model_dir: Directory where model artifacts are saved
        """
        script_content = f'''"""
Temperature Prediction Utility
=============================

This script loads the trained LSTM model and makes temperature predictions.
Usage: python predict_temperature.py

Generated automatically from temperature_prediction_model.py
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class TemperaturePredictor:
    def __init__(self, model_dir="{model_dir}"):
        """Load trained model and preprocessors."""
        print("Loading trained temperature prediction model...")
        
        self.model = load_model(f"{{model_dir}}/lstm_model.h5")
        self.scaler_features = joblib.load(f"{{model_dir}}/scaler_features.pkl")
        self.scaler_target = joblib.load(f"{{model_dir}}/scaler_target.pkl")
        self.config = joblib.load(f"{{model_dir}}/model_config.pkl")
        
        self.feature_columns = self.config['feature_columns']
        self.sequence_length = self.config['sequence_length']
        
        print(f"Model loaded successfully!")
        print(f"Sequence length: {{self.sequence_length}} time steps")
        print(f"Features required: {{len(self.feature_columns)}}")
    
    def predict_next_temperature(self, data):
        """
        Predict next temperature given recent sensor data.
        
        Args:
            data: pandas DataFrame with columns: date_time, humidity, pressure, light, temperature
                  Must contain at least 'sequence_length' recent rows
        
        Returns:
            float: Predicted temperature for next time step
        """
        if len(data) < self.sequence_length:
            raise ValueError(f"Need at least {{self.sequence_length}} data points, got {{len(data)}}")
        
        # Use the most recent sequence_length rows
        recent_data = data.tail(self.sequence_length).copy()
        
        # Engineer features (simplified version)
        processed_data = self._engineer_features_simple(recent_data)
        
        # Extract features
        feature_values = processed_data[self.feature_columns].values
        
        # Scale features
        feature_values_scaled = self.scaler_features.transform(feature_values)
        
        # Reshape for LSTM (1 sample, sequence_length, n_features)
        X = feature_values_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
        
        # Make prediction
        y_pred_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform to original scale
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled)[0, 0]
        
        return y_pred
    
    def _engineer_features_simple(self, df):
        """Simplified feature engineering for prediction."""
        df = df.copy()
        df['date_time'] = pd.to_datetime(df['date_time'])
        
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
        
        # Lag features
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for lag in [1, 5, 15, 30]:
                df[f'{{col}}_lag_{{lag}}'] = df[col].shift(lag)
        
        # Rolling features
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for window in [5, 15, 30]:
                df[f'{{col}}_roll_mean_{{window}}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{{col}}_roll_std_{{window}}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{{col}}_roll_min_{{window}}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{{col}}_roll_max_{{window}}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Rate of change
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            df[f'{{col}}_diff_1'] = df[col].diff(1)
            df[f'{{col}}_diff_5'] = df[col].diff(5)
            df[f'{{col}}_pct_change_5'] = df[col].pct_change(5)
        
        # Fill NaN values with forward fill for prediction
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = TemperaturePredictor()
    
    # Example: Load some recent data and make prediction
    # data = pd.read_csv("recent_sensor_data.csv")  # Your recent data
    # predicted_temp = predictor.predict_next_temperature(data)
    # print(f"Predicted next temperature: {{predicted_temp:.2f}}°C")
'''
        
        script_path = os.path.join(model_dir, "predict_temperature.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"   📄 Prediction script created: {script_path}")

def main():
    """
    Main function to run the complete temperature prediction pipeline.
    """
    print("="*70)
    print("🌡️  TEMPERATURE PREDICTION MODEL - LSTM TIME SERIES")
    print("="*70)
    print("Features: humidity, pressure, light + engineered time features")
    print("Target: temperature (t+1 prediction)")
    print("Model: LSTM neural network")
    print("="*70)
    
    # Initialize model with reduced sequence length for memory efficiency
    lstm_model = TemperatureLSTMModel(sequence_length=30, prediction_horizon=1)
    
    # Define file path
    data_file = r"C:\Users\operator\Desktop\smartvayu\data\combined_plus_sensor_data.csv"
    
    try:
        # 1. Load and preprocess data
        df = lstm_model.load_and_preprocess_data(data_file)
        
        # 2. Engineer features
        df_features = lstm_model.engineer_features(df)
        
        # 2.5. Sample data for memory efficiency (use every 3rd row to reduce size)
        print(f"🔄 Sampling data for memory efficiency...")
        original_size = len(df_features)
        df_features = df_features.iloc[::3].reset_index(drop=True)  # Take every 3rd row
        print(f"   Sampled from {original_size:,} to {len(df_features):,} records ({len(df_features)/original_size:.1%})")
        
        # 3. Create sequences for LSTM
        X, y = lstm_model.create_sequences(df_features)
        
        # 4. Split data chronologically
        X_train, X_val, X_test, y_train, y_val, y_test = lstm_model.time_series_split(X, y)
        
        # 5. Scale features and target
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = \
            lstm_model.scale_features(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 6. Train model
        print("\\n" + "="*50)
        print("🚀 STARTING MODEL TRAINING")
        print("="*50)
        
        history = lstm_model.train_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        
        # 7. Evaluate model
        print("\\n" + "="*50)
        print("📊 MODEL EVALUATION")
        print("="*50)
        
        train_metrics, y_train_orig, y_train_pred = lstm_model.evaluate_model(
            X_train_scaled, y_train_scaled, "Training"
        )
        
        val_metrics, y_val_orig, y_val_pred = lstm_model.evaluate_model(
            X_val_scaled, y_val_scaled, "Validation"
        )
        
        test_metrics, y_test_orig, y_test_pred = lstm_model.evaluate_model(
            X_test_scaled, y_test_scaled, "Test"
        )
        
        # 8. Create visualizations
        print("\\n" + "="*50)
        print("📈 CREATING VISUALIZATIONS")
        print("="*50)
        
        lstm_model.plot_results(history, y_test_orig, y_test_pred, "lstm_model_results.png")
        
        # 9. Save model artifacts
        print("\\n" + "="*50)
        print("💾 SAVING MODEL ARTIFACTS")
        print("="*50)
        
        lstm_model.save_model_artifacts("temperature_lstm_model")
        
        # 10. Print final summary
        print("\\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\\n📋 FINAL RESULTS SUMMARY:")
        print("-" * 30)
        print(f"Test MAE:  {test_metrics['MAE']:.4f}°C")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}°C")
        print(f"Test R²:   {test_metrics['R²']:.4f}")
        print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")
        
        print("\\n🎯 MODEL PERFORMANCE INTERPRETATION:")
        if test_metrics['R²'] > 0.8:
            print("   ✅ Excellent model performance (R² > 0.8)")
        elif test_metrics['R²'] > 0.6:
            print("   ✅ Good model performance (R² > 0.6)")
        else:
            print("   ⚠️  Model performance could be improved (R² < 0.6)")
            
        if test_metrics['MAE'] < 1.0:
            print("   ✅ Very accurate predictions (MAE < 1°C)")
        elif test_metrics['MAE'] < 2.0:
            print("   ✅ Accurate predictions (MAE < 2°C)")
        else:
            print("   ⚠️  Prediction accuracy could be improved (MAE > 2°C)")
        
        print("\\n📁 SAVED ARTIFACTS:")
        print("   📊 lstm_model_results.png - Model performance plots")
        print("   🤖 temperature_lstm_model/ - Model and preprocessing files")
        print("   📄 predict_temperature.py - Prediction utility script")
        
        print("\\n🚀 NEXT STEPS:")
        print("   1. Review the performance plots to understand model behavior")
        print("   2. Use predict_temperature.py for making new predictions")
        print("   3. Consider hyperparameter tuning if performance needs improvement")
        print("   4. Collect more recent data for model updates")
        
        print("\\n" + "="*70)
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
