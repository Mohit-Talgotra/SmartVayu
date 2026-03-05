"""
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
    def __init__(self, model_dir="temperature_lstm_model"):
        """Load trained model and preprocessors."""
        print("Loading trained temperature prediction model...")
        
        self.model = load_model(f"{model_dir}/lstm_model.h5")
        self.scaler_features = joblib.load(f"{model_dir}/scaler_features.pkl")
        self.scaler_target = joblib.load(f"{model_dir}/scaler_target.pkl")
        self.config = joblib.load(f"{model_dir}/model_config.pkl")
        
        self.feature_columns = self.config['feature_columns']
        self.sequence_length = self.config['sequence_length']
        
        print(f"Model loaded successfully!")
        print(f"Sequence length: {self.sequence_length} time steps")
        print(f"Features required: {len(self.feature_columns)}")
    
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
            raise ValueError(f"Need at least {self.sequence_length} data points, got {len(data)}")
        
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
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling features
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for window in [5, 15, 30]:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Rate of change
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_5'] = df[col].diff(5)
            df[f'{col}_pct_change_5'] = df[col].pct_change(5)
        
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
    # print(f"Predicted next temperature: {predicted_temp:.2f}°C")
