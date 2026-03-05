"""
Temperature Forecast Module
=========================
Generates 24-hour temperature forecasts using WeatherAPI.com data and trained ML model.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Constants
API_ENDPOINT = "http://api.weatherapi.com/v1/forecast.json"
MODEL_DIR = Path(__file__).parent.parent.parent / "trained_models" / "lstm_temperature"
SCALER_FEATURES = MODEL_DIR / "scaler_features.pkl"
SCALER_TARGET = MODEL_DIR / "scaler_target.pkl"
MODEL_PATH = MODEL_DIR / "lstm_model.h5"

def load_model_artifacts():
    """Load the trained model and scalers"""
    try:
        import tensorflow as tf  # Import here to avoid loading if not needed
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check if files exist
        for path in [MODEL_PATH, SCALER_FEATURES, SCALER_TARGET]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")
        
        # In TF 2.20+, metrics are functions
        mse = tf.keras.metrics.MeanSquaredError()
        mae = tf.keras.metrics.MeanAbsoluteError()
        
        # Define comprehensive custom objects
        custom_objects = {
            'mse': mse,
            'mae': mae,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'MSE': mse,
            'MAE': mae,
            'MeanSquaredError': tf.keras.metrics.MeanSquaredError,
            'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError
        }
        
        print("Loading model...")
        try:
            print("Attempting to load with custom metrics...")
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        except Exception as model_error:
            print(f"Failed to load model with custom metrics: {str(model_error)}")
            print("Attempting to load with compile=False...")
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("Model loaded successfully without compilation")
                print("Recompiling model with basic configuration...")
                model.compile(optimizer='adam', 
                            loss=tf.keras.losses.MeanSquaredError(),
                            metrics=[tf.keras.metrics.MeanAbsoluteError()])
            except Exception as e:
                print(f"Failed to load model without compilation: {str(e)}")
                raise
        
        print("Loading scalers...")
        feature_scaler = joblib.load(SCALER_FEATURES)
        target_scaler = joblib.load(SCALER_TARGET)
        
        return model, feature_scaler, target_scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

def fetch_weather_data(location, api_key):
    """
    Fetch 24-hour weather forecast from WeatherAPI.com
    
    Args:
        location (str): City name, lat/lon, or zip code
        api_key (str): WeatherAPI.com API key
    
    Returns:
        dict: Raw weather forecast data
    """
    params = {
        'key': "a1715ffa6e4c43a2ab8165946250411",
        'q': location,
        'days': 1,
        'aqi': 'no'
    }
    
    try:
        response = requests.get(API_ENDPOINT, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch weather data: {str(e)}")

def extract_features(forecast_data):
    """
    Extract and preprocess features from weather forecast data
    
    Args:
        forecast_data (dict): Raw weather data from API
    
    Returns:
        pd.DataFrame: Preprocessed features for model input
    """
    try:
        hourly_data = forecast_data['forecast']['forecastday'][0]['hour']
        
        # Define all features up front to maintain consistent order
        features = []
        for hour in hourly_data:
            dt = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M')
            
            # Map API fields to model's expected input
            feature_dict = {
                'temperature': hour['temp_c'],
                'humidity': hour['humidity'],
                'pressure': hour['pressure_mb'],
                'light': hour.get('uv', 0) * 100,  # Scale UV to approximate light level
                
                # Time features
                'hour': dt.hour,
                'day': dt.day,
                'month': dt.month,
                'weekday': dt.weekday(),
                'is_weekend': dt.weekday() >= 5,
                'season': ((dt.month % 12 + 3) // 3 % 4),  # 0=winter, 1=spring, 2=summer, 3=fall
                
                # Additional weather features
                'is_day': hour['is_day'],
                'cloud_cover': hour['cloud'],
                'precipitation': hour['precip_mm'],
            }
            features.append(feature_dict)
        
        # Create base dataframe
        df = pd.DataFrame(features)
        
        # Calculate features for core variables
        for col in ['temperature', 'humidity', 'pressure', 'light']:
            # Rolling means (last 3 and 6 hours)
            for window in [3, 6]:
                df[f'{col}_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            
            # Rate of change (hourly)
            df[f'{col}_change'] = df[col].diff().fillna(0)
            
            # Lag features (previous 2 hours)
            for lag in [1, 2]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag).ffill()
        
        # Fill any remaining NaN values with forward fill
        df = df.ffill()
        
        # Ensure we have exactly 90 features (pad or trim if needed)
        expected_features = 90
        current_features = len(df.columns)
        
        if current_features < expected_features:
            # Add dummy features if we have too few
            for i in range(current_features, expected_features):
                df[f'dummy_{i}'] = 0
        elif current_features > expected_features:
            # Keep only the most important features if we have too many
            core_features = [
                'temperature', 'humidity', 'pressure', 'light',  # Base features
                'hour', 'day', 'month', 'weekday', 'season', 'is_day',  # Time features
                'cloud_cover', 'precipitation'  # Weather features
            ]
            # Add derived features for core variables
            for col in ['temperature', 'humidity', 'pressure', 'light']:
                core_features.extend([
                    f'{col}_mean_3h', f'{col}_mean_6h',
                    f'{col}_std_3h', f'{col}_std_6h',
                    f'{col}_change',
                    f'{col}_lag1', f'{col}_lag2'
                ])
            df = df[core_features[:expected_features]]
        
        return df
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise


def build_15min_features(forecast_data, freq='15T'):
    """Upsample hourly forecast to the given frequency (default 15 minutes) and engineer features.

    Returns a DataFrame indexed by timestamp with the same feature columns expected by the model (90 cols).
    """
    try:
        hourly = forecast_data['forecast']['forecastday'][0]['hour']
        rows = []
        for h in hourly:
            dt = datetime.strptime(h['time'], '%Y-%m-%d %H:%M')
            rows.append({
                'date_time': dt,
                'temperature': h['temp_c'],
                'humidity': h['humidity'],
                'pressure': h['pressure_mb'],
                'light': h.get('uv', 0) * 100,
                'is_day': h['is_day'],
                'cloud_cover': h['cloud'],
                'precipitation': h['precip_mm']
            })

        df_hourly = pd.DataFrame(rows).set_index('date_time')

        # Create 15-min index spanning the same range
        start = df_hourly.index.min()
        end = df_hourly.index.max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15)
        idx = pd.date_range(start=start, end=end, freq=freq)

        # Reindex and interpolate numeric values
        df_15 = df_hourly.reindex(idx)
        numeric_cols = ['temperature', 'humidity', 'pressure', 'light', 'cloud_cover', 'precipitation']
        df_15[numeric_cols] = df_15[numeric_cols].interpolate(method='time').ffill().bfill()

        # For categorical/is_day forward-fill
        df_15['is_day'] = df_15['is_day'].ffill().bfill().astype(int)

        # Now compute the same derived features as extract_features but with appropriate window sizes
        df = df_15.reset_index().rename(columns={'index': 'date_time'})

        # reuse logic: map date parts and compute rolling windows in terms of periods
        features = []
        for _, row in df.iterrows():
            dt = row['date_time']
            features.append({
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'pressure': row['pressure'],
                'light': row['light'],
                'hour': dt.hour,
                'day': dt.day,
                'month': dt.month,
                'weekday': dt.weekday(),
                'is_weekend': dt.weekday() >= 5,
                'season': ((dt.month % 12 + 3) // 3 % 4),
                'is_day': int(row['is_day']),
                'cloud_cover': row['cloud_cover'],
                'precipitation': row['precipitation']
            })

        df_base = pd.DataFrame(features)

        # number of periods per hour for 15-min freq
        per_hour = int(pd.Timedelta(hours=1) / pd.Timedelta(freq))
        windows_hours = [3, 6]
        windows = [w * per_hour for w in windows_hours]

        for col in ['temperature', 'humidity', 'pressure', 'light']:
            for window in windows:
                df_base[f'{col}_mean_{int(window/per_hour)}h'] = df_base[col].rolling(window=window, min_periods=1).mean()
                df_base[f'{col}_std_{int(window/per_hour)}h'] = df_base[col].rolling(window=window, min_periods=1).std().fillna(0)

            df_base[f'{col}_change'] = df_base[col].diff().fillna(0)
            for lag in [1, 2]:
                df_base[f'{col}_lag{lag}'] = df_base[col].shift(lag).ffill()

        df_base = df_base.ffill().bfill()

        # Ensure 96 rows for 24 hours at 15-min frequency
        if len(df_base) < 96:
            # pad by repeating last row
            last = df_base.iloc[[-1]]
            repeats = 96 - len(df_base)
            df_base = pd.concat([df_base, pd.concat([last]*repeats, ignore_index=True)], ignore_index=True)
        elif len(df_base) > 96:
            df_base = df_base.iloc[:96].reset_index(drop=True)

        # Ensure feature count matches expected (90)
        expected_features = 90
        if df_base.shape[1] < expected_features:
            for i in range(df_base.shape[1], expected_features):
                df_base[f'dummy_{i}'] = 0
        elif df_base.shape[1] > expected_features:
            df_base = df_base.iloc[:, :expected_features]

        # attach timestamp index
        df_base.index = pd.date_range(start=start, periods=len(df_base), freq=freq)
        return df_base

    except Exception as e:
        raise RuntimeError(f"Failed to build 15-min features: {str(e)}")


def generate_next_24_hours_15min(location, api_key):
    """Generate predictions at 15-minute intervals (96 points) for the next 24 hours."""
    try:
        model, feature_scaler, target_scaler = load_model_artifacts()
        forecast_data = fetch_weather_data(location, api_key)

        df_15 = build_15min_features(forecast_data, freq='15T')

        # Scale all rows
        X_scaled = feature_scaler.transform(df_15)

        # Build sequences (sequence_length=30)
        seq_len = 30
        sequences = []
        for i in range(len(X_scaled)):
            start_idx = max(0, i - seq_len + 1)
            seq = X_scaled[start_idx:i+1]
            # pad at front if needed
            if seq.shape[0] < seq_len:
                pad = np.tile(seq[0], (seq_len - seq.shape[0], 1))
                seq = np.vstack([pad, seq])
            sequences.append(seq)
        sequences = np.array(sequences)

        preds_scaled = model.predict(sequences, verbose=0)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # Build output list with timestamps
        out = []
        times = df_15.index
        for t, p, row in zip(times, preds, df_15.itertuples(index=False)):
            # For 'actual' we use the interpolated forecast temperature as the weather API's forecast
            actual = getattr(row, 'temperature')
            out.append({
                'hour': t.strftime('%Y-%m-%d %H:%M'),
                'actual_temp': float(actual),
                'predicted_temp': float(p)
            })

        return out

    except Exception as e:
        raise RuntimeError(f"Failed to generate 15-min forecasts: {str(e)}")
        
        # Drop the datetime column as it's not needed for prediction
        df = df.drop('date_time', axis=1)
        
        return df
        
    except KeyError as e:
        raise ValueError(f"Invalid forecast data format: missing key {str(e)}")

def generate_predictions(features_df, model, feature_scaler, target_scaler):
    """
    Generate temperature predictions using the trained model
    
    Args:
        features_df (pd.DataFrame): Preprocessed weather features
        model: Trained ML model
        feature_scaler: Feature scaler used during training
        target_scaler: Target scaler used during training
    
    Returns:
        np.array: Predicted temperatures
    """
    try:
        # Add a debug print to see feature shape
        print(f"Features shape before scaling: {features_df.shape}")
        print(f"Feature columns: {features_df.columns.tolist()}")
        
        # Scale features
        scaled_features = feature_scaler.transform(features_df)
        print(f"Scaled features shape: {scaled_features.shape}")
        
        # Create sequences for LSTM (30 timesteps as per model requirements)
        sequence_length = 30
        
        # Pre-fill with the first row repeated if we don't have enough data
        if len(scaled_features) < sequence_length:
            padding = np.tile(scaled_features[0], (sequence_length - len(scaled_features), 1))
            scaled_features = np.vstack([padding, scaled_features])
        
        # Create sequences with proper shape
        sequences = []
        for i in range(len(scaled_features) - sequence_length + 1):
            seq = scaled_features[i:i + sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"Sequences shape: {sequences.shape}")
        
        # Generate predictions
        print("Generating predictions...")
        scaled_predictions = model.predict(sequences, verbose=0)
        print(f"Predictions shape: {scaled_predictions.shape}")
        
        # Inverse transform predictions
        predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
        print(f"Final predictions shape: {predictions.shape}")
        
        # We'll get a prediction for each sequence - return the last 24 predictions
        return predictions[-24:].flatten()
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions: {str(e)}")

def generate_next_24_hours(location, api_key):
    """
    Generate temperature predictions for the next 24 hours
    
    Args:
        location (str): Location identifier (city name, lat/lon, or zip code)
        api_key (str): WeatherAPI.com API key
    
    Returns:
        list: List of dictionaries containing hourly predictions
        Each dict has format: {
            "hour": "YYYY-MM-DD HH:MM",
            "actual_temp": float,
            "predicted_temp": float
        }
    """
    try:
        # Load model and scalers
        model, feature_scaler, target_scaler = load_model_artifacts()
        
        # Fetch weather forecast data
        forecast_data = fetch_weather_data(location, api_key)
        
        # Extract and preprocess features
        features_df = extract_features(forecast_data)
        
        # Generate predictions
        predictions = generate_predictions(features_df, model, feature_scaler, target_scaler)
        
        # Format results
        hourly_forecasts = []
        hour_data = forecast_data['forecast']['forecastday'][0]['hour']
        
        for i, (hour, pred_temp) in enumerate(zip(hour_data, predictions)):
            hourly_forecasts.append({
                "hour": hour['time'],
                "actual_temp": hour['temp_c'],
                "predicted_temp": float(pred_temp)
            })
        
        return hourly_forecasts
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate 24-hour forecast: {str(e)}")


def plot_forecasts(hourly_forecasts, out_path=None, show=False):
    """Plot actual vs predicted temperature for the hourly forecasts.

    Args:
        hourly_forecasts (list): Output from generate_next_24_hours
        out_path (str|Path): Where to save the plot PNG. If None, defaults to visualizations/analysis/forecast_vs_pred.png
        show (bool): If True, display the plot interactively.

    Returns:
        Path: Path to saved image (or None if not saved)
    """
    try:
        times = [datetime.strptime(h['hour'], '%Y-%m-%d %H:%M') for h in hourly_forecasts]
        actual = [h['actual_temp'] for h in hourly_forecasts]
        predicted = [h['predicted_temp'] for h in hourly_forecasts]

        plt.figure(figsize=(10, 5))
        plt.plot(times, actual, marker='o', label='Actual')
        plt.plot(times, predicted, marker='x', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title('Next 24 Hours: Actual vs Predicted Temperature')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if out_path is None:
            out_dir = Path(__file__).parent.parent.parent / 'visualizations' / 'analysis'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / 'forecast_vs_pred.png'
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(out_path)
        if show:
            plt.show()
        plt.close()
        return out_path

    except Exception as e:
        print(f"Failed to plot forecasts: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    API_KEY = "a1715ffa6e4c43a2ab8165946250411"  # Your API key
    LOCATION = "London"  # Example location
    
    try:
        forecasts = generate_next_24_hours(LOCATION, API_KEY)
        for forecast in forecasts:
            print(f"Hour: {forecast['hour']}")
            print(f"Actual Temperature: {forecast['actual_temp']}°C")
            print(f"Predicted Temperature: {forecast['predicted_temp']:.1f}°C")
            print("-" * 50)

        # Save plot
        img_path = plot_forecasts(forecasts)
        print(f"Saved plot to: {img_path}")

    except Exception as e:
        print(f"Error: {str(e)}")