
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from src.models.temperature_forecast import fetch_weather_data

API_KEY = "a1715ffa6e4c43a2ab8165946250411"
LOCATION = "Vellore, Tamil Nadu, India"
MODEL_DIR = "trained_models/lstm_temperature"

# Fetch hourly data
print("Fetching weather data...")
data = fetch_weather_data(LOCATION, API_KEY)
hourly = data['forecast']['forecastday'][0]['hour']

# Convert to 15-min DataFrame
print("Processing to 15-minute intervals...")
rows = []
for h in hourly:
    dt = pd.to_datetime(h['time'])
    base_data = {
        'time': dt,
        'temperature': h['temp_c'],
        'humidity': h['humidity'],
        'pressure': h['pressure_mb'],
        'light': h.get('uv', 0) * 100,  # Scale UV to approximate light
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'day_of_year': dt.dayofyear,
        'month': dt.month,
        'quarter': (dt.month - 1) // 3 + 1
    }
    rows.append(base_data)

# Create hourly DataFrame and resample to 15 minutes
df_hourly = pd.DataFrame(rows)
df_hourly.set_index('time', inplace=True)

# Create 15-min time range
start = df_hourly.index.min()
end = df_hourly.index.max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15)
idx_15min = pd.date_range(start=start, end=end, freq='15T')

# Resample to 15-min intervals with interpolation
df15 = df_hourly.reindex(idx_15min)
numeric_cols = ['temperature', 'humidity', 'pressure', 'light']
df15[numeric_cols] = df15[numeric_cols].interpolate(method='time')

# Add cyclical time features
df15['hour_sin'] = np.sin(2 * np.pi * df15.index.hour / 24)
df15['hour_cos'] = np.cos(2 * np.pi * df15.index.hour / 24)
df15['day_sin'] = np.sin(2 * np.pi * df15.index.day / 31)
df15['day_cos'] = np.cos(2 * np.pi * df15.index.day / 31)
df15['month_sin'] = np.sin(2 * np.pi * df15.index.month / 12)
df15['month_cos'] = np.cos(2 * np.pi * df15.index.month / 12)

# Create lag features
for col in numeric_cols:
    for lag in [1, 5, 15, 30]:
        df15[f'{col}_lag_{lag}'] = df15[col].shift(lag)

# Create rolling statistics
for col in numeric_cols:
    for window in [5, 15, 30]:
        roll = df15[col].rolling(window=window, min_periods=1)
        df15[f'{col}_roll_mean_{window}'] = roll.mean()
        df15[f'{col}_roll_std_{window}'] = roll.std()
        df15[f'{col}_roll_min_{window}'] = roll.min()
        df15[f'{col}_roll_max_{window}'] = roll.max()

# Create difference and percent change features
for col in numeric_cols:
    df15[f'{col}_diff_1'] = df15[col].diff()
    df15[f'{col}_diff_5'] = df15[col].diff(5)
    df15[f'{col}_pct_change_5'] = df15[col].pct_change(5)

# Fill any NaN values and handle infinities
df15 = df15.ffill().bfill()
df15 = df15.replace([np.inf, -np.inf], np.nan)
df15 = df15.fillna(method='ffill').fillna(method='bfill')

# Clean up any remaining problematic values
for col in df15.columns:
    if df15[col].dtype in ['float64', 'float32']:
        # Replace extreme values with column mean
        mean_val = df15[col].mean()
        std_val = df15[col].std()
        df15.loc[df15[col].abs() > mean_val + 5*std_val, col] = mean_val

# Load model and scalers
print("Loading model and scalers...")
model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"), compile=False)
scaler_features = joblib.load(os.path.join(MODEL_DIR, "scaler_features.pkl"))
scaler_target = joblib.load(os.path.join(MODEL_DIR, "scaler_target.pkl"))
config = joblib.load(os.path.join(MODEL_DIR, "model_config.pkl"))
feature_columns = config['feature_columns']
sequence_length = config['sequence_length']

# Prepare features for prediction
print("Preparing sequences for prediction...")
def create_sequences(df, sequence_length):
    features = df.values
    sequences = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i + sequence_length])
    return np.array(sequences)

# Verify we have all required features
missing_features = [col for col in feature_columns if col not in df15.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X = create_sequences(df15[feature_columns], sequence_length)
print(f"Created sequences shape: {X.shape}")

# Scale features
X_reshaped = X.reshape(-1, len(feature_columns))
X_scaled = scaler_features.transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape[0], sequence_length, len(feature_columns))

# Make predictions
print("Generating predictions...")
pred_scaled = model.predict(X_scaled, verbose=0)
pred = scaler_target.inverse_transform(pred_scaled).flatten()

# Save predictions with timestamps
print("Saving predictions...")
times = df15.index[sequence_length-1:]
indoor_df = pd.DataFrame({'time': times, 'indoor_temperature': pred})
indoor_df.to_csv('indoor_15min.csv', index=False)
print("Saved 15-min indoor temperature predictions to indoor_15min.csv")

# Print sample of predictions
print("\nSample predictions:")
print(indoor_df.head(8).to_string())
