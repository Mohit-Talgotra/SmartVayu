import requests
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_weather_data(location, api_key):
    """Fetch hourly weather data from WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/forecast.json"
    params = {
        'key': api_key,
        'q': location,
        'days': 1,
        'aqi': 'no'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    hours = data['forecast']['forecastday'][0]['hour']
    
    measurements = []
    for hour in hours:
        measurements.append({
            'date_time': hour['time'],
            'temperature': hour['temp_c'],
            'humidity': hour['humidity'],
            'pressure': hour['pressure_mb'],
            'light': hour['uv']  # Using UV as light proxy
        })
    
    df = pd.DataFrame(measurements)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

def engineer_features(df):
    """Create the exact 90 features the model expects, in correct order."""
    # First, repeat the first row 30 times to provide history for initial sequence
    first_row = pd.concat([df.iloc[0:1]] * 30, ignore_index=True)
    first_row['date_time'] = pd.date_range(
        end=df['date_time'].iloc[0],
        periods=30,
        freq='H'
    )
    
    # Concatenate with actual data
    df = pd.concat([first_row, df], ignore_index=True)
    df = df.copy()
    
    # 1. Basic time features
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['day_of_year'] = df['date_time'].dt.dayofyear
    df['month'] = df['date_time'].dt.month
    df['quarter'] = df['date_time'].dt.quarter
    
    # 2. Cyclical time encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Lag features
    for col in ['temperature', 'humidity', 'pressure', 'light']:
        for lag in [1, 5, 15, 30]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # 4. Rolling statistics
    for col in ['temperature', 'humidity', 'pressure', 'light']:
        for window in [5, 15, 30]:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    
    # 5. Rate of change features
    for col in ['temperature', 'humidity', 'pressure', 'light']:
        df[f'{col}_diff_1'] = df[col].diff(1)
        df[f'{col}_diff_5'] = df[col].diff(5)
        df[f'{col}_pct_change_5'] = df[col].pct_change(5)
    
    # Handle NaN and infinite values from feature engineering
    for col in df.columns:
        if col != 'date_time':
            # Replace inf/-inf with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Then fill NaN values
            df[col] = df[col].ffill().bfill()
    
    return df

def load_model_artifacts(model_dir):
    """Load the trained model and preprocessing objects."""
    print("Loading model artifacts...")
    
    model = load_model(os.path.join(model_dir, "lstm_model.h5"), compile=False)
    scaler_features = joblib.load(os.path.join(model_dir, "scaler_features.pkl"))
    scaler_target = joblib.load(os.path.join(model_dir, "scaler_target.pkl"))
    config = joblib.load(os.path.join(model_dir, "model_config.pkl"))
    
    return model, scaler_features, scaler_target, config

def create_sequences(df, feature_columns, sequence_length):
    """Create sequences for LSTM prediction."""
    features = df[feature_columns].values
    sequences = []
    
    # Create sequences that use padding + actual data
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i + sequence_length])
    
    if not sequences:
        raise ValueError(f"No sequences created. Data shape: {features.shape}")
    
    return np.array(sequences)

def plot_predictions(df, predictions, save_path='predictions_vs_actual.png'):
    """Plot actual vs predicted temperatures with enhanced visualization."""
    plt.style.use('default')  # Use default style instead of seaborn
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
    
    # Get the data for plotting
    times = df['date_time'].iloc[-24:]
    actuals = df['temperature'].iloc[-24:]
    differences = predictions[-24:] - actuals
    
    # Main temperature plot
    ax1.plot(times, actuals, 
            label='API Forecast', color='#2E86C1', marker='o', 
            linewidth=2, markersize=8, alpha=0.8)
    ax1.plot(times, predictions[-24:], 
            label='Model Predictions', color='#E74C3C', marker='x', 
            linewidth=2, markersize=8, alpha=0.8)
    
    # Fill between the lines
    ax1.fill_between(times, actuals, predictions[-24:], 
                    alpha=0.2, color='gray', label='Difference')
    
    # Customize main plot
    ax1.set_title('24-Hour Temperature Forecast for Vellore\nAPI vs Model Predictions', 
                  pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.tick_params(axis='x', rotation=45)
    
    # Add temperature annotations every 6 hours
    for i in range(0, 24, 6):
        ax1.annotate(f'{actuals.iloc[i]:.1f}°C', 
                    (times.iloc[i], actuals.iloc[i]), 
                    xytext=(10, 10), textcoords='offset points')
        ax1.annotate(f'{predictions[-24:][i]:.1f}°C', 
                    (times.iloc[i], predictions[-24:][i]), 
                    xytext=(10, -15), textcoords='offset points')
    
    # Difference plot
    ax2.bar(times, differences, alpha=0.6, 
            color=['#2ECC71' if x < 0 else '#E74C3C' for x in differences])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Prediction Error (Model - API)', pad=10)
    ax2.set_ylabel('Difference (°C)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add error statistics text box
    mae = np.mean(np.abs(differences))
    rmse = np.sqrt(np.mean(differences**2))
    stats_text = f'Mean Absolute Error: {mae:.2f}°C\nRoot Mean Square Error: {rmse:.2f}°C'
    ax1.text(1.02, 0.02, stats_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {save_path}")

# Configuration
LOCATION = "Vellore, Tamil Nadu, India"
API_KEY = os.getenv('WEATHERAPI_KEY', 'a1715ffa6e4c43a2ab8165946250411')
MODEL_DIR = "trained_models/lstm_temperature"  # Updated path to correct location

print(f"\nFetching weather data for {LOCATION}...")
df = fetch_weather_data(LOCATION, API_KEY)

print("\nLoading model and preprocessing artifacts...")
model, scaler_features, scaler_target, config = load_model_artifacts(MODEL_DIR)

print("\nEngineering features...")
df_features = engineer_features(df)

# Ensure features are in correct order from model_config
feature_columns = config['feature_columns']
sequence_length = config['sequence_length']

print(f"\nCreating sequences (length={sequence_length})...")
X = create_sequences(df_features, feature_columns, sequence_length)

print("\nScaling features...")
n_samples = X.shape[0]
X_reshaped = X.reshape(-1, len(feature_columns))
X_scaled = scaler_features.transform(X_reshaped)
X_scaled = X_scaled.reshape(n_samples, sequence_length, len(feature_columns))

print("\nMaking predictions...")
predictions_scaled = model.predict(X_scaled, verbose=0)
predictions = scaler_target.inverse_transform(predictions_scaled).flatten()

# Print predictions alongside actual values
print("\nHourly Predictions vs API Forecast:")
print("-" * 60)
print("Time          API(°C)  Predicted(°C)  Difference")
print("-" * 60)

# Get only the last 24 hours (actual forecast period)
forecast_times = df['date_time'].iloc[-24:]
forecast_temps = df['temperature'].iloc[-24:]
predictions = predictions[-24:]  # Match the number of actual forecasts

for time, actual, pred in zip(forecast_times, forecast_temps, predictions):
    diff = pred - actual
    print(f"{time.strftime('%H:%M')}      {actual:6.1f}    {pred:8.1f}     {diff:+6.1f}")

print("-" * 60)

# Calculate statistics
forecast_temps = df['temperature'].iloc[-24:]  # Last 24 hours
mae = np.mean(np.abs(predictions - forecast_temps))
rmse = np.sqrt(np.mean((predictions - forecast_temps)**2))

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}°C")
print(f"Root Mean Square Error: {rmse:.2f}°C")

# Create visualization
plot_predictions(df, predictions, 'visualizations/analysis/temperature_predictions.png')