import requests
from datetime import datetime
import pandas as pd
import os

def fetch_core_weather(location, api_key):
    """Fetch core weather measurements for next 24 hours."""
    
    # Forecast API endpoint
    url = f"http://api.weatherapi.com/v1/forecast.json"
    
    # Request parameters
    params = {
        'key': api_key,
        'q': location,
        'days': 1,  # 1 day forecast
        'aqi': 'no'  # we don't need air quality data
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract hourly forecasts
    hours = data['forecast']['forecastday'][0]['hour']
    
    # Create a list to store the measurements
    measurements = []
    
    for hour in hours:
        measurements.append({
            'time': hour['time'],
            'temperature': hour['temp_c'],
            'humidity': hour['humidity'],
            'pressure': hour['pressure_mb'],
            'uv': hour['uv']  # using UV as light approximation
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(measurements)
    df['time'] = pd.to_datetime(df['time'])
    
    return df

# Configuration
LOCATION = "Vellore, Tamil Nadu, India"
API_KEY = os.getenv('WEATHERAPI_KEY')

if not API_KEY:
    print("Error: WEATHERAPI_KEY environment variable not set")
    exit(1)

# Fetch data
print(f"\nFetching weather data for {LOCATION}...")
df = fetch_core_weather(LOCATION, API_KEY)

# Print the data nicely
print(f"\nCore Weather Measurements for next 24 hours:")
print(f"Location: {LOCATION}")
print(f"Date: {df['time'].dt.date.iloc[0]}")
print("\nHourly measurements:")
print("-" * 80)
print("Time          Temp(°C)  Humidity(%)  Pressure(mb)  UV Index")
print("-" * 80)

for _, row in df.iterrows():
    time_str = row['time'].strftime('%H:%M')
    print(f"{time_str}      {row['temperature']:6.1f}    {row['humidity']:6.0f}     {row['pressure']:8.1f}    {row['uv']:6.1f}")

print("-" * 80)

# Print some statistics
print("\nSummary Statistics:")
print(f"Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
print(f"Humidity range: {df['humidity'].min():.0f}% to {df['humidity'].max():.0f}%")
print(f"Pressure range: {df['pressure'].min():.1f} to {df['pressure'].max():.1f} mb")
print(f"UV Index range: {df['uv'].min():.1f} to {df['uv'].max():.1f}")