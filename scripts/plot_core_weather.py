import requests
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def fetch_core_weather(location, api_key):
    """Fetch core weather measurements for next 24 hours."""
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
            'time': hour['time'],
            'temperature': hour['temp_c'],
            'humidity': hour['humidity'],
            'pressure': hour['pressure_mb'],
            'uv': hour['uv']
        })
    
    df = pd.DataFrame(measurements)
    df['time'] = pd.to_datetime(df['time'])
    return df

def plot_weather_data(df, location, save_path='weather_measurements.png'):
    """Create a multi-axis plot of weather measurements."""
    
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Set title
    plt.title(f'24-Hour Weather Forecast - {location}\n{df["time"].dt.date.iloc[0]}', 
              pad=20, fontsize=14, fontweight='bold')

    # Temperature plot on primary axis (left)
    color1 = '#FF5733'  # Warm orange for temperature
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color=color1)
    line1 = ax1.plot(df['time'], df['temperature'], color=color1, 
                     label='Temperature', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Format x-axis to show hours
    ax1.set_xlim(df['time'].iloc[0], df['time'].iloc[-1])
    plt.xticks(rotation=45)
    
    # Humidity on the same axis but different color
    color2 = '#3498DB'  # Cool blue for humidity
    line2 = ax1.plot(df['time'], df['humidity'], color=color2, 
                     label='Humidity %', linewidth=2, linestyle='--')
    
    # Pressure on secondary y-axis (right)
    ax2 = ax1.twinx()
    color3 = '#2ECC71'  # Green for pressure
    ax2.set_ylabel('Pressure (mb)', color=color3)
    line3 = ax2.plot(df['time'], df['pressure'], color=color3, 
                     label='Pressure', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color3)
    
    # UV Index on tertiary axis (far right)
    ax3 = ax1.twinx()
    # Offset the third axis
    ax3.spines['right'].set_position(('outward', 60))
    color4 = '#9B59B6'  # Purple for UV
    ax3.set_ylabel('UV Index', color=color4)
    line4 = ax3.plot(df['time'], df['uv'], color=color4, 
                     label='UV Index', linewidth=2, linestyle=':')
    ax3.tick_params(axis='y', labelcolor=color4)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Combine all lines for the legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as: {save_path}")
    
    return fig

# Configuration
LOCATION = "Vellore, Tamil Nadu, India"
API_KEY = os.getenv('WEATHERAPI_KEY', 'a1715ffa6e4c43a2ab8165946250411')
SAVE_PATH = "visualizations/analysis/weather_measurements.png"

# Create visualizations directory if it doesn't exist
os.makedirs("visualizations/analysis", exist_ok=True)

# Fetch data
print(f"\nFetching weather data for {LOCATION}...")
df = fetch_core_weather(LOCATION, API_KEY)

# Create and save the plot
fig = plot_weather_data(df, LOCATION, SAVE_PATH)

# Print statistics
print("\nSummary Statistics:")
print(f"Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
print(f"Humidity range: {df['humidity'].min():.0f}% to {df['humidity'].max():.0f}%")
print(f"Pressure range: {df['pressure'].min():.1f} to {df['pressure'].max():.1f} mb")
print(f"UV Index range: {df['uv'].min():.1f} to {df['uv'].max():.1f}")

# Display plot
plt.show()