import pandas as pd
import matplotlib.pyplot as plt

# Load both CSVs
outdoor = pd.read_csv('outdoor_15min.csv')
indoor = pd.read_csv('indoor_15min.csv')

# Convert time columns to datetime
outdoor['time'] = pd.to_datetime(outdoor['time'])
indoor['time'] = pd.to_datetime(indoor['time'])

# Merge the two dataframes on time
df = pd.merge(outdoor, indoor, on='time', how='inner')

# Count minutes where outdoor < indoor
df['outdoor_cooler'] = df['outdoor_temperature'] < df['indoor_temperature']

# Each row represents 15 minutes
minutes_outdoor_cooler = df['outdoor_cooler'].sum() * 15

print(f"\nBasic temperature comparison:")
print(f"Minutes where outdoor < indoor: {minutes_outdoor_cooler} minutes")
print(f"Hours where outdoor < indoor: {minutes_outdoor_cooler/60:.1f} hours")

# Create a plot of both temperatures
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['outdoor_temperature'], label='Outdoor Temperature', color='red')
plt.plot(df['time'], df['indoor_temperature'], label='Indoor Temperature', color='blue')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Indoor vs Outdoor Temperature Comparison')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('temperature_comparison.png')
plt.close()

print("\nPlot saved as: temperature_comparison.png")

print("\nTime periods where outdoor < indoor:")
# Print the actual time periods
current_period = None
for idx, row in df.iterrows():
    if row['outdoor_cooler'] and current_period is None:
        current_period = row['time']
    elif not row['outdoor_cooler'] and current_period is not None:
        end_time = df.iloc[idx-1]['time']
        duration = (end_time - current_period).total_seconds() / 60 + 15  # add 15 for the last interval
        print(f"  {current_period.strftime('%H:%M')} - {end_time.strftime('%H:%M')}  ({int(duration)} minutes)")
        current_period = None

# Handle case if we're still in a period at the end of the day
if current_period is not None:
    end_time = df.iloc[-1]['time']
    duration = (end_time - current_period).total_seconds() / 60 + 15
    print(f"  {current_period.strftime('%H:%M')} - {end_time.strftime('%H:%M')}  ({int(duration)} minutes)")