import pandas as pd
from datetime import datetime

# Constants
RATE_PER_KWH = 5  # rupees per kilowatt-hour
AC_EFFICIENCY = 0.90  # 90% efficiency
AC_POWER = 1.0  # 1 ton AC = ~3.5 kW
AC_POWER_KW = AC_POWER * 3.5  # Convert tons to kilowatts

# Load both CSVs
outdoor = pd.read_csv('outdoor_15min.csv')
indoor = pd.read_csv('indoor_15min.csv')

# Convert time columns to datetime
outdoor['time'] = pd.to_datetime(outdoor['time'])
indoor['time'] = pd.to_datetime(indoor['time'])

# Merge the two dataframes on time
df = pd.merge(outdoor, indoor, on='time', how='inner')

# Calculate when AC needs to be on (outdoor > indoor)
df['ac_needed'] = df['outdoor_temperature'] > df['indoor_temperature']

# Calculate total hours AC is not needed
total_intervals = len(df[df['ac_needed'] == False])
hours_ac_not_needed = total_intervals * 15 / 60  # convert 15-min intervals to hours

# Calculate energy and cost
hours_ac_needed = 24 - hours_ac_not_needed
energy_kwh = hours_ac_needed * AC_POWER_KW / AC_EFFICIENCY  # kWh with efficiency adjustment
cost_rupees = energy_kwh * RATE_PER_KWH

print("\nAC Cost Calculation")
print("=" * 50)
print(f"\nParameters:")
print(f"- Electricity rate: ₹{RATE_PER_KWH}/kWh")
print(f"- AC power rating: {AC_POWER} ton ({AC_POWER_KW:.1f} kW)")
print(f"- AC efficiency: {AC_EFFICIENCY*100}%")

print(f"\nCalculation:")
print(f"1. Hours when outdoor < indoor: {hours_ac_not_needed:.1f} hours")
print(f"2. Hours AC needed: {hours_ac_needed:.1f} hours")
print(f"3. Energy consumed: {energy_kwh:.2f} kWh")
print(f"   = {hours_ac_needed:.1f} hours × {AC_POWER_KW:.1f} kW ÷ {AC_EFFICIENCY}")
print(f"4. Total cost: ₹{cost_rupees:.2f}")
print(f"   = {energy_kwh:.2f} kWh × ₹{RATE_PER_KWH}/kWh")

# Print time periods when AC is needed
print("\nTime periods when AC is needed:")
current_period = None
for idx, row in df.iterrows():
    if row['ac_needed'] and current_period is None:
        current_period = row['time']
    elif not row['ac_needed'] and current_period is not None:
        end_time = df.iloc[idx-1]['time']
        duration = (end_time - current_period).total_seconds() / 3600  # convert to hours
        print(f"  {current_period.strftime('%H:%M')} - {end_time.strftime('%H:%M')}  ({duration:.1f} hours)")
        current_period = None

# Handle case if we're still in a period at the end of the day
if current_period is not None:
    end_time = df.iloc[-1]['time']
    duration = (end_time - current_period).total_seconds() / 3600
    print(f"  {current_period.strftime('%H:%M')} - {end_time.strftime('%H:%M')}  ({duration:.1f} hours)")