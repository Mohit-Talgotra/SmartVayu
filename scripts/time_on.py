import pandas as pd
import numpy as np
from datetime import timedelta

# Load outdoor and indoor temperature CSVs
df_out = pd.read_csv('outdoor_15min.csv')
df_in = pd.read_csv('indoor_15min.csv')

# Ensure time columns are datetime and aligned
df_out['time'] = pd.to_datetime(df_out['time'])
df_in['time'] = pd.to_datetime(df_in['time'])

# Merge on time (inner join to ensure alignment)
df = pd.merge(df_out, df_in, on='time', how='inner')

# Calculate temperature difference (outdoor - indoor)
df['diff'] = df['outdoor_temperature'] - df['indoor_temperature']

# AC ON if outdoor > indoor + 1.0
threshold = 1.0
df['ac_on'] = df['diff'] > threshold

# Calculate time intervals (in minutes)
df['interval_min'] = (df['time'].shift(-1) - df['time']).dt.total_seconds() / 60
# For last interval, fill with previous (assume regular intervals)
df['interval_min'].iloc[-1] = df['interval_min'].iloc[-2]

# Calculate total AC ON time (sum intervals where ac_on is True)
total_minutes = df.loc[df['ac_on'], 'interval_min'].sum()
hours = int(total_minutes // 60)
minutes = int(total_minutes % 60)

print(f"Total AC ON time: {hours} hours {minutes} minutes")

# Find contiguous AC ON periods
def get_ac_on_periods(df):
    periods = []
    start = None
    for i, row in df.iterrows():
        if row['ac_on'] and start is None:
            start = row['time']
        elif not row['ac_on'] and start is not None:
            end = df['time'].iloc[i-1]
            periods.append((start, end))
            start = None
    if start is not None:
        periods.append((start, df['time'].iloc[-1]))
    return periods

periods = get_ac_on_periods(df)

print("\nAC ON periods:")
for start, end in periods:
    duration = end - start + timedelta(minutes=df['interval_min'].iloc[0])
    h, m = divmod(duration.total_seconds()//60, 60)
    print(f"  {start.strftime('%H:%M')} - {end.strftime('%H:%M')}  (duration: {int(h)}h {int(m)}m)")

# Calculate area between curves (degree-minutes of cooling demand)
df['excess'] = np.where(df['ac_on'], df['diff'] - threshold, 0)
area = (df['excess'] * df['interval_min']).sum()
print(f"\nTotal cooling demand (degree-minutes above threshold): {area:.1f}")
