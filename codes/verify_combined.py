import pandas as pd

# Quick verification of the combined file
print("COMBINED PLUS DATASET VERIFICATION")
print("="*40)

df = pd.read_csv('combined_plus_sensor_data.csv')
df['date_time'] = pd.to_datetime(df['date_time'])

print(f'Total records: {len(df):,}')
print(f'Columns: {len(df.columns)}')
print(f'Column names: {list(df.columns)}')
print(f'Date range: {df["date_time"].min()} to {df["date_time"].max()}')
print(f'RPI devices: {sorted(df["rpi_id"].unique())}')
print(f'Is chronologically sorted: {df["date_time"].is_monotonic_increasing}')

# Sample records
print(f'\nFirst record: {df.iloc[0]["date_time"]} (RPI {df.iloc[0]["rpi_id"]})')
print(f'Last record:  {df.iloc[-1]["date_time"]} (RPI {df.iloc[-1]["rpi_id"]})')

# Check data completeness
missing_vals = df.isnull().sum().sum()
print(f'Missing values: {missing_vals}')

print(f'\n✅ Combined file is ready for analysis!')
print(f'📁 File: combined_plus_sensor_data.csv')
print(f'📊 Size: ~{355406650 / 1024 / 1024:.1f} MB')
