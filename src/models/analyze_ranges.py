import pandas as pd

# Load dataset
df = pd.read_csv(r'C:\Users\operator\Desktop\smartvayu\data\combined_plus_sensor_data.csv')

print('Dataset Analysis:')
print(f'Records: {len(df):,}')

print('\nVariable Ranges:')
for col in ['temperature', 'humidity', 'pressure', 'light']:
    min_val = df[col].min()
    max_val = df[col].max()
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f'{col.capitalize():<12}: {min_val:8.2f} to {max_val:8.2f} (mean: {mean_val:6.2f}, std: {std_val:6.2f})')

print(f'\nDate range: {df["date_time"].min()} to {df["date_time"].max()}')

# Get reasonable ranges for sliders (mean ± 3*std, but within actual min/max)
print('\nRecommended Slider Ranges:')
for col in ['humidity', 'pressure', 'light']:
    min_val = df[col].min()
    max_val = df[col].max()
    mean_val = df[col].mean()
    std_val = df[col].std()
    
    # Use 3 standard deviations from mean, but clamp to actual min/max
    slider_min = max(min_val, mean_val - 3*std_val)
    slider_max = min(max_val, mean_val + 3*std_val)
    
    print(f'{col.capitalize():<12}: {slider_min:8.2f} to {slider_max:8.2f}')