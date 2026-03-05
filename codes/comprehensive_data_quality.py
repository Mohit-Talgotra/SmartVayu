import pandas as pd
import numpy as np

print('COMPREHENSIVE DATA QUALITY ANALYSIS - PLUS DATASET')
print('='*60)

csv_files = ['rpi_20_plus.csv', 'rpi_21_plus.csv', 'rpi_30_plus.csv', 
             'rpi_39_plus.csv', 'rpi_50_plus.csv']

# Define expected columns and reasonable ranges
expected_columns = ['id', 'date_time', 'rpi_id', 'proximity', 'humidity', 'pressure', 
                   'light', 'oxidised', 'reduced', 'nh3', 'temperature', 
                   'sound_high', 'sound_mid', 'sound_low', 'sound_amp']

# Define reasonable sensor ranges
sensor_ranges = {
    'proximity': (0, 2048),  # Typical proximity sensor range
    'humidity': (0, 100),    # Relative humidity percentage
    'pressure': (300, 1200), # Atmospheric pressure (hPa/mbar)
    'light': (0, 100000),   # Light sensor values
    'temperature': (-40, 85), # Realistic temperature range for outdoor sensors
    'oxidised': (0, 10000),   # Gas sensor values (estimated)
    'reduced': (0, 10000),    # Gas sensor values (estimated) 
    'nh3': (0, 1000),         # NH3 sensor values (estimated)
    'sound_high': (0, 10000), # Sound levels (estimated)
    'sound_mid': (0, 10000),  # Sound levels (estimated)
    'sound_low': (0, 10000),  # Sound levels (estimated)
    'sound_amp': (0, 1000),   # Sound amplitude (estimated)
}

total_issues = 0
all_problems = []

for csv_file in csv_files:
    print(f'\n{"="*50}')
    print(f'ANALYZING: {csv_file}')
    print(f'{"="*50}')
    
    try:
        df = pd.read_csv(csv_file)
        file_problems = []
        
        print(f'📊 Basic Info: {len(df):,} records')
        
        # 1. Check column structure
        print(f'\n🔍 COLUMN STRUCTURE CHECK:')
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        if missing_cols:
            print(f'❌ Missing columns: {missing_cols}')
            file_problems.append(f'Missing columns: {missing_cols}')
        else:
            print(f'✅ All expected columns present')
            
        if extra_cols:
            print(f'ℹ️  Extra columns: {extra_cols}')
        
        # 2. Check for missing values in each column
        print(f'\n📋 MISSING VALUES CHECK:')
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            print(f'❌ Total missing values: {total_missing:,}')
            for col, missing_count in missing_summary[missing_summary > 0].items():
                pct = (missing_count / len(df)) * 100
                print(f'  - {col}: {missing_count:,} ({pct:.2f}%)')
                file_problems.append(f'{col}: {missing_count} missing values ({pct:.2f}%)')
        else:
            print(f'✅ No missing values found')
        
        # 3. Check for duplicate records
        print(f'\n🔄 DUPLICATE RECORDS CHECK:')
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f'❌ Duplicate records: {duplicates:,}')
            file_problems.append(f'Duplicate records: {duplicates}')
        else:
            print(f'✅ No duplicate records')
        
        # 4. Check date_time column
        print(f'\n📅 DATE-TIME VALIDATION:')
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        invalid_dates = df['date_time'].isnull().sum()
        
        if invalid_dates > 0:
            print(f'❌ Invalid date-time values: {invalid_dates:,}')
            file_problems.append(f'Invalid dates: {invalid_dates}')
        else:
            print(f'✅ All date-time values valid')
            
        # Check for chronological order
        if not df['date_time'].is_monotonic_increasing:
            print(f'⚠️  Data is NOT in chronological order')
            file_problems.append('Data not chronologically ordered')
        else:
            print(f'✅ Data is in chronological order')
            
        # Check for time gaps within the file
        df_sorted = df.sort_values('date_time')
        time_diffs = df_sorted['date_time'].diff()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
        
        if len(large_gaps) > 0:
            print(f'⚠️  Found {len(large_gaps)} time gaps > 2 hours')
            max_gap = time_diffs.max()
            print(f'   Largest gap: {max_gap}')
            file_problems.append(f'{len(large_gaps)} time gaps > 2 hours (max: {max_gap})')
        
        # 5. Check sensor value ranges
        print(f'\n🎯 SENSOR VALUE RANGE CHECK:')
        numeric_cols = [col for col in sensor_ranges.keys() if col in df.columns]
        
        for col in numeric_cols:
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            min_val, max_val = sensor_ranges[col]
            below_range = (df[col] < min_val).sum()
            above_range = (df[col] > max_val).sum()
            nan_values = df[col].isnull().sum()
            
            if below_range > 0 or above_range > 0 or nan_values > 0:
                print(f'⚠️  {col}:')
                if nan_values > 0:
                    print(f'    - Non-numeric values: {nan_values:,}')
                    file_problems.append(f'{col}: {nan_values} non-numeric values')
                if below_range > 0:
                    print(f'    - Below range (<{min_val}): {below_range:,}')
                    print(f'    - Minimum value: {df[col].min():.2f}')
                    file_problems.append(f'{col}: {below_range} values below range')
                if above_range > 0:
                    print(f'    - Above range (>{max_val}): {above_range:,}')
                    print(f'    - Maximum value: {df[col].max():.2f}')
                    file_problems.append(f'{col}: {above_range} values above range')
            else:
                print(f'✅ {col}: All values in expected range [{min_val}, {max_val}]')
        
        # 6. Check for constant values (stuck sensors)
        print(f'\n🔧 STUCK SENSOR CHECK:')
        for col in numeric_cols:
            if df[col].nunique() == 1:
                print(f'❌ {col}: All values identical ({df[col].iloc[0]})')
                file_problems.append(f'{col}: Stuck sensor (constant value)')
            elif df[col].nunique() < 10:
                print(f'⚠️  {col}: Very few unique values ({df[col].nunique()})')
        
        # 7. Statistical outliers check
        print(f'\n📈 STATISTICAL OUTLIERS (Z-score > 4):')
        for col in numeric_cols:
            if df[col].dtype in ['float64', 'int64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 4).sum()
                if outliers > 0:
                    outlier_pct = (outliers / len(df)) * 100
                    print(f'⚠️  {col}: {outliers:,} statistical outliers ({outlier_pct:.2f}%)')
                    if outlier_pct > 1:  # Only report as problem if >1%
                        file_problems.append(f'{col}: {outliers} statistical outliers')
        
        # Store file problems
        if file_problems:
            all_problems.append({'file': csv_file, 'problems': file_problems})
            total_issues += len(file_problems)
            print(f'\n❌ TOTAL ISSUES IN THIS FILE: {len(file_problems)}')
        else:
            print(f'\n✅ NO MAJOR ISSUES FOUND IN THIS FILE')
            
    except Exception as e:
        print(f'❌ ERROR reading {csv_file}: {e}')
        all_problems.append({'file': csv_file, 'problems': [f'Read error: {e}']})
        total_issues += 1

# Final summary
print(f'\n{"="*60}')
print(f'FINAL SUMMARY REPORT')
print(f'{"="*60}')

print(f'📊 Files analyzed: {len(csv_files)}')
print(f'📊 Total data quality issues found: {total_issues}')

if total_issues == 0:
    print(f'\n🎉 EXCELLENT! No significant data quality issues found!')
    print(f'The Plus dataset appears to be very clean and ready for analysis.')
else:
    print(f'\n⚠️  ISSUES SUMMARY BY FILE:')
    for file_info in all_problems:
        print(f'\n📁 {file_info["file"]}:')
        for problem in file_info['problems']:
            print(f'  - {problem}')

print(f'\n🎯 RECOMMENDATIONS:')
if total_issues == 0:
    print(f'✅ Dataset is ready for analysis')
    print(f'✅ No data cleaning required')
    print(f'✅ Proceed with statistical analysis and modeling')
elif total_issues < 10:
    print(f'⚠️  Minor issues found - consider light data cleaning')
    print(f'⚠️  Most issues appear to be outliers rather than data corruption')
else:
    print(f'❌ Significant issues found - data cleaning recommended')
    print(f'❌ Address range violations and missing values before analysis')

print(f'\n📋 Next steps: If issues are minimal, you can proceed with temperature analysis')
print(f'similar to what we did with the reg dataset.')
