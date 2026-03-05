import pandas as pd
import numpy as np

print('CHECKING FOR MISSING TEMPERATURE VALUES IN PLUS FILES')
print('='*60)

csv_files = ['rpi_20_plus.csv', 'rpi_21_plus.csv', 'rpi_30_plus.csv', 
             'rpi_39_plus.csv', 'rpi_50_plus.csv']

total_records = 0
total_missing = 0
file_summaries = []

for csv_file in csv_files:
    print(f'\nFile: {csv_file}')
    print('-' * 40)
    
    try:
        df = pd.read_csv(csv_file)
        total_records += len(df)
        
        # Check basic info
        print(f'Total records: {len(df):,}')
        
        # Convert date_time for range analysis
        df['date_time'] = pd.to_datetime(df['date_time'])
        print(f'Date range: {df["date_time"].min().date()} to {df["date_time"].max().date()}')
        
        # Check temperature column for missing values
        temp_missing = df['temperature'].isna().sum()
        temp_empty_strings = (df['temperature'] == '').sum()
        temp_null_strings = (df['temperature'] == 'null').sum()
        temp_nan_strings = (df['temperature'] == 'NaN').sum()
        
        # Check for non-numeric values
        df_temp_copy = df['temperature'].copy()
        df_temp_numeric = pd.to_numeric(df_temp_copy, errors='coerce')
        temp_non_numeric = df_temp_numeric.isna().sum() - temp_missing
        
        total_temp_issues = temp_missing + temp_empty_strings + temp_null_strings + temp_nan_strings + temp_non_numeric
        total_missing += total_temp_issues
        
        print(f'Temperature analysis:')
        print(f'  - NaN/missing values: {temp_missing:,}')
        print(f'  - Empty strings: {temp_empty_strings:,}')
        print(f'  - "null" strings: {temp_null_strings:,}')
        print(f'  - "NaN" strings: {temp_nan_strings:,}')
        print(f'  - Non-numeric values: {temp_non_numeric:,}')
        print(f'  - TOTAL ISSUES: {total_temp_issues:,} ({total_temp_issues/len(df)*100:.2f}%)')
        
        # Check temperature value ranges
        valid_temps = df_temp_numeric.dropna()
        if len(valid_temps) > 0:
            print(f'Valid temperature stats:')
            print(f'  - Count: {len(valid_temps):,}')
            print(f'  - Range: {valid_temps.min():.2f}°C to {valid_temps.max():.2f}°C')
            print(f'  - Mean: {valid_temps.mean():.2f}°C')
            
            # Check for outliers (beyond reasonable temperature ranges)
            extreme_low = (valid_temps < -40).sum()
            extreme_high = (valid_temps > 100).sum()
            if extreme_low > 0 or extreme_high > 0:
                print(f'  - Extreme outliers: {extreme_low} below -40°C, {extreme_high} above 100°C')
        
        # Store summary for later
        file_summaries.append({
            'file': csv_file,
            'total_records': len(df),
            'missing_temp': total_temp_issues,
            'valid_temp': len(valid_temps),
            'date_range': f"{df['date_time'].min().date()} to {df['date_time'].max().date()}"
        })
        
    except Exception as e:
        print(f'Error reading {csv_file}: {e}')
        file_summaries.append({
            'file': csv_file,
            'error': str(e)
        })

print('\n' + '='*60)
print('SUMMARY REPORT')
print('='*60)

print(f'Total records across all files: {total_records:,}')
print(f'Total temperature issues: {total_missing:,}')
print(f'Overall temperature issue rate: {total_missing/total_records*100:.2f}%')

print(f'\nFile-by-file summary:')
print(f'{"File":<20} {"Records":<10} {"Missing Temp":<12} {"Valid Temp":<12} {"Issue %":<8}')
print('-' * 70)

for summary in file_summaries:
    if 'error' not in summary:
        issue_pct = summary['missing_temp'] / summary['total_records'] * 100
        print(f"{summary['file']:<20} {summary['total_records']:<10,} {summary['missing_temp']:<12,} {summary['valid_temp']:<12,} {issue_pct:<8.2f}%")
    else:
        print(f"{summary['file']:<20} ERROR: {summary['error']}")

print(f'\nRecommendations:')
if total_missing == 0:
    print('✅ No temperature data issues found! Data quality is excellent.')
elif total_missing / total_records < 0.01:  # Less than 1%
    print('✅ Very low temperature data issues (<1%). Data quality is very good.')
elif total_missing / total_records < 0.05:  # Less than 5%
    print('⚠️  Some temperature data issues (1-5%). Consider data cleaning.')
else:
    print('❌ Significant temperature data issues (>5%). Data cleaning required.')

print(f'- For analysis, consider removing or interpolating missing values')
print(f'- Check for systematic patterns in missing data (by time/device)')
print(f'- Validate extreme temperature values for sensor malfunctions')
