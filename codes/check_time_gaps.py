import pandas as pd

print('CHECKING FOR TIME GAPS IN PLUS FILES')
print('='*45)

csv_files = ['rpi_20_plus.csv', 'rpi_21_plus.csv', 'rpi_30_plus.csv', 
             'rpi_39_plus.csv', 'rpi_50_plus.csv']

for csv_file in csv_files:
    print(f'\nFile: {csv_file}')
    print('-' * 30)
    
    try:
        df = pd.read_csv(csv_file)
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        # Overall stats
        print(f'Total records: {len(df):,}')
        print(f'Date range: {df["date_time"].min().date()} to {df["date_time"].max().date()}')
        
        # Check if this RPI has 2022 data and look for gaps
        data_2022 = df[df['date_time'].dt.year == 2022]
        if len(data_2022) > 0:
            print(f'2022 records: {len(data_2022):,}')
            print(f'2022 range: {data_2022["date_time"].min().date()} to {data_2022["date_time"].max().date()}')
            
            # Check each month in 2022 for gaps
            print('2022 monthly data:')
            has_gap = False
            for month in range(1, 8):
                month_data = data_2022[data_2022['date_time'].dt.month == month]
                if len(month_data) > 0:
                    first = month_data['date_time'].min().date()
                    last = month_data['date_time'].max().date()
                    print(f'  2022-{month:02d}: {len(month_data):,} records ({first} to {last})')
                else:
                    print(f'  2022-{month:02d}: NO DATA')
                    has_gap = True
            
            if has_gap:
                print('  ⚠️  TIME GAP DETECTED')
            else:
                print('  ✅  No gaps in 2022')
        else:
            print('No 2022 data in this file')
            
    except Exception as e:
        print(f'Error reading {csv_file}: {e}')

print('\n' + '='*45)
print('TIME GAP ANALYSIS SUMMARY')
print('='*45)
print('This analysis shows whether the plus dataset has')
print('the same time gaps as the reg dataset we analyzed earlier.')
