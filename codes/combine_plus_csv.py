#!/usr/bin/env python3
"""
Combine Plus CSV Files Script
=============================
Combines all plus CSV files into one chronologically sorted dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def combine_plus_files():
    """Combine all plus CSV files into one chronologically ordered file"""
    
    print("COMBINING PLUS CSV FILES")
    print("=" * 50)
    
    # Define the CSV files to combine
    csv_files = ['rpi_20_plus.csv', 'rpi_21_plus.csv', 'rpi_30_plus.csv', 
                 'rpi_39_plus.csv', 'rpi_50_plus.csv']
    
    # Expected columns for Plus dataset
    expected_columns = ['id', 'date_time', 'rpi_id', 'proximity', 'humidity', 'pressure', 
                       'light', 'oxidised', 'reduced', 'nh3', 'temperature', 
                       'sound_high', 'sound_mid', 'sound_low', 'sound_amp']
    
    # Initialize list to store all dataframes
    all_dataframes = []
    total_records = 0
    
    print(f"Processing {len(csv_files)} files...")
    print()
    
    # Read each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing: {csv_file}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            print(f"  - Records loaded: {len(df):,}")
            print(f"  - Columns: {list(df.columns)}")
            
            # Verify column structure
            missing_cols = set(expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_columns)
            
            if missing_cols:
                print(f"  - ⚠️  Missing columns: {missing_cols}")
            if extra_cols:
                print(f"  - ℹ️  Extra columns: {extra_cols}")
            
            # Convert date_time to datetime
            df['date_time'] = pd.to_datetime(df['date_time'])
            
            # Show date range
            date_min = df['date_time'].min()
            date_max = df['date_time'].max()
            print(f"  - Date range: {date_min} to {date_max}")
            
            # Add to list
            all_dataframes.append(df)
            total_records += len(df)
            
            print(f"  - ✅ Successfully processed")
            
        except Exception as e:
            print(f"  - ❌ Error processing {csv_file}: {e}")
            continue
        
        print()
    
    if not all_dataframes:
        print("❌ No files were successfully processed!")
        return
    
    print("COMBINING AND SORTING DATA")
    print("=" * 30)
    
    # Combine all dataframes
    print("Concatenating all data...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined records: {len(combined_df):,}")
    
    # Sort by date_time
    print("Sorting by date_time...")
    combined_df = combined_df.sort_values('date_time').reset_index(drop=True)
    
    # Show final statistics
    print(f"Final record count: {len(combined_df):,}")
    print(f"Date range: {combined_df['date_time'].min()} to {combined_df['date_time'].max()}")
    print(f"RPI IDs: {sorted(combined_df['rpi_id'].unique())}")
    
    # Check for any issues in combined data
    print("\nDATA QUALITY CHECK")
    print("=" * 20)
    
    # Check for missing values
    missing_counts = combined_df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"⚠️  Missing values found: {total_missing}")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  - {col}: {count}")
    else:
        print("✅ No missing values")
    
    # Check for duplicates
    duplicates = combined_df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  Duplicate records: {duplicates}")
    else:
        print("✅ No duplicate records")
    
    # Show monthly distribution
    print("\nMONTHLY DISTRIBUTION")
    print("=" * 22)
    monthly_counts = combined_df.groupby(combined_df['date_time'].dt.to_period('M')).size()
    for period, count in monthly_counts.items():
        print(f"{period}: {count:,} records")
    
    # Show RPI distribution
    print("\nRPI DISTRIBUTION")
    print("=" * 18)
    rpi_counts = combined_df.groupby('rpi_id').size().sort_index()
    for rpi_id, count in rpi_counts.items():
        print(f"RPI {rpi_id}: {count:,} records")
    
    # Save combined file
    output_filename = 'combined_plus_sensor_data.csv'
    print(f"\nSAVING COMBINED FILE")
    print("=" * 22)
    print(f"Output file: {output_filename}")
    
    try:
        combined_df.to_csv(output_filename, index=False)
        print(f"✅ Successfully saved {len(combined_df):,} records")
        
        # Verify the saved file
        verification_df = pd.read_csv(output_filename, nrows=5)
        print(f"✅ File verification: {len(verification_df.columns)} columns, sample loaded")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Final summary
    print(f"\n{'=' * 50}")
    print("COMBINATION COMPLETE!")
    print(f"{'=' * 50}")
    print(f"📊 Input files: {len(csv_files)}")
    print(f"📊 Total records: {len(combined_df):,}")
    print(f"📊 Date span: {(combined_df['date_time'].max() - combined_df['date_time'].min()).days} days")
    print(f"📊 Output file: {output_filename}")
    print(f"📊 File size: ~{len(combined_df) * len(combined_df.columns) * 8 / 1024 / 1024:.1f} MB (estimated)")
    
    print(f"\n🎯 The combined Plus dataset is ready for analysis!")
    print(f"   - Chronologically sorted from {combined_df['date_time'].min().date()}")
    print(f"   - to {combined_df['date_time'].max().date()}")
    print(f"   - Including data from RPI devices: {', '.join(map(str, sorted(combined_df['rpi_id'].unique())))}")

if __name__ == "__main__":
    combine_plus_files()
