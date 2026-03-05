import sys
from pathlib import Path
# Ensure project root is on sys.path so `src` package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.temperature_forecast import fetch_weather_data, build_15min_features

API_KEY = "a1715ffa6e4c43a2ab8165946250411"
# Vellore, Tamil Nadu, India
LOCATION = "Vellore, Tamil Nadu, India"

if __name__ == '__main__':
    print(f"Fetching hourly forecast for: {LOCATION}")
    data = fetch_weather_data(LOCATION, API_KEY)
    print("Building 15-minute features...")
    df15 = build_15min_features(data, freq='15T')
    print(f"Result shape: {df15.shape}")
    print("Columns (count={}):".format(len(df15.columns)))
    for i, c in enumerate(df15.columns):
        print(f"{i+1:03d}: {c}")
    print("\nFirst 12 rows:")
    print(df15.head(12).to_string())

    # Save 15-min outdoor temperature to CSV (use index as time column)
    df15_reset = df15.reset_index().rename(columns={'index': 'time'})
    out_temp = df15_reset[['time', 'temperature']].copy()
    out_temp = out_temp.rename(columns={'temperature': 'outdoor_temperature'})
    out_temp.to_csv('outdoor_15min.csv', index=False)
    print("\nSaved 15-min outdoor temperature to outdoor_15min.csv")

    # Optionally, create a placeholder for indoor temperature (to be filled by model)
    # pd.DataFrame({'time': out_temp['time'], 'indoor_temperature': np.nan}).to_csv('indoor_15min.csv', index=False)
