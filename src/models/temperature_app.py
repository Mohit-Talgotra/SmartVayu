import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from temperature_lstm_model.predict_temperature import TemperaturePredictor

# Initialize predictor and config
predictor = TemperaturePredictor(model_dir="temperature_lstm_model")
SEQUENCE_LENGTH = predictor.sequence_length
DATA_FILE = os.path.join("data", "combined_plus_sensor_data.csv")

# Folder to save prediction records
records_folder = "prediction_records"
os.makedirs(records_folder, exist_ok=True)

def create_input_df(date_time, humidity, pressure, light):
    df = pd.DataFrame({
        'date_time': [date_time],
        'humidity': [humidity],
        'pressure': [pressure],
        'light': [light],
        'temperature': [np.nan]
    })
    return df

def load_recent_history(target_dt: datetime, required_rows: int) -> pd.DataFrame:
    """Load last required_rows of history up to target_dt from the dataset."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)
    df = df[df['date_time'] <= target_dt]
    if len(df) == 0:
        raise ValueError("No historical data available before the selected datetime.")
    return df.tail(required_rows)

# Function to save prediction record

def save_prediction_record(df):
    date_str = df['date_time'].iloc[0].strftime('%Y-%m-%d')
    file_path = os.path.join(records_folder, f"predictions_{date_str}.csv")
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

# Streamlit UI
st.title("Temperature Prediction Interface")

# Input current situation variables
col1, col2 = st.columns(2)

with col1:
    date_time = st.date_input("Date", value=datetime.now().date())
    time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))

with col2:
    humidity = st.selectbox("Humidity (%)", options=list(range(0, 101)), index=50)
    pressure = st.selectbox("Pressure (hPa)", options=list(range(950, 1051)), index=50)
    light = st.selectbox("Light (lux)", options=list(range(0, 1001, 10)), index=500)

# Combine date and time
input_datetime = datetime.combine(date_time, time)

# Prepare window: history + current input
try:
    history_df = load_recent_history(input_datetime, SEQUENCE_LENGTH - 1)
    input_df = create_input_df(input_datetime, humidity, pressure, light)
    window_df = pd.concat([history_df, input_df], ignore_index=True)
    if len(window_df) < SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient history. Need {SEQUENCE_LENGTH} rows, have {len(window_df)}.")
    predicted_temp = predictor.predict_next_temperature(window_df)
except Exception as e:
    st.error(f"Prediction error: {e}")
    predicted_temp = None

if predicted_temp is not None:
    st.write(f"### Predicted Temperature: {predicted_temp:.2f} °C")

# Feature to predict for the rest of the day at 5-minute intervals

if st.button("Predict for rest of the day"):
    # Generate datetime range from next 5 min to end of day
    start_dt = input_datetime + timedelta(minutes=5)
    end_dt = datetime.combine(input_datetime.date(), datetime.max.time())
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='5min')

    # Initialize records list
    records = []

    # Start with the same history window
    rolling_window = window_df.copy()
    current_humidity = humidity
    current_pressure = pressure
    current_light = light
    current_temp = predicted_temp

    # Generate predictions iteratively
    for dt in date_range:
        # Create input df for prediction
        df_pred = pd.DataFrame({
            'date_time': [dt],
            'humidity': [current_humidity],
            'pressure': [current_pressure],
            'light': [current_light],
            'temperature': [current_temp]
        })
        # Predict next temperature using rolling window
        try:
            rolling_window = pd.concat([rolling_window, df_pred], ignore_index=True).tail(SEQUENCE_LENGTH)
            next_temp = predictor.predict_next_temperature(rolling_window)
        except Exception as e:
            st.error(f"Prediction error at {dt}: {e}")
            break
        # Save record
        records.append({
            'date_time': dt,
            'humidity': current_humidity,
            'pressure': current_pressure,
            'light': current_light,
            'temperature': next_temp
        })
        # Update current temperature for next iteration
        current_temp = next_temp

    # Save all records
    if records:
        df_records = pd.DataFrame(records)
        save_prediction_record(df_records)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_records, x='date_time', y='temperature', ax=ax)
        ax.set_title('Temperature Predictions for Rest of the Day')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Show table
        st.dataframe(df_records)

# Summary

# This Streamlit app provides a clean UI with dropdowns for humidity, pressure, light, and date/time inputs.
# It uses the existing LSTM model to predict temperature based on the current inputs.
# The "Predict for rest of the day" button iteratively predicts temperature at 5-minute intervals for the rest of the day,
# saving all predictions in CSV files under the 'prediction_records' folder.
# The app displays the prediction graph and table without blocking the flow.


