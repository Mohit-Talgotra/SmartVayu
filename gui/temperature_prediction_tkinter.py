"""
Temperature Prediction GUI (Tkinter)
===================================
A simple offline GUI for the LSTM temperature prediction model.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from temperature_lstm_model.predict_temperature import TemperaturePredictor
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    # Patch the TemperaturePredictor to load with compile=False and recompile
    class PatchedTemperaturePredictor(TemperaturePredictor):
        def __init__(self, model_dir="temperature_lstm_model"):
            print("Loading trained temperature prediction model (patched)...")
            self.model = load_model(f"{model_dir}/lstm_model.h5", compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            import joblib
            self.scaler_features = joblib.load(f"{model_dir}/scaler_features.pkl")
            self.scaler_target = joblib.load(f"{model_dir}/scaler_target.pkl")
            self.config = joblib.load(f"{model_dir}/model_config.pkl")
            self.feature_columns = self.config['feature_columns']
            self.sequence_length = self.config['sequence_length']
            print(f"Model loaded successfully!")
            print(f"Sequence length: {self.sequence_length} time steps")
            print(f"Features required: {len(self.feature_columns)}")
    predictor = PatchedTemperaturePredictor()
except Exception as e:
    predictor = None
    error_msg = str(e)

# Advice for input ranges
ADVICE = {
    'humidity': 'Typical: 20-80%',
    'pressure': 'Typical: 950-1050 hPa',
    'light': 'Typical: 0-100000 lux',
    'temperature': 'Typical: -10 to 40°C'
}

# Main window
root = tk.Tk()
root.title("Temperature Predictor (Offline)")
root.geometry("400x400")
root.configure(bg="#f0f4f8")

# Title
title = tk.Label(root, text="Temperature Prediction", font=("Arial", 18, "bold"), bg="#f0f4f8", fg="#2a4d69")
title.pack(pady=10)

# Input frame
frame = tk.Frame(root, bg="#f0f4f8")
frame.pack(pady=10)

fields = {}
labels = [
    ("Humidity (%)", "humidity", 50.0),
    ("Pressure (hPa)", "pressure", 1013.0),
    ("Light Level (lux)", "light", 1000.0),
    ("Current Temperature (°C)", "temperature", 25.0)
]

for idx, (label, key, default) in enumerate(labels):
    lbl = tk.Label(frame, text=label, font=("Arial", 12), bg="#f0f4f8")
    lbl.grid(row=idx, column=0, sticky="w", padx=5, pady=5)
    entry = tk.Entry(frame, font=("Arial", 12))
    entry.insert(0, str(default))
    entry.grid(row=idx, column=1, padx=5, pady=5)
    advice = tk.Label(frame, text=ADVICE[key], font=("Arial", 10, "italic"), bg="#f0f4f8", fg="#4f8a8b")
    advice.grid(row=idx, column=2, sticky="w", padx=5)
    fields[key] = entry

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Arial", 14), bg="#f0f4f8", fg="#2a4d69")
result_label.pack(pady=10)

# Prediction function
def predict():
    if not predictor:
        messagebox.showerror("Model Error", f"Could not load model: {error_msg}")
        return
    try:
        humidity = float(fields['humidity'].get())
        pressure = float(fields['pressure'].get())
        light = float(fields['light'].get())
        temperature = float(fields['temperature'].get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")
        return
    # Create 30 timesteps with current values
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(29, -1, -1)]
    data = pd.DataFrame({
        'date_time': timestamps,
        'humidity': [humidity] * 30,
        'pressure': [pressure] * 30,
        'light': [light] * 30,
        'temperature': [temperature] * 30
    })
    try:
        prediction = predictor.predict_next_temperature(data)
        temp_change = prediction - temperature
        change_text = "increase" if temp_change > 0 else "decrease"
        result_var.set(f"Predicted next temperature: {prediction:.2f}°C\nExpected {change_text}: {abs(temp_change):.2f}°C")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")

# Run button
run_btn = tk.Button(root, text="Predict", font=("Arial", 14, "bold"), bg="#4f8a8b", fg="white", command=predict)
run_btn.pack(pady=20)

# Footer
footer = tk.Label(root, text="Made with Tkinter and TensorFlow", font=("Arial", 10), bg="#f0f4f8", fg="#888")
footer.pack(side="bottom", pady=10)

root.mainloop()
