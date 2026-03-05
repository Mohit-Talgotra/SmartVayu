#!/usr/bin/env python3
"""
Temperature Prediction GUI Application
=====================================
Simple Tkinter interface to use the trained temperature prediction model.

Features:
- Input fields for day, time, humidity, pressure, season
- Real-time prediction display
- Model confidence information
- Clean, user-friendly interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import traceback

class TemperaturePredictionGUI:
    """GUI application for temperature prediction"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Temperature Prediction System")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Model artifacts
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.model_info = None
        
        # GUI variables
        self.day_var = tk.StringVar()
        self.hour_var = tk.StringVar()
        self.minute_var = tk.StringVar()
        self.humidity_var = tk.StringVar()
        self.pressure_var = tk.StringVar()
        self.season_var = tk.StringVar()
        self.prediction_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        
        # Initialize GUI
        self.create_gui()
        self.load_model_artifacts()
        
    def create_gui(self):
        """Create the GUI interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🌡️ Temperature Prediction System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="15")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Day of Year
        ttk.Label(input_frame, text="Day of Year (1-365):").grid(row=0, column=0, sticky=tk.W, pady=5)
        day_entry = ttk.Entry(input_frame, textvariable=self.day_var, width=15)
        day_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Time input
        ttk.Label(input_frame, text="Time (HH:MM):").grid(row=1, column=0, sticky=tk.W, pady=5)
        time_frame = ttk.Frame(input_frame)
        time_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        hour_spinbox = ttk.Spinbox(time_frame, from_=0, to=23, textvariable=self.hour_var, 
                                  width=5, format="%02.0f")
        hour_spinbox.grid(row=0, column=0)
        ttk.Label(time_frame, text=":").grid(row=0, column=1, padx=5)
        minute_spinbox = ttk.Spinbox(time_frame, from_=0, to=59, increment=15, 
                                    textvariable=self.minute_var, width=5, format="%02.0f")
        minute_spinbox.grid(row=0, column=2)
        
        # Humidity
        ttk.Label(input_frame, text="Humidity (%):").grid(row=2, column=0, sticky=tk.W, pady=5)
        humidity_entry = ttk.Entry(input_frame, textvariable=self.humidity_var, width=15)
        humidity_entry.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Pressure
        ttk.Label(input_frame, text="Pressure (hPa):").grid(row=3, column=0, sticky=tk.W, pady=5)
        pressure_entry = ttk.Entry(input_frame, textvariable=self.pressure_var, width=15)
        pressure_entry.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Season
        ttk.Label(input_frame, text="Season:").grid(row=4, column=0, sticky=tk.W, pady=5)
        season_combo = ttk.Combobox(input_frame, textvariable=self.season_var, 
                                   values=["1 - Winter", "2 - Spring", "3 - Summer", "4 - Fall"],
                                   state="readonly", width=12)
        season_combo.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Predict button
        predict_btn = ttk.Button(input_frame, text="🔮 Predict Temperature", 
                               command=self.predict_temperature)
        predict_btn.grid(row=5, column=0, columnspan=2, pady=(15, 5))
        
        # Results section
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Prediction display
        ttk.Label(result_frame, text="Predicted Temperature:", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=5)
        prediction_label = ttk.Label(result_frame, textvariable=self.prediction_var, 
                                    font=('Arial', 14, 'bold'), foreground='blue')
        prediction_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Confidence display
        ttk.Label(result_frame, text="Model Confidence:", font=('Arial', 10)).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        confidence_label = ttk.Label(result_frame, textvariable=self.confidence_var, 
                                   font=('Arial', 10), foreground='green')
        confidence_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Quick input buttons
        quick_frame = ttk.LabelFrame(main_frame, text="Quick Input Examples", padding="10")
        quick_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Button(quick_frame, text="🌅 Morning", command=lambda: self.set_quick_input("morning")).grid(
            row=0, column=0, padx=5, pady=2)
        ttk.Button(quick_frame, text="🌞 Noon", command=lambda: self.set_quick_input("noon")).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(quick_frame, text="🌆 Evening", command=lambda: self.set_quick_input("evening")).grid(
            row=0, column=2, padx=5, pady=2)
        ttk.Button(quick_frame, text="🌙 Night", command=lambda: self.set_quick_input("night")).grid(
            row=0, column=3, padx=5, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load model and make predictions")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Set default values
        self.set_default_values()
        
    def set_default_values(self):
        """Set default input values"""
        current_date = datetime.now()
        self.day_var.set(str(current_date.timetuple().tm_yday))
        self.hour_var.set(f"{current_date.hour:02d}")
        self.minute_var.set("00")
        self.humidity_var.set("70.0")
        self.pressure_var.set("1013.25")
        
        # Set season based on current month
        season_map = {1: "1 - Winter", 2: "1 - Winter", 3: "2 - Spring", 4: "2 - Spring", 
                     5: "2 - Spring", 6: "3 - Summer", 7: "3 - Summer", 8: "3 - Summer",
                     9: "4 - Fall", 10: "4 - Fall", 11: "4 - Fall", 12: "1 - Winter"}
        self.season_var.set(season_map.get(current_date.month, "2 - Spring"))
        
    def set_quick_input(self, time_type):
        """Set quick input examples"""
        current_date = datetime.now()
        day_of_year = current_date.timetuple().tm_yday
        
        examples = {
            "morning": {"hour": "06", "minute": "30", "humidity": "75", "pressure": "1015"},
            "noon": {"hour": "12", "minute": "00", "humidity": "65", "pressure": "1013"},
            "evening": {"hour": "18", "minute": "00", "humidity": "70", "pressure": "1012"},
            "night": {"hour": "23", "minute": "30", "humidity": "80", "pressure": "1014"}
        }
        
        if time_type in examples:
            example = examples[time_type]
            self.hour_var.set(example["hour"])
            self.minute_var.set(example["minute"])
            self.humidity_var.set(example["humidity"])
            self.pressure_var.set(example["pressure"])
            
            # Auto-predict
            self.predict_temperature()
            
    def load_model_artifacts(self):
        """Load the trained model and preprocessing artifacts"""
        try:
            model_dir = "../models"
            
            # Check if model files exist
            required_files = [
                "temperature_model.pkl",
                "feature_scaler.pkl", 
                "target_scaler.pkl",
                "model_info.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(model_dir, file)):
                    missing_files.append(file)
                    
            if missing_files:
                error_msg = f"Missing model files: {', '.join(missing_files)}\n\nPlease run the training script first!"
                messagebox.showerror("Model Not Found", error_msg)
                self.status_var.set("ERROR - Model files not found")
                return False
            
            # Load model artifacts
            self.model = joblib.load(os.path.join(model_dir, "temperature_model.pkl"))
            self.feature_scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
            self.target_scaler = joblib.load(os.path.join(model_dir, "target_scaler.pkl"))
            
            with open(os.path.join(model_dir, "model_info.json"), 'r') as f:
                self.model_info = json.load(f)
                
            self.status_var.set("Model loaded successfully - Ready for predictions")
            return True
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            messagebox.showerror("Model Load Error", error_msg)
            self.status_var.set("ERROR - Failed to load model")
            return False
            
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            # Day of year
            day = int(self.day_var.get())
            if not 1 <= day <= 365:
                raise ValueError("Day must be between 1-365")
                
            # Time
            hour = int(self.hour_var.get())
            minute = int(self.minute_var.get())
            if not 0 <= hour <= 23:
                raise ValueError("Hour must be between 0-23")
            if not 0 <= minute <= 59:
                raise ValueError("Minute must be between 0-59")
                
            # Humidity
            humidity = float(self.humidity_var.get())
            if not 0 <= humidity <= 100:
                raise ValueError("Humidity must be between 0-100%")
                
            # Pressure
            pressure = float(self.pressure_var.get())
            if not 900 <= pressure <= 1100:
                raise ValueError("Pressure must be reasonable (900-1100 hPa)")
                
            # Season
            season_text = self.season_var.get()
            if not season_text:
                raise ValueError("Please select a season")
            season = int(season_text.split(" - ")[0])
            
            return {
                'day_of_year': day,
                'time_chunk': hour * 4 + (minute // 15),  # Convert to 15-min chunk
                'humidity': humidity,
                'pressure': pressure,
                'season': season
            }
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating inputs: {str(e)}")
            return None
            
    def predict_temperature(self):
        """Make temperature prediction"""
        if not self.model:
            messagebox.showerror("Model Error", "Model not loaded. Please check model files.")
            return
            
        # Validate inputs
        inputs = self.validate_inputs()
        if not inputs:
            return
            
        try:
            # Prepare feature vector
            feature_vector = np.array([[
                inputs['day_of_year'],
                inputs['time_chunk'],
                inputs['humidity'],
                inputs['pressure'],
                inputs['season']
            ]])
            
            # Scale features
            feature_vector_scaled = self.feature_scaler.transform(feature_vector)
            
            # Make prediction (scaled)
            prediction_scaled = self.model.predict(feature_vector_scaled)
            
            # Convert back to original scale
            prediction = self.target_scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).flatten()[0]
            
            # Display prediction
            self.prediction_var.set(f"{prediction:.2f} °C")
            
            # Calculate confidence (inverse of typical error for this time)
            time_chunk = inputs['time_chunk']
            confidence_score = self.calculate_confidence(time_chunk, prediction)
            self.confidence_var.set(confidence_score)
            
            # Update status
            self.status_var.set(f"Prediction made: {prediction:.2f}°C")
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            messagebox.showerror("Prediction Error", error_msg)
            self.status_var.set("ERROR - Prediction failed")
            print(traceback.format_exc())  # For debugging
            
    def calculate_confidence(self, time_chunk, prediction):
        """Calculate prediction confidence based on time and value"""
        try:
            # Simple confidence based on typical time-based patterns
            # In practice, this could use model uncertainty or historical errors
            
            # Time-based confidence (stable periods = higher confidence)
            stable_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Early morning
            
            if time_chunk in stable_times:
                time_confidence = "High"
            elif 20 <= time_chunk <= 40:  # Mid-morning to afternoon
                time_confidence = "Medium"
            else:  # Evening and transition periods
                time_confidence = "Medium"
                
            # Temperature range confidence
            if 25 <= prediction <= 30:  # Typical range
                temp_confidence = "Good"
            else:  # Extreme values
                temp_confidence = "Fair"
                
            return f"{time_confidence} ({temp_confidence} range)"
            
        except Exception:
            return "Moderate"

def main():
    """Main GUI function"""
    root = tk.Tk()
    app = TemperaturePredictionGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
