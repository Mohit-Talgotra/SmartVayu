import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
from datetime import datetime, timedelta
import shutil
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlp.command_parser import parse_command

class TemperatureAdjuster:
    def __init__(self, root):
        self.root = root
        self.root.title("Temperature Adjustment System")
        
        # Load original temperature data
        self.df = pd.read_csv('indoor_15min.csv')
        self.df['time'] = pd.to_datetime(self.df['time'])
        
        # Create backup of original CSV
        shutil.copy2('indoor_15min.csv', 'indoor_15min_original.csv')
        
        # Store original temperatures for reference
        self.original_temps = self.df['indoor_temperature'].copy()
        
        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Time selection
        time_frame = ttk.LabelFrame(self.root, text="Select Time", padding="10")
        time_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # Hours (00-23)
        ttk.Label(time_frame, text="Hour:").grid(row=0, column=0, padx=5)
        self.hour_var = tk.StringVar()
        self.hour_combo = ttk.Combobox(time_frame, textvariable=self.hour_var, width=5)
        self.hour_combo['values'] = [f"{i:02d}" for i in range(24)]
        self.hour_combo.grid(row=0, column=1, padx=5)
        
        # Minutes (00, 15, 30, 45)
        ttk.Label(time_frame, text="Minute:").grid(row=0, column=2, padx=5)
        self.minute_var = tk.StringVar()
        self.minute_combo = ttk.Combobox(time_frame, textvariable=self.minute_var, width=5)
        self.minute_combo['values'] = ['00', '15', '30', '45']
        self.minute_combo.grid(row=0, column=3, padx=5)
        
        # Command input
        command_frame = ttk.LabelFrame(self.root, text="Enter Command", padding="10")
        command_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.command_entry = ttk.Entry(command_frame, width=50)
        self.command_entry.grid(row=0, column=0, padx=5, pady=5)
        
        # Process button
        self.process_btn = ttk.Button(command_frame, text="Process Command", command=self.process_command)
        self.process_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Status display
        self.status_var = tk.StringVar()
        status_label = ttk.Label(self.root, textvariable=self.status_var, wraplength=400)
        status_label.grid(row=2, column=0, padx=10, pady=5)
        
        # Plot button
        self.plot_btn = ttk.Button(self.root, text="Show Temperature Plot", command=self.plot_temperatures)
        self.plot_btn.grid(row=3, column=0, pady=10)
        
        # Reset button
        self.reset_btn = ttk.Button(self.root, text="Reset to Original", command=self.reset_temperatures)
        self.reset_btn.grid(row=4, column=0, pady=5)

    def process_command(self):
        # Validate time input
        try:
            hour = int(self.hour_var.get())
            minute = int(self.minute_var.get())
            if not (0 <= hour <= 23 and minute in [0, 15, 30, 45]):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please select valid time (hour: 0-23, minute: 00/15/30/45)")
            return
        
        # Get command text
        command = self.command_entry.get().strip()
        if not command:
            messagebox.showerror("Error", "Please enter a command")
            return
        
        # Process through NLP
        result = parse_command(command)
        
        # Check if temperature adjustment is needed
        temp_action = result.get('temperature')
        if not temp_action:
            self.status_var.set("No temperature adjustment needed")
            return
        
        # Calculate temperature change
        if temp_action['type'] == 'relative':
            delta = temp_action['delta']  # This will be negative for "hot" and positive for "cold"
            change = delta  # Each unit is roughly 1°C
        else:
            # For absolute temperature, calculate difference from current
            target_time = pd.Timestamp(datetime.now().date()) + timedelta(hours=hour, minutes=minute)
            current_temp = self.get_temperature_at_time(target_time)
            change = temp_action['value_c'] - current_temp
        
        # Apply the temperature change
        self.adjust_temperature(hour, minute, change)
        
        # Update status
        self.status_var.set(f"Processed: {command}\nTemperature change: {change:+.1f}°C")
        
        # Update the plot
        self.plot_temperatures()

    def get_temperature_at_time(self, target_time):
        # Find the temperature at the given time
        idx = self.df['time'].searchsorted(target_time)
        if idx >= len(self.df):
            idx = -1
        return self.df.iloc[idx]['indoor_temperature']

    def adjust_temperature(self, hour, minute, change):
        # Find the index for the specified time
        target_time = pd.Timestamp(datetime.now().date()) + timedelta(hours=hour, minutes=minute)
        start_idx = self.df['time'].searchsorted(target_time)
        
        # Calculate end index (3 hours later)
        end_time = target_time + timedelta(hours=3)
        end_idx = self.df['time'].searchsorted(end_time)
        
        if end_idx >= len(self.df):
            end_idx = len(self.df) - 1
        
        # Calculate gradual return over 3 hours
        num_points = end_idx - start_idx + 1  # Include the end point
        if num_points > 1:
            # Create linear gradient from change back to original
            gradients = np.linspace(change, 0, num_points)
            temp_slice = self.original_temps.iloc[start_idx:end_idx+1].values
            self.df.loc[start_idx:end_idx, 'indoor_temperature'] = temp_slice + gradients
        
        # Save modified data
        self.df.to_csv('indoor_15min.csv', index=False)

    def plot_temperatures(self):
        plt.figure(figsize=(12, 6))
        
        # Plot original temperatures
        plt.plot(self.df['time'], self.original_temps, 
                label='Original Prediction', 
                color='blue', 
                linestyle='--', 
                alpha=0.5)
        
        # Plot modified temperatures
        plt.plot(self.df['time'], self.df['indoor_temperature'], 
                label='Modified Temperature', 
                color='red', 
                linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title('Indoor Temperature: Original vs Modified')
        plt.legend()
        
        # Format x-axis to show hours
        plt.gca().xaxis.set_major_locator(HourLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temperature_adjustment.png', dpi=300, bbox_inches='tight')
        plt.close()

    def reset_temperatures(self):
        # Restore original temperatures
        self.df['indoor_temperature'] = self.original_temps.copy()
        self.df.to_csv('indoor_15min.csv', index=False)
        
        # Update status and plot
        self.status_var.set("Reset to original temperatures")
        self.plot_temperatures()

if __name__ == "__main__":
    root = tk.Tk()
    app = TemperatureAdjuster(root)
    root.mainloop()