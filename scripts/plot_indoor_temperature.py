import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter

# Load the indoor temperature predictions
df = pd.read_csv('indoor_15min.csv')
df['time'] = pd.to_datetime(df['time'])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot indoor temperature
plt.plot(df['time'], df['indoor_temperature'], 
         color='darkblue', 
         linewidth=2, 
         label='Predicted Indoor Temperature')

# Add labels and title
plt.xlabel('Time of Day')
plt.ylabel('Temperature (°C)')
plt.title('Predicted Indoor Temperature Throughout the Day')

# Format x-axis to show hours nicely
plt.gcf().autofmt_xdate()  # Angle and align the tick labels
plt.gca().xaxis.set_major_locator(HourLocator(interval=2))  # Show every 2 hours
plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add legend
plt.legend()

# Show min and max temperatures
min_temp = df['indoor_temperature'].min()
max_temp = df['indoor_temperature'].max()
min_time = df.loc[df['indoor_temperature'].idxmin(), 'time']
max_time = df.loc[df['indoor_temperature'].idxmax(), 'time']

# Add annotations for min and max temperatures
plt.annotate(f'Min: {min_temp:.1f}°C', 
            xy=(min_time, min_temp),
            xytext=(10, 10),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.annotate(f'Max: {max_temp:.1f}°C',
            xy=(max_time, max_temp),
            xytext=(10, -10),
            textcoords='offset points',
            ha='left',
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Add statistical summary as text
stats_text = (f'Statistics:\n'
             f'Mean: {df["indoor_temperature"].mean():.1f}°C\n'
             f'Std Dev: {df["indoor_temperature"].std():.1f}°C')
plt.text(0.02, 0.98, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout and save
plt.tight_layout()
plt.savefig('indoor_temperature_prediction.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as: indoor_temperature_prediction.png")

# Print summary statistics
print("\nIndoor Temperature Summary:")
print(f"Minimum: {min_temp:.1f}°C at {min_time.strftime('%H:%M')}")
print(f"Maximum: {max_temp:.1f}°C at {max_time.strftime('%H:%M')}")
print(f"Average: {df['indoor_temperature'].mean():.1f}°C")
print(f"Standard Deviation: {df['indoor_temperature'].std():.1f}°C")