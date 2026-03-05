from src.models.temperature_forecast import generate_next_24_hours

# Your API key is already included in the module
location = "London"  # or any other location
forecasts = generate_next_24_hours(location, "a1715ffa6e4c43a2ab8165946250411")

# Print forecasts
for forecast in forecasts:
    print(f"Time: {forecast['hour']}")
    print(f"Actual: {forecast['actual_temp']}°C")
    print(f"Predicted: {forecast['predicted_temp']}°C")