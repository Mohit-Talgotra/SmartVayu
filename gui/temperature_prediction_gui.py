"""
Temperature Prediction GUI
========================
A Streamlit-based interface for the LSTM temperature prediction model.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from temperature_lstm_model.predict_temperature import TemperaturePredictor

# Page config
st.set_page_config(
    page_title="Temperature Predictor",
    page_icon="🌡️",
    layout="centered"
)

# Title and description
st.title("🌡️ Temperature Prediction")
st.markdown("""
Enter current sensor readings to predict the next temperature value.
The model will use historical patterns to make its prediction.
""")

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return TemperaturePredictor()

try:
    predictor = load_predictor()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Input section
st.subheader("Current Sensor Readings")

# Create columns for inputs
col1, col2 = st.columns(2)

with col1:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Relative humidity (0-100%)"
    )
    
    pressure = st.number_input(
        "Pressure (hPa)",
        min_value=900.0,
        max_value=1100.0,
        value=1013.0,
        help="Atmospheric pressure (typical range: 950-1050 hPa)"
    )

with col2:
    light = st.number_input(
        "Light Level (lux)",
        min_value=0.0,
        max_value=100000.0,
        value=1000.0,
        help="Light intensity (0 = dark, 100000 = very bright)"
    )
    
    current_temp = st.number_input(
        "Current Temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=25.0,
        help="Current temperature (-20 to 50°C)"
    )

# Prediction button
if st.button("🔮 Predict Next Temperature", type="primary"):
    # Create a DataFrame with 30 timesteps
    # For simplicity, we'll use the current values repeated
    timestamps = [
        datetime.now() - timedelta(minutes=i)
        for i in range(29, -1, -1)
    ]
    
    data = pd.DataFrame({
        'date_time': timestamps,
        'humidity': [humidity] * 30,
        'pressure': [pressure] * 30,
        'light': [light] * 30,
        'temperature': [current_temp] * 30
    })
    
    # Make prediction
    with st.spinner("🔄 Making prediction..."):
        try:
            prediction = predictor.predict_next_temperature(data)
            
            # Show prediction
            st.success("✅ Prediction complete!")
            
            # Calculate temperature change
            temp_change = prediction - current_temp
            change_text = "increase" if temp_change > 0 else "decrease"
            
            # Display results
            st.markdown("### Prediction Results")
            st.markdown(f"""
            🌡️ **Next temperature:** {prediction:.2f}°C
            
            Expected {change_text}: {abs(temp_change):.2f}°C
            """)
            
            # Add a simple visualization
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add current temperature point
            fig.add_trace(go.Scatter(
                x=['Current'],
                y=[current_temp],
                mode='markers+text',
                name='Current',
                text=[f'{current_temp:.1f}°C'],
                textposition='top center',
                marker=dict(size=15, color='blue')
            ))
            
            # Add predicted temperature point
            fig.add_trace(go.Scatter(
                x=['Predicted'],
                y=[prediction],
                mode='markers+text',
                name='Predicted',
                text=[f'{prediction:.1f}°C'],
                textposition='top center',
                marker=dict(size=15, color='green')
            ))
            
            # Add arrow connecting points
            fig.add_shape(
                type="line",
                x0=0,
                y0=current_temp,
                x1=1,
                y1=prediction,
                line=dict(color="gray", width=2, dash="dot"),
                row=1,
                col=1
            )
            
            # Update layout
            fig.update_layout(
                title="Temperature Change Visualization",
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Temperature (°C)",
                yaxis=dict(range=[
                    min(current_temp, prediction) - 1,
                    max(current_temp, prediction) + 1
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Add information section
with st.expander("ℹ️ About this predictor"):
    st.markdown("""
    This interface uses a trained LSTM (Long Short-Term Memory) neural network to predict 
    temperature values based on current sensor readings. The model was trained on a year 
    of historical data and achieved:
    
    - Training accuracy: 99.32%
    - Validation accuracy: 98.97%
    - Expected precision: ±0.35°C
    
    For best results, provide sensor readings within typical ranges:
    - Humidity: 20-80%
    - Pressure: 950-1050 hPa
    - Light: 0-100000 lux (0 = dark, ~10000 = daylight)
    - Temperature: -10 to 40°C (typical range)
    """)

# Footer
st.markdown("---")
st.markdown("*Made with Streamlit and TensorFlow*")