# SmartVayu Project Handover Documentation

## Project Overview
SmartVayu is a temperature prediction system that uses machine learning to forecast temperature values based on sensor data. The project includes two main models:
1. An LSTM-based temperature predictor (main production model)
2. A 15-minute pattern analysis model for temporal pattern discovery

## Directory Structure and Components

### 📁 src/
Core source code for the project.

#### models/
- `lstm_model.py`: Main LSTM temperature prediction model
  - Implements `TemperatureLSTMModel` class
  - Features: 90 engineered features including time, lag, and sensor data
  - Accuracy: 99.32% training, 98.97% validation
  - Input: 30 timesteps of sensor data
  - Output: Next temperature prediction (t+1)

- `mlp_model.py`: 15-minute pattern analysis model
  - Uses MLPRegressor for pattern discovery
  - Features: Day of year, time chunks, humidity, pressure, season
  - Purpose: Analyze 15-minute temperature patterns

#### preprocessing/
- `feature_engineering.py`: Feature engineering utilities
  - Time-based features
  - Lag features
  - Rolling statistics
  - Rate of change calculations

- `data_cleaning.py`: Data preprocessing utilities
  - Data validation
  - Missing value handling
  - Timestamp processing

#### utils/
- `time_features.py`: Time-related utility functions
  - Cyclical time encoding
  - Time window calculations
  - Sequence creation

#### nlp/
Natural Language Processing module for command interpretation
- `command_parser.py`: Parses temperature and fan control commands
- `lexicons.py`: Word lists and vocabularies
- `tests_samples.jsonl`: Example commands and expected outputs

### 📁 trained_models/
Trained model artifacts and configurations.

#### lstm_temperature/
Main LSTM model files:
- `lstm_model.h5`: Trained neural network weights
- `scaler_features.pkl`: Feature preprocessing scaler
- `scaler_target.pkl`: Target variable scaler
- `model_config.pkl`: Model configuration and parameters

#### pattern_analysis/
15-minute pattern model files:
- `temperature_model.pkl`: Trained MLPRegressor
- `scalers/`: Preprocessing scalers

### 📁 data/
Data storage and management.

#### raw/
- `plus_separate_files/`: Original sensor data files
  - Format: CSV files named `rpi_XX_plus.csv`
  - Contains: Raw sensor readings from different RPi devices

#### processed/
- `combined_plus_sensor_data.csv`: Main dataset
  - Combined and cleaned sensor data
  - Used for model training
  - ~1.5M records from June 2021 to July 2022

### 📁 gui/
Graphical user interfaces for the project.

- `temperature_prediction_gui.py`: Main prediction interface
  - Input: Current sensor readings
  - Output: Temperature prediction with visualization
  - Features: Real-time prediction, input validation

- `nlp_command_parser_gui.py`: NLP command interface
  - Purpose: Parse natural language temperature commands
  - Features: Command interpretation, JSON output display

### 📁 scripts/
Utility and training scripts.

- `train_lstm.py`: LSTM model training script
- `train_pattern_model.py`: 15-min pattern model training
- `validation/`:
  - `quick_validation.py`: Fast model validation
  - `comprehensive_validation.py`: Detailed validation suite

### 📁 reports/
Documentation and analysis results.

- `MODEL_PERFORMANCE_REPORT.txt`: Detailed model performance
  - Training results
  - Validation metrics
  - Architecture details
  - Usage recommendations

- `figures/`: Analysis visualizations
  - Temperature distributions
  - Correlation analyses
  - Time series plots
  - Model diagnostics

## Quick Start Guide

1. Setting up the environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Running the temperature prediction GUI:
   ```bash
   python gui/temperature_prediction_gui.py
   ```

3. Using the NLP command parser:
   ```bash
   python gui/nlp_command_parser_gui.py
   ```

## Model Usage

### LSTM Temperature Predictor
```python
from src.models.lstm_model import TemperatureLSTMModel
predictor = TemperatureLSTMModel()
# Provide 30 timesteps of sensor data
prediction = predictor.predict_next_temperature(sensor_data)
```

### NLP Command Parser
```python
from src.nlp.command_parser import parse_command
result = parse_command("set temperature to 25 degrees")
```

## Key Files and Their Purposes

1. **Data Processing**
   - `combine_plus_csv.py`: Combines separate sensor files
   - `check_missing_temperature.py`: Validates temperature data
   - `check_time_gaps.py`: Checks for temporal continuity

2. **Model Training**
   - `train_lstm.py`: Main LSTM model training
   - `train_pattern_model.py`: 15-min pattern model training

3. **Validation**
   - `quick_validation_check.py`: Fast validation suite
   - `model_validation_comprehensive.py`: Detailed validation

4. **User Interfaces**
   - `temperature_prediction_gui.py`: Main prediction interface
   - `nlp_command_parser_gui.py`: Command parsing interface

## Dependencies
Key requirements (from requirements.txt):
- streamlit>=1.24.0
- pandas>=1.5.3
- numpy>=1.23.5
- plotly>=5.13.1
- tensorflow>=2.12.0
- scikit-learn>=1.2.2
- imbalanced-learn>=0.11.0

## Model Performance Summary

### LSTM Model
- Training Accuracy: 99.32%
- Validation Accuracy: 98.97%
- MAE: ~0.286°C
- Features: 90 engineered inputs
- Sequence Length: 30 timesteps

### Pattern Analysis Model
- Purpose: 15-minute pattern discovery
- Architecture: MLPRegressor
- Features: Time, season, and sensor data
- Use Case: Pattern analysis and research

## Common Tasks

1. **Making Predictions**
   - Use the GUI for single predictions
   - Use `predict_temperature.py` for batch predictions

2. **Training New Models**
   - Run `train_lstm.py` for LSTM model
   - Run `train_pattern_model.py` for pattern model

3. **Validating Models**
   - Quick check: `quick_validation_check.py`
   - Deep analysis: `model_validation_comprehensive.py`

## Troubleshooting

1. **Model Loading Issues**
   - Check model artifacts in trained_models/
   - Verify keras/tensorflow versions
   - Use compile=False when loading

2. **Data Issues**
   - Check data/processed/ for latest dataset
   - Verify sensor data format
   - Check for missing values

3. **GUI Problems**
   - Verify requirements installation
   - Check model paths
   - Validate input ranges

## Future Improvements

1. **Model Enhancements**
   - Multi-step forecasting
   - Uncertainty quantification
   - Online learning capability

2. **Interface Updates**
   - Real-time monitoring
   - Batch prediction interface
   - API deployment

3. **Data Pipeline**
   - Automated data collection
   - Real-time preprocessing
   - Data quality monitoring

## Maintenance

1. **Regular Tasks**
   - Monthly model retraining
   - Data quality checks
   - Performance monitoring

2. **Updates**
   - Dependencies updates
   - Model recalibration
   - Feature engineering refinement

## Contact and Resources

For questions or issues:
- Check the model performance report
- Review validation results
- Consult the requirements.txt for versions