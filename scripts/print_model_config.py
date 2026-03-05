import joblib
import os

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_models', 'lstm_temperature', 'model_config.pkl')
print('Loading model config from:', config_path)
config = joblib.load(config_path)

print('\nConfig keys:')
for k in config.keys():
    print(' -', k)

feature_columns = config.get('feature_columns')
sequence_length = config.get('sequence_length')
prediction_horizon = config.get('prediction_horizon', 1)

print(f"\nSequence length: {sequence_length}")
print(f"Prediction horizon: {prediction_horizon}")

if feature_columns is None:
    print('\nNo feature_columns found in config.')
else:
    print(f"\nNumber of feature columns: {len(feature_columns)}\n")
    for i, col in enumerate(feature_columns, 1):
        print(f"{i:03d}: {col}")
