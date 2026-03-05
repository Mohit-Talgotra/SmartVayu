"""
Reorganize project files into the new structure.
"""
import os
import shutil

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define the paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create directories if they don't exist
dirs = [
    'src/models',
    'src/preprocessing',
    'src/utils',
    'trained_models/lstm_temperature',
    'trained_models/pattern_analysis',
    'data/raw/plus_separate_files',
    'data/processed',
    'gui',
    'scripts/validation',
    'tests'
]

for d in dirs:
    ensure_dir(os.path.join(BASE_DIR, d))

# Define file movements
moves = [
    # GUI files
    ('temperature_prediction_gui.py', 'gui/temperature_prediction_gui.py'),
    ('temperature_prediction_tkinter.py', 'gui/temperature_prediction_tkinter.py'),
    ('nlp_command_parser_tkinter.py', 'gui/nlp_command_parser_gui.py'),
    
    # Model files
    ('temperature_prediction_model.py', 'src/models/lstm_model.py'),
    ('temperature_prediction_app.py', 'src/models/temperature_app.py'),
    
    # Validation scripts
    ('model_validation_comprehensive.py', 'scripts/validation/comprehensive_validation.py'),
    ('quick_validation_check.py', 'scripts/validation/quick_validation.py'),
    
    # NLP files - preserve structure
    ('nlp', 'src/nlp'),
    
    # Data files
    ('plus separate files reduntant', 'data/raw/plus_separate_files'),
    ('combined_plus_sensor_data.csv', 'data/processed/combined_plus_sensor_data.csv')
]

# Move files
for src, dst in moves:
    src_path = os.path.join(BASE_DIR, src)
    dst_path = os.path.join(BASE_DIR, dst)
    
    if not os.path.exists(src_path):
        print(f"Source not found: {src_path}")
        continue
        
    try:
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)
        print(f"Moved: {src} -> {dst}")
    except Exception as e:
        print(f"Error moving {src}: {str(e)}")

print("\nReorganization completed!")