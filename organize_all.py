"""
Complete project reorganization script
====================================
Moves ALL files into appropriate folders, leaving only essential files in root.
"""

import os
import shutil
from pathlib import Path

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Define all directories
DIRS = {
    'src': {
        'models': None,
        'preprocessing': None,
        'utils': None,
        'nlp': None,
    },
    'trained_models': {
        'lstm_temperature': None,
        'pattern_analysis': None,
    },
    'data': {
        'raw': {
            'plus_separate_files': None
        },
        'processed': None,
    },
    'gui': None,
    'scripts': {
        'validation': None,
    },
    'tests': None,
    'reports': {
        'figures': None,
        'model_performance': None,
    },
    'visualizations': {
        'analysis': None,
        'validation': None,
        'training': None,
    },
}

# Create directory structure
def create_directories(base_path, structure):
    for key, value in structure.items():
        path = base_path / key
        ensure_dir(path)
        if value:  # If there are subdirectories
            create_directories(path, value)

# File categorization rules
FILE_RULES = {
    # Images
    'visualizations/analysis': [
        '*analysis*.png',
        '*pattern*.png',
        'smote*.png',
    ],
    'visualizations/validation': [
        '*validation*.png',
        '*performance*.png',
        '*drift*.png',
    ],
    'visualizations/training': [
        '*model_results*.png',
        '*training*.png',
    ],
    
    # Reports and documentation
    'reports/model_performance': [
        'MODEL_PERFORMANCE_REPORT.txt',
        '*REPORT*.txt',
        '*EXPLAINED.md',
    ],
    'reports/figures': [
        'reports/*.png',
    ],
    
    # Source code
    'src/models': [
        '*model*.py',
        '*predict*.py',
        '!*gui*.py',  # Exclude GUI files
    ],
    'gui': [
        '*gui*.py',
        '*tkinter*.py',
    ],
    'scripts/validation': [
        '*validation*.py',
        '*check*.py',
    ],
    
    # Data
    'data/raw/plus_separate_files': [
        'rpi_*.csv',
    ],
    'data/processed': [
        'combined_*.csv',
    ],
    
    # Model artifacts
    'trained_models/lstm_temperature': [
        'lstm_model.h5',
        'scaler_*.pkl',
        'model_config.pkl',
    ],
}

# Files to keep in root
ROOT_FILES = {
    'requirements.txt',
    'README.md',
    'HANDOVER.md',
    '.gitignore',
    'reorganize.py',
    'organize_all.py'
}

def should_ignore(file):
    """Check if file should be ignored"""
    ignore_patterns = {
        '__pycache__',
        '*.pyc',
        '.git',
        '.vscode',
    }
    return any(pattern in str(file) for pattern in ignore_patterns)

def matches_pattern(file, pattern):
    """Check if file matches the given pattern"""
    from fnmatch import fnmatch
    if pattern.startswith('!'):  # Exclude pattern
        return not fnmatch(file.name, pattern[1:])
    return fnmatch(file.name, pattern)

def categorize_file(file):
    """Determine which directory a file should go in"""
    if file.name in ROOT_FILES:
        return None
        
    if should_ignore(file):
        return None
        
    for target_dir, patterns in FILE_RULES.items():
        for pattern in patterns:
            if matches_pattern(file, pattern):
                return target_dir
    
    return 'misc'  # Default directory for uncategorized files

def move_file(src, dst_dir):
    """Move a file to its new location"""
    if not dst_dir:  # Keep in root
        return
        
    dst = BASE_DIR / dst_dir / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if dst.exists():
            print(f"File already exists: {dst}")
            return
            
        shutil.move(str(src), str(dst))
        print(f"Moved: {src.name} -> {dst_dir}")
    except Exception as e:
        print(f"Error moving {src.name}: {str(e)}")

def main():
    print("Creating directory structure...")
    create_directories(BASE_DIR, DIRS)
    
    print("\nMoving files...")
    for file in BASE_DIR.glob('**/*'):
        if file.is_file():
            relative_path = file.relative_to(BASE_DIR)
            if len(relative_path.parts) == 1:  # Only process files in root
                target_dir = categorize_file(file)
                if target_dir:
                    move_file(file, target_dir)

    print("\nReorganization completed!")

if __name__ == "__main__":
    main()