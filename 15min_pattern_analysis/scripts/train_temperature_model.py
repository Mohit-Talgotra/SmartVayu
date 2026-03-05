#!/usr/bin/env python3
"""
Temperature Prediction Model Training Script
==========================================
Trains a deep neural network to predict temperature based on:
- Day of year (1-365)
- Time of day (15-minute chunks 0-95)
- Humidity 
- Pressure
- Season (1-4)

Uses the Plus dataset and saves trained model + preprocessing artifacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class TemperaturePredictionModel:
    """Deep learning model for temperature prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.day_encoder = None
        self.feature_columns = ['day_of_year', 'time_chunk', 'humidity', 'pressure', 'season']
        
        # Create models directory
        os.makedirs('../models', exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare the Plus dataset for training"""
        print("Loading Plus dataset for model training...")
        
        # Load data
        df = pd.read_csv('../../data/combined_plus_sensor_data.csv')
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"Loaded {len(df):,} records")
        print(f"Date range: {df['date_time'].min()} to {df['date_time'].max()}")
        
        # Create features
        print("\nCreating features...")
        
        # Time features
        df['day_of_year'] = df['date_time'].dt.dayofyear
        df['hour'] = df['date_time'].dt.hour
        df['minute'] = df['date_time'].dt.minute
        df['time_chunk'] = df['hour'] * 4 + (df['minute'] // 15)  # 0-95 for 15-min chunks
        df['season'] = df['date_time'].dt.quarter
        
        # Remove rows with missing values in key columns
        required_cols = ['temperature'] + self.feature_columns
        df_clean = df[required_cols].dropna()
        
        print(f"After cleaning: {len(df_clean):,} records")
        print(f"Removed {len(df) - len(df_clean):,} records with missing values")
        
        return df_clean
        
    def prepare_features_and_target(self, df):
        """Prepare feature matrix and target variable"""
        print("\nPreparing features and target...")
        
        # Features
        X = df[self.feature_columns].copy()
        
        # Target
        y = df['temperature'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Feature statistics
        print("\nFeature statistics:")
        print(X.describe())
        
        return X, y
        
    def preprocess_data(self, X_train, X_test, y_train, y_test):
        """Preprocess features and target for training"""
        print("\nPreprocessing data...")
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Scale target (helps with neural network training)
        self.target_scaler = StandardScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        print("Data preprocessing complete")
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
        
    def build_and_train_model(self, X_train, y_train, X_test, y_test):
        """Build and train the neural network model"""
        print("\nBuilding and training neural network...")
        
        # Multi-layer perceptron with multiple hidden layers
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),  # Deep architecture
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=True
        )
        
        print("Model architecture:")
        print(f"- Input layer: {X_train.shape[1]} features")
        print(f"- Hidden layers: {self.model.hidden_layer_sizes}")
        print(f"- Output layer: 1 (temperature)")
        print(f"- Total parameters: ~{self._estimate_parameters(X_train.shape[1])}")
        
        # Train the model
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        
        print(f"Training completed in {self.model.n_iter_} iterations")
        return self.model
        
    def _estimate_parameters(self, input_size):
        """Estimate total number of parameters in the network"""
        layers = [input_size] + list(self.model.hidden_layer_sizes) + [1]
        params = 0
        for i in range(len(layers)-1):
            params += layers[i] * layers[i+1] + layers[i+1]  # weights + biases
        return params
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\nEvaluating model performance...")
        
        # Make predictions (scaled)
        y_pred_scaled = self.model.predict(X_test)
        
        # Convert back to original scale
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        print(f"Model Performance Metrics:")
        print(f"- Mean Absolute Error (MAE): {mae:.4f}°C")
        print(f"- Root Mean Square Error (RMSE): {rmse:.4f}°C")
        print(f"- R² Score: {r2:.4f}")
        
        # Additional insights
        print(f"\nPrediction Quality:")
        print(f"- Average prediction error: ±{mae:.2f}°C")
        print(f"- 68% of predictions within: ±{rmse:.2f}°C")
        print(f"- Variance explained: {r2*100:.1f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test_orig,
            'y_pred': y_pred_orig
        }
        
    def create_evaluation_plots(self, results):
        """Create evaluation plots"""
        print("\nCreating evaluation plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # 1. Actual vs Predicted scatter plot
        ax1.scatter(y_test, y_pred, alpha=0.5, s=1)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Temperature (°C)')
        ax1.set_ylabel('Predicted Temperature (°C)')
        ax1.set_title(f'Actual vs Predicted (R² = {results["r2"]:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=1)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Temperature (°C)')
        ax2.set_ylabel('Residuals (°C)')
        ax2.set_title(f'Residuals (MAE = {results["mae"]:.3f}°C)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax3.hist(residuals, bins=50, alpha=0.7, density=True)
        ax3.set_xlabel('Prediction Error (°C)')
        ax3.set_ylabel('Density')
        ax3.set_title('Error Distribution')
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning curve (if available)
        if hasattr(self.model, 'loss_curve_'):
            ax4.plot(self.model.loss_curve_)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss Curve')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning curve\nnot available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Training Progress')
        
        plt.tight_layout()
        plt.savefig('../figures/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: model_evaluation.png")
        
    def save_model_artifacts(self):
        """Save trained model and preprocessing artifacts"""
        print("\nSaving model artifacts...")
        
        # Save model
        joblib.dump(self.model, '../models/temperature_model.pkl')
        print("  Saved: temperature_model.pkl")
        
        # Save scalers
        joblib.dump(self.feature_scaler, '../models/feature_scaler.pkl')
        joblib.dump(self.target_scaler, '../models/target_scaler.pkl')
        print("  Saved: feature_scaler.pkl, target_scaler.pkl")
        
        # Save feature information
        feature_info = {
            'feature_columns': self.feature_columns,
            'model_type': 'MLPRegressor',
            'input_features': {
                'day_of_year': 'Day of year (1-365)',
                'time_chunk': '15-minute time chunk (0-95)',
                'humidity': 'Relative humidity (%)',
                'pressure': 'Atmospheric pressure',
                'season': 'Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)'
            }
        }
        
        import json
        with open('../models/model_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        print("  Saved: model_info.json")
        
    def create_feature_importance_analysis(self, X_train, y_train):
        """Analyze feature importance using permutation"""
        print("\nAnalyzing feature importance...")
        
        # Simple feature importance by training separate models
        base_score = self.model.score(X_train, y_train)
        importances = []
        
        for i, feature in enumerate(self.feature_columns):
            # Create a copy and shuffle one feature
            X_permuted = X_train.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self.model.score(X_permuted, y_train)
            importance = base_score - permuted_score
            importances.append(importance)
        
        # Create importance plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), [importances[i] for i in indices])
        plt.xlabel('Features')
        plt.ylabel('Importance (Score Decrease)')
        plt.title('Feature Importance Analysis')
        plt.xticks(range(len(importances)), [self.feature_columns[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('../figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: feature_importance.png")
        
        # Print importance ranking
        print("\nFeature Importance Ranking:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.feature_columns[idx]}: {importances[idx]:.4f}")
            
    def run_full_training_pipeline(self):
        """Run the complete training pipeline"""
        print("="*80)
        print("TEMPERATURE PREDICTION MODEL TRAINING")
        print("="*80)
        
        # 1. Load and prepare data
        df = self.load_and_prepare_data()
        
        # 2. Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # 3. Train-test split
        print(f"\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df['season']
        )
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # 4. Preprocess data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.preprocess_data(
            X_train, X_test, y_train, y_test
        )
        
        # 5. Build and train model
        self.build_and_train_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        
        # 6. Evaluate model
        results = self.evaluate_model(X_test_scaled, y_test_scaled)
        
        # 7. Create evaluation plots
        self.create_evaluation_plots(results)
        
        # 8. Analyze feature importance
        self.create_feature_importance_analysis(X_train_scaled, y_train_scaled)
        
        # 9. Save model artifacts
        self.save_model_artifacts()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"[MODEL] Saved trained neural network")
        print(f"[SCALERS] Saved preprocessing artifacts")
        print(f"[PLOTS] Generated evaluation visualizations")
        print(f"[PERFORMANCE] MAE: {results['mae']:.3f}°C, R²: {results['r2']:.3f}")
        print("="*80)
        
        return results

def main():
    """Main training function"""
    trainer = TemperaturePredictionModel()
    results = trainer.run_full_training_pipeline()
    return trainer, results

if __name__ == "__main__":
    main()
