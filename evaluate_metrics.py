#!/usr/bin/env python3
"""
Model Metrics Evaluation
Shows R² score and other metrics for all trained models
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate synthetic test data (same as training)
np.random.seed(42)
N_FEATURES = 30
N_SAMPLES = 1000

X = np.zeros((N_SAMPLES, N_FEATURES))
X[:, 0] = np.random.uniform(800, 1050, N_SAMPLES)  # Pressure
X[:, 1] = np.random.randint(0, 7, N_SAMPLES)        # Day of week
X[:, 2] = np.random.randint(0, 2, N_SAMPLES)        # Is weekend
X[:, 3] = np.random.randint(1, 54, N_SAMPLES)       # Week of year
X[:, 4] = np.random.randint(1, 5, N_SAMPLES)        # Quarter
X[:, 5] = np.random.randint(1, 367, N_SAMPLES)      # Day of year
X[:, 6] = np.random.randint(0, 2, N_SAMPLES)        # Is holiday
for i in range(7, 13):
    X[:, i] = np.random.uniform(-1, 1, N_SAMPLES)
for i in range(13, 17):
    X[:, i] = np.random.uniform(20, 70, N_SAMPLES)
X[:, 17] = np.random.uniform(20, 70, N_SAMPLES)
X[:, 18] = np.random.uniform(0, 10, N_SAMPLES)
X[:, 19] = np.random.uniform(20, 70, N_SAMPLES)
X[:, 20] = X[:, 0] * X[:, 4]
X[:, 21] = X[:, 8] * X[:, 0]
X[:, 22] = X[:, 0] ** 2
X[:, 23] = X[:, 0] ** 3
X[:, 24] = np.random.uniform(30, 60, N_SAMPLES)
X[:, 25] = np.random.uniform(5, 15, N_SAMPLES)
X[:, 26] = np.random.uniform(30, 70, N_SAMPLES)
X[:, 27] = np.random.uniform(30, 70, N_SAMPLES)
X[:, 28] = np.random.uniform(30, 70, N_SAMPLES)
X[:, 29] = np.random.uniform(-20, 20, N_SAMPLES)

y = (
    30 +
    0.02 * X[:, 0] +
    0.5 * X[:, 13] +
    0.3 * X[:, 14] +
    np.sin(2 * np.pi * X[:, 4] / 4) * 5 +
    0.1 * np.random.randn(N_SAMPLES) * 5
)
y = np.clip(y, 20, 90)

# Load scaler
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# Evaluate models
model_names = [
    'dummy_baseline',
    'linear_regression', 
    'decision_tree',
    'random_forest',
    'xgboost',
    'neural_network',
    'lstm',
    'gru'
]

print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS".center(80))
print("="*80)
print(f"{'Model':<20} {'Type':<15} {'R² Score':<15} {'MAE':<15} {'RMSE':<15}")
print("-"*80)

metrics_data = []

for model_name in model_names:
    try:
        # Try ML models first
        pkl_file = f'./models/{model_name}.pkl'
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_scaled)
        else:
            # Try DL models with underscore naming
            pkl_file = f'./models/{model_name}_model.pkl'
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_scaled)
            else:
                print(f"{model_name:<20} {'❌':<15} File not found")
                continue
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Categorize as ML or DL
        model_type = "DL (sklearn)" if model_name in ['neural_network', 'lstm', 'gru'] else "ML"
        
        print(f"{model_name:<20} {model_type:<15} {r2:<15.4f} {mae:<15.4f} {rmse:<15.4f}")
        metrics_data.append({
            'name': model_name,
            'type': model_type,
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        })
        
    except Exception as e:
        print(f"{model_name:<20} {'ERROR':<15} {str(e)[:40]}")

print("-"*80)

# Summary statistics
if metrics_data:
    r2_scores = [m['r2'] for m in metrics_data]
    mae_scores = [m['mae'] for m in metrics_data]
    
    print(f"\n{'SUMMARY STATISTICS':^80}")
    print("-"*80)
    print(f"Best R² Score:   {max(r2_scores):.4f} ({metrics_data[r2_scores.index(max(r2_scores))]['name']})")
    print(f"Worst R² Score:  {min(r2_scores):.4f} ({metrics_data[r2_scores.index(min(r2_scores))]['name']})")
    print(f"Average R² Score: {np.mean(r2_scores):.4f}")
    print(f"\nBest MAE:        {min(mae_scores):.4f}")
    print(f"Average MAE:     {np.mean(mae_scores):.4f}")

print("="*80 + "\n")
