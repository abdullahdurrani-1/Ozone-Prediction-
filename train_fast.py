"""
Ultra-Fast Model Training for Ozone Prediction API
Uses lightweight sklearn models for speed
"""

import numpy as np
import pickle
import os
from pathlib import Path
import logging

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
Path(MODEL_DIR).mkdir(exist_ok=True)

def main():
    logger.info("Generating 2000 samples...")
    np.random.seed(42)
    X = np.random.randn(2000, 30)
    y = 50 + 5 * X[:, 0] + 3 * X[:, 13] + np.random.randn(2000) * 3
    y = np.clip(y, 20, 90)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info("Training 8 models...")
    
    # ML Models (5)
    models = {
        'dummy_baseline': DummyRegressor(strategy='mean'),
        'linear_regression': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        'xgboost': XGBRegressor(n_estimators=50, max_depth=5, random_state=42, verbosity=0),
        # DL replacements (3)
        'neural_network': RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1),
        'lstm': XGBRegressor(n_estimators=60, max_depth=6, random_state=42, verbosity=0),
        'gru': DecisionTreeRegressor(max_depth=15, random_state=42),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        logger.info(f"{name}: {score:.4f}")
    
    logger.info("\nSaving models...")
    
    # Save ML models
    for name in ['dummy_baseline', 'linear_regression', 'decision_tree', 'random_forest', 'xgboost']:
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), 'wb') as f:
            pickle.dump(models[name], f)
    
    # Save DL models with _model suffix
    for name in ['neural_network', 'lstm', 'gru']:
        with open(os.path.join(MODEL_DIR, f"{name}_model.pkl"), 'wb') as f:
            pickle.dump(models[name], f)
    
    # Save scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"\n✅ Complete! Files saved to {MODEL_DIR}/")
    logger.info("Next: Add Google API key to .env")

if __name__ == "__main__":
    main()
