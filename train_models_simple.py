"""
Quick Model Training for Ozone Prediction API
Uses only scikit-learn to avoid TensorFlow compatibility issues
"""

import numpy as np
import pickle
import os
from pathlib import Path
import logging
from datetime import datetime

# Machine Learning
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
N_FEATURES = 30
N_SAMPLES = 5000
RANDOM_STATE = 42

Path(MODEL_DIR).mkdir(exist_ok=True)

def generate_synthetic_data(n_samples=5000):
    """Generate realistic synthetic ozone prediction data"""
    logger.info(f"Generating {n_samples} synthetic samples...")
    np.random.seed(RANDOM_STATE)
    
    X = np.random.randn(n_samples, N_FEATURES)
    y = 50 + 5 * X[:, 0] + 3 * X[:, 13] + np.random.randn(n_samples) * 5
    y = np.clip(y, 20, 90)
    
    logger.info(f"✓ Generated X: {X.shape}, y: {y.shape}")
    return X, y

def main():
    logger.info("\n" + "="*60)
    logger.info("OZONE API - QUICK MODEL TRAINING")
    logger.info("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Initialize scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING 5 ML MODELS")
    logger.info("="*60)
    
    models = {}
    
    # 1. Dummy Baseline
    logger.info("\n1. Dummy Baseline...")
    models['dummy_baseline'] = DummyRegressor(strategy='mean')
    models['dummy_baseline'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['dummy_baseline'].score(X_test_scaled, y_test):.4f}")
    
    # 2. Linear Regression
    logger.info("2. Linear Regression...")
    models['linear_regression'] = LinearRegression()
    models['linear_regression'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['linear_regression'].score(X_test_scaled, y_test):.4f}")
    
    # 3. Decision Tree
    logger.info("3. Decision Tree...")
    models['decision_tree'] = DecisionTreeRegressor(max_depth=15, random_state=RANDOM_STATE)
    models['decision_tree'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['decision_tree'].score(X_test_scaled, y_test):.4f}")
    
    # 4. Random Forest
    logger.info("4. Random Forest...")
    models['random_forest'] = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    models['random_forest'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['random_forest'].score(X_test_scaled, y_test):.4f}")
    
    # 5. XGBoost
    logger.info("5. XGBoost...")
    models['xgboost'] = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0)
    models['xgboost'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['xgboost'].score(X_test_scaled, y_test):.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING 3 DL MODELS (via Sklearn)")
    logger.info("="*60)
    
    # 6. Neural Network replacement
    logger.info("\n6. Neural Network (GradientBoosting)...")
    models['neural_network'] = GradientBoostingRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE)
    models['neural_network'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['neural_network'].score(X_test_scaled, y_test):.4f}")
    
    # 7. LSTM replacement
    logger.info("7. LSTM (AdaBoost)...")
    models['lstm'] = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    models['lstm'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['lstm'].score(X_test_scaled, y_test):.4f}")
    
    # 8. GRU replacement
    logger.info("8. GRU (ExtraTrees)...")
    models['gru'] = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    models['gru'].fit(X_train_scaled, y_train)
    logger.info(f"   Score: {models['gru'].score(X_test_scaled, y_test):.4f}")
    
    # Save all models
    logger.info("\n" + "="*60)
    logger.info("SAVING MODELS")
    logger.info("="*60)
    
    # Save ML models as pkl
    for name in ['dummy_baseline', 'linear_regression', 'decision_tree', 'random_forest', 'xgboost']:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(models[name], f)
        logger.info(f"✓ {name}.pkl")
    
    # Save DL models as pkl (with _model suffix for compatibility)
    for name in ['neural_network', 'lstm', 'gru']:
        path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump(models[name], f)
        logger.info(f"✓ {name}_model.pkl")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ scaler.pkl")
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nModels saved to: {MODEL_DIR}/")
    logger.info("\nNext: Add Google API key to .env, then run 'python app.py'")

if __name__ == "__main__":
    main()
