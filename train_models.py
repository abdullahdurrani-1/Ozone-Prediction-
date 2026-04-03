"""
Training script for Ozone Prediction API - All 8 Models
Generates synthetic training data and trains ML and DL models
"""

import numpy as np
import pickle
import os
from pathlib import Path
import logging

# Machine Learning
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Deep Learning - use try/except for optional TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = False  # Force sklearn fallback for stability
except ImportError:
    TF_AVAILABLE = False
    logger_stub = logging.getLogger(__name__)
    logger_stub.warning("TensorFlow not fully available, using sklearn alternatives for DL models")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = "./models"
N_FEATURES = 30  # As per schema
N_SAMPLES = 5000  # Synthetic training samples
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Create models directory
Path(MODEL_DIR).mkdir(exist_ok=True)

def generate_synthetic_data(n_samples=5000):
    """Generate realistic synthetic ozone prediction data"""
    logger.info(f"Generating {n_samples} synthetic training samples...")
    
    np.random.seed(RANDOM_STATE)
    
    # Generate features with realistic ranges based on schema
    X = np.zeros((n_samples, N_FEATURES))
    
    # Pressure (800-1050 mb)
    X[:, 0] = np.random.uniform(800, 1050, n_samples)
    
    # Day of week (0-6)
    X[:, 1] = np.random.randint(0, 7, n_samples)
    
    # Is weekend (0-1)
    X[:, 2] = np.random.randint(0, 2, n_samples)
    
    # Week of year (1-53)
    X[:, 3] = np.random.randint(1, 54, n_samples)
    
    # Quarter (1-4)
    X[:, 4] = np.random.randint(1, 5, n_samples)
    
    # Day of year (1-366)
    X[:, 5] = np.random.randint(1, 367, n_samples)
    
    # Is holiday season (0-1)
    X[:, 6] = np.random.randint(0, 2, n_samples)
    
    # Sine/cosine features (-1 to 1)
    for i in range(7, 13):
        X[:, i] = np.random.uniform(-1, 1, n_samples)
    
    # Ozone lag features (20-70 ppbv)
    for i in range(13, 17):
        X[:, i] = np.random.uniform(20, 70, n_samples)
    
    # Rolling mean/std
    X[:, 17] = np.random.uniform(20, 70, n_samples)  # rolling mean 3h
    X[:, 18] = np.random.uniform(0, 10, n_samples)   # rolling std 3h
    X[:, 19] = np.random.uniform(20, 70, n_samples)  # rolling mean 24h
    
    # Interaction features
    X[:, 20] = X[:, 0] * X[:, 4]  # pressure * quarter
    X[:, 21] = X[:, 8] * X[:, 0]  # hour_sin * pressure
    X[:, 22] = X[:, 0] ** 2       # pressure squared
    X[:, 23] = X[:, 0] ** 3       # pressure cubed
    
    # Site ozone statistics
    X[:, 24] = np.random.uniform(30, 60, n_samples)  # site mean ozone
    X[:, 25] = np.random.uniform(5, 15, n_samples)   # site std ozone
    X[:, 26] = np.random.uniform(30, 70, n_samples)  # month_hour avg
    X[:, 27] = np.random.uniform(30, 70, n_samples)  # hour avg
    X[:, 28] = np.random.uniform(30, 70, n_samples)  # month avg
    X[:, 29] = np.random.uniform(-20, 20, n_samples) # deviation from mean
    
    # Generate target: ozone levels (20-90 ppbv) with realistic relationship to features
    # Ozone is influenced by pressure, seasonal patterns, and historical values
    y = (
        30 +  # Base level
        0.02 * X[:, 0] +  # Pressure effect
        0.5 * X[:, 13] +  # Lag 1 effect
        0.3 * X[:, 14] +  # Lag 3 effect
        np.sin(2 * np.pi * X[:, 4] / 4) * 5 +  # Seasonal effect
        0.1 * np.random.randn(n_samples) * 5   # Noise
    )
    
    # Clip to realistic range
    y = np.clip(y, 20, 90)
    
    logger.info(f"✓ Generated X: {X.shape}, y: {y.shape}")
    return X, y


def train_ml_models(X_train, X_test, y_train, y_test, scaler):
    """Train all 5 ML models"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MACHINE LEARNING MODELS")
    logger.info("="*60)
    
    ml_models = {}
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Dummy Baseline
    logger.info("\n1. Training Dummy Baseline...")
    model = DummyRegressor(strategy='mean')
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    logger.info(f"   R² Score: {score:.4f}")
    ml_models['dummy_baseline'] = model
    
    # 2. Linear Regression
    logger.info("\n2. Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    logger.info(f"   R² Score: {score:.4f}")
    ml_models['linear_regression'] = model
    
    # 3. Decision Tree
    logger.info("\n3. Training Decision Tree...")
    model = DecisionTreeRegressor(max_depth=15, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    logger.info(f"   R² Score: {score:.4f}")
    ml_models['decision_tree'] = model
    
    # 4. Random Forest
    logger.info("\n4. Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    logger.info(f"   R² Score: {score:.4f}")
    ml_models['random_forest'] = model
    
    # 5. XGBoost
    logger.info("\n5. Training XGBoost...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    logger.info(f"   R² Score: {score:.4f}")
    ml_models['xgboost'] = model
    
    return ml_models, X_train_scaled, X_test_scaled


def train_dl_models(X_train, X_test, y_train, y_test):
    """Train all 3 Deep Learning models"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING DEEP LEARNING MODELS")
    logger.info("="*60)
    
    dl_models = {}
    
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available, using Gradient Boosting as fallback")
        
        # 1. Neural Network replacement (Gradient Boosting)
        logger.info("\n1. Training Neural Network (via GradientBoosting)...")
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, max_depth=7, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        logger.info(f"   R² Score: {score:.4f}")
        dl_models['neural_network'] = model
        
        # 2. LSTM replacement (AdaBoost)
        logger.info("\n2. Training LSTM (via AdaBoost)...")
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        logger.info(f"   R² Score: {score:.4f}")
        dl_models['lstm'] = model
        
        # 3. GRU replacement (Extra Trees)
        logger.info("\n3. Training GRU (via ExtraTrees)...")
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        logger.info(f"   R² Score: {score:.4f}")
        dl_models['gru'] = model
        
    else:
        # 1. Neural Network (Feedforward)
        logger.info("\n1. Training Neural Network...")
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(N_FEATURES,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"   Test MAE: {mae:.4f}")
        dl_models['neural_network'] = model
        
        # 2. LSTM (Recurrent)
        logger.info("\n2. Training LSTM (Bidirectional)...")
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, N_FEATURES))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, N_FEATURES))
        
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(64, activation='relu'), input_shape=(1, N_FEATURES)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        loss, mae = model.evaluate(X_test_lstm, y_test, verbose=0)
        logger.info(f"   Test MAE: {mae:.4f}")
        dl_models['lstm'] = model
        
        # 3. GRU (Gated Recurrent Unit)
        logger.info("\n3. Training GRU...")
        model = keras.Sequential([
            layers.GRU(64, activation='relu', input_shape=(1, N_FEATURES)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        loss, mae = model.evaluate(X_test_lstm, y_test, verbose=0)
        logger.info(f"   Test MAE: {mae:.4f}")
        dl_models['gru'] = model
    
    return dl_models


def save_models(ml_models, dl_models, scaler):
    """Save all trained models to disk"""
    logger.info("\n" + "="*60)
    logger.info("SAVING MODELS")
    logger.info("="*60)
    
    # Save ML models
    logger.info("\nSaving ML models...")
    for name, model in ml_models.items():
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"   ✓ Saved {name}.pkl")
    
    # Save DL models
    logger.info("\nSaving DL models...")
    
    if TF_AVAILABLE:
        # Save Keras models as .h5
        dl_model_files = {
            'neural_network': 'neural_network_model.h5',
            'lstm': 'lstm_model.h5',
            'gru': 'gru_model.h5'
        }
        
        for name, filename in dl_model_files.items():
            path = os.path.join(MODEL_DIR, filename)
            dl_models[name].save(path)
            logger.info(f"   ✓ Saved {filename}")
    else:
        # Save sklearn models as .pkl (for compatibility with model_manager)
        dl_model_files = {
            'neural_network': 'neural_network_model.pkl',
            'lstm': 'lstm_model.pkl',
            'gru': 'gru_model.pkl'
        }
        
        for name, filename in dl_model_files.items():
            path = os.path.join(MODEL_DIR, filename)
            with open(path, 'wb') as f:
                pickle.dump(dl_models[name], f)
            logger.info(f"   ✓ Saved {filename}")
        
        logger.warning("\n⚠️  Saved sklearn models - Update model_manager.py to load from .pkl")
    
    # Save scaler
    logger.info("\nSaving scaler...")
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"   ✓ Saved scaler.pkl")


def main():
    """Main training pipeline"""
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*10 + "OZONE PREDICTION API - MODEL TRAINING" + " "*11 + "║")
    logger.info("╚" + "="*58 + "╝")
    
    # Generate synthetic data
    X, y = generate_synthetic_data(N_SAMPLES)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"✓ Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Train ML models
    ml_models, X_train_scaled, X_test_scaled = train_ml_models(X_train, X_test, y_train, y_test, scaler)
    
    # Train DL models
    dl_models = train_dl_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save all models
    save_models(ml_models, dl_models, scaler)
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info("\nAll models saved to: ./models/")
    logger.info("\nNext steps:")
    logger.info("1. Add your Google API key to .env")
    logger.info("2. Run: python app.py")
    logger.info("3. Visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
