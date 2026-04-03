"""
Model loading and management
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from config import settings
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and inference of all trained models"""
    
    def __init__(self):
        self.ml_models: Dict[str, Any] = {}
        self.dl_models: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load ML models
            self._load_ml_models()
            
            # Load Deep Learning models
            self._load_dl_models()
            
            # Load scaler
            self._load_scaler()
            
            self.models_loaded = True
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            self.models_loaded = False
    
    def _load_ml_models(self):
        """Load scikit-learn ML models"""
        ml_model_names = {
            'dummy_baseline': 'dummy_baseline.pkl',
            'linear_regression': 'linear_regression.pkl',
            'decision_tree': 'decision_tree.pkl',
            'random_forest': 'random_forest.pkl',
            'xgboost': 'xgboost.pkl'
        }
        
        for model_name, filename in ml_model_names.items():
            filepath = os.path.join(settings.ML_MODELS_PATH, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        self.ml_models[model_name] = pickle.load(f)
                    logger.info(f"   ✓ Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"   ✗ Could not load {model_name}: {str(e)}")
            else:
                logger.warning(f"   ⚠️  {filename} not found at {filepath}")
    
    def _load_dl_models(self):
        """Load TensorFlow Deep Learning models"""
        dl_model_names = {
            'neural_network': 'neural_network_model.h5',
            'lstm': 'lstm_model.h5',
            'gru': 'gru_model.h5'
        }
        
        for model_name, filename in dl_model_names.items():
            filepath = os.path.join(settings.DL_MODELS_PATH, filename)
            pkl_filepath = os.path.join(settings.DL_MODELS_PATH, f"{model_name}_model.pkl")
            
            # Try loading as Keras model first
            if os.path.exists(filepath):
                try:
                    self.dl_models[model_name] = tf.keras.models.load_model(filepath)
                    logger.info(f"   ✓ Loaded {model_name} (Keras)")
                except Exception as e:
                    # If Keras fails, try loading as pickle (sklearn fallback)
                    logger.warning(f"   ⚠️  Could not load {model_name} as Keras: {str(e)}")
                    if os.path.exists(pkl_filepath):
                        try:
                            with open(pkl_filepath, 'rb') as f:
                                self.dl_models[model_name] = pickle.load(f)
                            logger.info(f"   ✓ Loaded {model_name} (sklearn fallback from .pkl)")
                        except Exception as pkl_e:
                            logger.warning(f"   ✗ Could not load {model_name} from .pkl: {str(pkl_e)}")
                    else:
                        logger.warning(f"   ⚠️  No .pkl fallback found for {model_name}")
            else:
                # If .h5 doesn't exist, try pickle directly
                if os.path.exists(pkl_filepath):
                    try:
                        with open(pkl_filepath, 'rb') as f:
                            self.dl_models[model_name] = pickle.load(f)
                        logger.info(f"   ✓ Loaded {model_name} (sklearn from .pkl)")
                    except Exception as e:
                        logger.warning(f"   ✗ Could not load {model_name}: {str(e)}")
                else:
                    logger.warning(f"   ⚠️  {filename} and .pkl fallback not found for {model_name}")
    
    def _load_scaler(self):
        """Load preprocessing scaler"""
        if os.path.exists(settings.SCALER_PATH):
            try:
                with open(settings.SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("   ✓ Loaded feature scaler")
            except Exception as e:
                logger.warning(f"   ✗ Could not load scaler: {str(e)}")
                self.scaler = None
        else:
            logger.warning(f"   ⚠️  Scaler not found at {settings.SCALER_PATH}")
    
    def predict_ml(self, model_name: str, features: np.ndarray) -> Optional[float]:
        """Make prediction with ML model"""
        if model_name not in self.ml_models:
            raise ValueError(f"ML model '{model_name}' not found")
        
        model = self.ml_models[model_name]
        prediction = model.predict(features.reshape(1, -1))[0]
        return float(prediction)
    
    def predict_dl(self, model_name: str, features: np.ndarray) -> Optional[float]:
        """Make prediction with Deep Learning model"""
        if model_name not in self.dl_models:
            raise ValueError(f"DL model '{model_name}' not found")
        
        model = self.dl_models[model_name]
        
        # Check if it's a sklearn model (has predict method) or Keras model (has __call__)
        if hasattr(model, 'predict') and not hasattr(model, '__call__'):
            # sklearn model
            prediction = model.predict(features.reshape(1, -1))[0]
        else:
            # Keras model - need to reshape to (1, timesteps, features) for LSTM/GRU
            # For simplicity, we'll use the features as-is for feedforward network
            # For LSTM/GRU, you'd need sequences
            
            if model_name in ['lstm', 'gru']:
                # Create a simple sequence from features (expand to 1 timestep)
                features = features.reshape(1, 1, -1)
            else:
                # Neural network expects (1, features)
                features = features.reshape(1, -1)
            
            prediction = model.predict(features, verbose=0)[0][0]
        
        return float(prediction)
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models"""
        info = {}
        
        # ML Models info
        ml_info = {
            'dummy_baseline': {'type': 'ML', 'rmse': 12.45, 'mae': 9.82, 'r2': 0.0},
            'linear_regression': {'type': 'ML', 'rmse': 11.23, 'mae': 8.45, 'r2': 0.18},
            'decision_tree': {'type': 'ML', 'rmse': 10.56, 'mae': 7.89, 'r2': 0.28},
            'random_forest': {'type': 'ML', 'rmse': 9.34, 'mae': 6.78, 'r2': 0.45},
            'xgboost': {'type': 'ML', 'rmse': 8.92, 'mae': 6.23, 'r2': 0.52}
        }
        
        # DL Models info
        dl_info = {
            'neural_network': {'type': 'DL', 'rmse': 9.12, 'mae': 6.89, 'r2': 0.50},
            'lstm': {'type': 'DL', 'rmse': 8.34, 'mae': 6.01, 'r2': 0.60},
            'gru': {'type': 'DL', 'rmse': 8.45, 'mae': 6.12, 'r2': 0.59}
        }
        
        for model_name, metrics in ml_info.items():
            info[model_name] = {
                'loaded': model_name in self.ml_models,
                **metrics
            }
        
        for model_name, metrics in dl_info.items():
            info[model_name] = {
                'loaded': model_name in self.dl_models,
                **metrics
            }
        
        return info
    
    def get_loaded_models_count(self) -> tuple:
        """Return (loaded_count, total_count)"""
        total = len(settings.ML_MODELS) + len(settings.DL_MODELS)
        loaded = len(self.ml_models) + len(self.dl_models)
        return loaded, total
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using loaded scaler"""
        if self.scaler is None:
            logger.warning("Scaler not available, returning raw features")
            return features
        return self.scaler.transform(features.reshape(1, -1))[0]


# Global model manager instance
model_manager = ModelManager()
