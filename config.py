"""
Configuration management for Ozone Prediction API
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Configuration
    APP_NAME: str = "Ozone Prediction API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "ML/DL models serving ozone predictions with AI explanations"
    DEBUG: bool = False
    
    # API Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model Configuration
    ML_MODELS_PATH: str = "./models/"
    DL_MODELS_PATH: str = "./models/"
    SCALER_PATH: str = "./models/scaler.pkl"
    
    # Google AI Studio Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL_NAME: str = "gemini-pro"
    
    # Feature Configuration
    FEATURE_COLUMNS: list = [
        'Pressure', 'DayOfWeek', 'IsWeekend', 'WeekOfYear', 'Quarter', 'DayOfYear',
        'IsHolidaySeason', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Ozone_lag1', 'Ozone_lag3', 'Ozone_lag6',
        'Ozone_lag24', 'Ozone_rolling_mean_3h', 'Ozone_rolling_std_3h',
        'Ozone_rolling_mean_24h', 'Pressure_Month_interaction', 'Hour_Pressure_interaction',
        'Pressure_squared', 'Pressure_cubed', 'Site_mean_ozone', 'Site_std_ozone',
        'Month_Hour_avg_ozone', 'Hour_avg_ozone', 'Month_avg_ozone',
        'Ozone_deviation_from_site_mean'
    ]
    
    # Model Names
    ML_MODELS: list = [
        'dummy_baseline',
        'linear_regression',
        'decision_tree',
        'random_forest',
        'xgboost'
    ]
    
    DL_MODELS: list = [
        'neural_network',
        'lstm',
        'gru'
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Load settings
settings = Settings()
