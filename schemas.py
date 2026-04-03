"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Enumeration of available model types"""
    DUMMY_BASELINE = "dummy_baseline"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"


class PredictionRequest(BaseModel):
    """Input schema for single prediction"""
    
    pressure: float = Field(..., description="Atmospheric pressure (mb)", ge=0)
    day_of_week: int = Field(..., description="Day of week (0-6)", ge=0, le=6)
    is_weekend: int = Field(..., description="Is weekend (0 or 1)", ge=0, le=1)
    week_of_year: int = Field(..., description="Week of year (1-53)", ge=1, le=53)
    quarter: int = Field(..., description="Quarter (1-4)", ge=1, le=4)
    day_of_year: int = Field(..., description="Day of year (1-366)", ge=1, le=366)
    is_holiday_season: int = Field(..., description="Is holiday season (0 or 1)", ge=0, le=1)
    hour_sin: float = Field(..., description="Hour sine component (-1 to 1)")
    hour_cos: float = Field(..., description="Hour cosine component (-1 to 1)")
    month_sin: float = Field(..., description="Month sine component (-1 to 1)")
    month_cos: float = Field(..., description="Month cosine component (-1 to 1)")
    day_of_week_sin: float = Field(..., description="Day of week sine component (-1 to 1)")
    day_of_week_cos: float = Field(..., description="Day of week cosine component (-1 to 1)")
    ozone_lag1: float = Field(..., description="Ozone 1 hour lag (ppbv)")
    ozone_lag3: float = Field(..., description="Ozone 3 hour lag (ppbv)")
    ozone_lag6: float = Field(..., description="Ozone 6 hour lag (ppbv)")
    ozone_lag24: float = Field(..., description="Ozone 24 hour lag (ppbv)")
    ozone_rolling_mean_3h: float = Field(..., description="Ozone 3h rolling mean (ppbv)")
    ozone_rolling_std_3h: float = Field(..., description="Ozone 3h rolling std (ppbv)")
    ozone_rolling_mean_24h: float = Field(..., description="Ozone 24h rolling mean (ppbv)")
    pressure_month_interaction: float = Field(..., description="Pressure × Month interaction")
    hour_pressure_interaction: float = Field(..., description="Hour × Pressure interaction")
    pressure_squared: float = Field(..., description="Pressure²")
    pressure_cubed: float = Field(..., description="Pressure³")
    site_mean_ozone: float = Field(..., description="Site mean ozone (ppbv)")
    site_std_ozone: float = Field(..., description="Site std ozone (ppbv)")
    month_hour_avg_ozone: float = Field(..., description="Month-Hour avg ozone (ppbv)")
    hour_avg_ozone: float = Field(..., description="Hour avg ozone (ppbv)")
    month_avg_ozone: float = Field(..., description="Month avg ozone (ppbv)")
    ozone_deviation_from_site_mean: float = Field(..., description="Ozone deviation from site mean")

    class Config:
        json_schema_extra = {
            "example": {
                "pressure": 1013.25,
                "day_of_week": 3,
                "is_weekend": 0,
                "week_of_year": 15,
                "quarter": 2,
                "day_of_year": 100,
                "is_holiday_season": 0,
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "month_sin": 0.866,
                "month_cos": 0.5,
                "day_of_week_sin": 0.6,
                "day_of_week_cos": 0.8,
                "ozone_lag1": 45.2,
                "ozone_lag3": 44.8,
                "ozone_lag6": 43.5,
                "ozone_lag24": 40.2,
                "ozone_rolling_mean_3h": 45.0,
                "ozone_rolling_std_3h": 2.5,
                "ozone_rolling_mean_24h": 42.0,
                "pressure_month_interaction": 10132.5,
                "hour_pressure_interaction": 3039.75,
                "pressure_squared": 1026576.5625,
                "pressure_cubed": 1040049887.2,
                "site_mean_ozone": 42.5,
                "site_std_ozone": 8.2,
                "month_hour_avg_ozone": 46.3,
                "hour_avg_ozone": 43.1,
                "month_avg_ozone": 41.8,
                "ozone_deviation_from_site_mean": 2.7
            }
        }


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions"""
    
    samples: List[PredictionRequest] = Field(..., description="List of prediction requests")


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    
    model: ModelType
    prediction: float = Field(..., description="Predicted ozone level (ppbv)")
    confidence: Optional[float] = Field(None, description="Model confidence (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "random_forest",
                "prediction": 45.23,
                "confidence": 0.87
            }
        }


class ExplanationResponse(BaseModel):
    """Response schema for AI explanation"""
    
    prediction: float
    model_used: ModelType
    explanation: str
    confidence: Optional[float]
    factors: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 45.23,
                "model_used": "random_forest",
                "explanation": "The ozone level is predicted at 45.23 ppbv, which is moderately elevated. This is driven by high afternoon temperatures and moderate wind speeds from the northeast.",
                "confidence": 0.87,
                "factors": {
                    "temperature_effect": "high",
                    "wind_effect": "moderate",
                    "pressure_effect": "low"
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    total_samples: int
    predictions: List[PredictionResponse]
    average_confidence: Optional[float]


class ModelInfo(BaseModel):
    """Information about a single model"""
    
    name: ModelType
    model_type: str = Field(..., description="Type: ML or DL")
    accuracy_metric: str = Field(..., description="R² score on test set")
    rmse: float
    mae: float
    parameters: int = Field(..., description="Model parameters count")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str
    version: str
    models_loaded: int
    total_models: int
    google_ai_available: bool
