"""
Business logic and integrations (predictions, explanations, Google AI)
"""
import numpy as np
import logging
from typing import Dict, Optional, Any
from model_manager import model_manager
from config import settings
from schemas import PredictionRequest, ModelType

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionService:
    """Handles model predictions"""
    
    @staticmethod
    def get_features_array(request: PredictionRequest) -> np.ndarray:
        """Convert request to feature array"""
        features = np.array([
            request.pressure,
            request.day_of_week,
            request.is_weekend,
            request.week_of_year,
            request.quarter,
            request.day_of_year,
            request.is_holiday_season,
            request.hour_sin,
            request.hour_cos,
            request.month_sin,
            request.month_cos,
            request.day_of_week_sin,
            request.day_of_week_cos,
            request.ozone_lag1,
            request.ozone_lag3,
            request.ozone_lag6,
            request.ozone_lag24,
            request.ozone_rolling_mean_3h,
            request.ozone_rolling_std_3h,
            request.ozone_rolling_mean_24h,
            request.pressure_month_interaction,
            request.hour_pressure_interaction,
            request.pressure_squared,
            request.pressure_cubed,
            request.site_mean_ozone,
            request.site_std_ozone,
            request.month_hour_avg_ozone,
            request.hour_avg_ozone,
            request.month_avg_ozone,
            request.ozone_deviation_from_site_mean
        ], dtype=np.float32)
        
        return features
    
    @staticmethod
    def predict(model_name: str, request: PredictionRequest) -> Dict[str, Any]:
        """Make prediction with specified model"""
        try:
            # Get features
            features = PredictionService.get_features_array(request)
            
            # Scale features
            scaled_features = model_manager.scale_features(features)
            
            # Predict based on model type
            if model_name in settings.ML_MODELS:
                prediction = model_manager.predict_ml(model_name, scaled_features)
                confidence = PredictionService._get_ml_confidence(model_name)
            elif model_name in settings.DL_MODELS:
                prediction = model_manager.predict_dl(model_name, scaled_features)
                confidence = PredictionService._get_dl_confidence(model_name)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Clamp prediction to reasonable range (0-200 ppbv)
            prediction = max(0, min(200, float(prediction)))
            
            return {
                "model": model_name,
                "prediction": round(prediction, 2),
                "confidence": round(confidence, 3)
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    @staticmethod
    def predict_ensemble(request: PredictionRequest) -> Dict[str, Any]:
        """Get predictions from all models and return ensemble"""
        try:
            ml_predictions = []
            dl_predictions = []
            
            # Get ML predictions
            for model_name in settings.ML_MODELS:
                if model_name in model_manager.ml_models:
                    pred = PredictionService.predict(model_name, request)
                    ml_predictions.append(pred["prediction"])
            
            # Get DL predictions
            for model_name in settings.DL_MODELS:
                if model_name in model_manager.dl_models:
                    pred = PredictionService.predict(model_name, request)
                    dl_predictions.append(pred["prediction"])
            
            # Calculate ensemble
            all_predictions = ml_predictions + dl_predictions
            
            if not all_predictions:
                raise ValueError("No models available for prediction")
            
            ensemble_prediction = np.mean(all_predictions)
            ensemble_std = np.std(all_predictions)
            ensemble_confidence = 1.0 - (ensemble_std / 50.0)  # Normalize
            ensemble_confidence = max(0.1, min(1.0, ensemble_confidence))
            
            return {
                "ensemble_prediction": round(float(ensemble_prediction), 2),
                "std_deviation": round(float(ensemble_std), 2),
                "confidence": round(float(ensemble_confidence), 3),
                "ml_predictions": [round(p, 2) for p in ml_predictions],
                "dl_predictions": [round(p, 2) for p in dl_predictions],
                "models_used": len(all_predictions)
            }
        
        except Exception as e:
            logger.error(f"Ensemble prediction error: {str(e)}")
            raise
    
    @staticmethod
    def _get_ml_confidence(model_name: str) -> float:
        """Get confidence score for ML model (based on historical performance)"""
        confidence_map = {
            'dummy_baseline': 0.1,
            'linear_regression': 0.18,
            'decision_tree': 0.28,
            'random_forest': 0.45,
            'xgboost': 0.52
        }
        return confidence_map.get(model_name, 0.3)
    
    @staticmethod
    def _get_dl_confidence(model_name: str) -> float:
        """Get confidence score for DL model"""
        confidence_map = {
            'neural_network': 0.50,
            'lstm': 0.60,
            'gru': 0.59
        }
        return confidence_map.get(model_name, 0.5)


class ExplanationService:
    """Handles AI-powered explanations using Google AI Studio"""
    
    def __init__(self):
        self.available = GOOGLE_AI_AVAILABLE and settings.GOOGLE_API_KEY
        if self.available:
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.model = genai.GenerativeModel(model_name=settings.GOOGLE_MODEL_NAME)
                logger.info("✅ Google AI Studio configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure Google AI: {str(e)}")
                self.available = False
    
    def explain_prediction(self, prediction: float, model_name: str, 
                          request: PredictionRequest) -> Optional[str]:
        """Get AI explanation for prediction"""
        
        if not self.available:
            logger.warning("Google AI not available, returning generic explanation")
            return self._generic_explanation(prediction, model_name)
        
        try:
            # Build context for the model
            context = self._build_context(prediction, model_name, request)
            
            # Generate explanation
            response = self.model.generate_content(context)
            explanation = response.text
            
            return explanation
        
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return self._generic_explanation(prediction, model_name)
    
    def _build_context(self, prediction: float, model_name: str, 
                      request: PredictionRequest) -> str:
        """Build context for AI explanation"""
        
        severity = "LOW"
        if prediction > 100:
            severity = "CRITICAL"
        elif prediction > 70:
            severity = "HIGH"
        elif prediction > 50:
            severity = "MODERATE"
        
        context = f"""
You are an atmospheric science expert explaining ozone predictions. 

PREDICTION DETAILS:
- Predicted Ozone Level: {prediction:.2f} ppbv
- Severity: {severity}
- Model Used: {model_name}

ATMOSPHERIC CONDITION INPUTS:
- Pressure: {request.pressure:.2f} mb
- Day of Week: {request.day_of_week} (0=Monday, 6=Sunday)
- Is Weekend: {'Yes' if request.is_weekend else 'No'}
- Recent Ozone (1h ago): {request.ozone_lag1:.2f} ppbv
- Recent Ozone (24h ago): {request.ozone_lag24:.2f} ppbv
- Site Mean Ozone: {request.site_mean_ozone:.2f} ppbv
- 24h Rolling Average: {request.ozone_rolling_mean_24h:.2f} ppbv

Please provide a brief (2-3 sentences) explanation of:
1. Why the ozone level is at this level
2. What atmospheric factors are driving this prediction
3. Whether this is normal for the time/location

Be concise and accurate without overwhelming technical jargon.
"""
        return context
    
    def _generic_explanation(self, prediction: float, model_name: str) -> str:
        """Generate generic explanation without AI"""
        
        if prediction > 100:
            severity = "critically elevated"
            action = "immediate attention needed"
        elif prediction > 70:
            severity = "elevated"
            action = "monitoring recommended"
        elif prediction > 50:
            severity = "moderate"
            action = "within typical range but elevated"
        else:
            severity = "normal"
            action = "healthy levels"
        
        explanation = f"""
The predicted ozone level is {prediction:.2f} ppbv, which is {severity}. 
This prediction was made using the {model_name} model. {action.capitalize()}.
"""
        return explanation
    
    def get_recommendation(self, prediction: float) -> str:
        """Get health/policy recommendation based on prediction"""
        
        if prediction > 120:
            return "🔴 CRITICAL: Outdoor activities should be restricted. Vulnerable populations should stay indoors."
        elif prediction > 100:
            return "🟠 HIGH: Limit prolonged outdoor activities. Sensitive groups should take precautions."
        elif prediction > 70:
            return "🟡 MODERATE: Outdoor activities are acceptable. Sensitive groups should monitor conditions."
        elif prediction > 50:
            return "🟢 GOOD: Air quality is acceptable for outdoor activities."
        else:
            return "🟢 EXCELLENT: Air quality is very good. Ideal conditions for outdoor activities."


# Global service instances
prediction_service = PredictionService()
explanation_service = ExplanationService()
