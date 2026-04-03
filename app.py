"""
FastAPI Application - Ozone Prediction API
Production-ready ML/DL models serving with Google AI explanations
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
from config import settings
from schemas import (
    PredictionRequest, BatchPredictionRequest, PredictionResponse, 
    ExplanationResponse, HealthResponse, ModelType
)
from services import prediction_service, explanation_service
from model_manager import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 1. HEALTH & INFO ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns the status of the API and loaded models
    """
    loaded, total = model_manager.get_loaded_models_count()
    
    return HealthResponse(
        status="healthy" if loaded > 0 else "degraded",
        version=settings.APP_VERSION,
        models_loaded=loaded,
        total_models=total,
        google_ai_available=explanation_service.available
    )


@app.get("/info", tags=["System"])
async def get_api_info():
    """Get detailed API and model information"""
    
    loaded, total = model_manager.get_loaded_models_count()
    model_info = model_manager.get_model_info()
    
    return {
        "api": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION
        },
        "models": {
            "total": total,
            "loaded": loaded,
            "details": model_info
        },
        "features": {
            "google_ai_available": explanation_service.available,
            "total_features": len(settings.FEATURE_COLUMNS),
            "feature_names": settings.FEATURE_COLUMNS
        }
    }


# ============================================
# 2. SINGLE PREDICTION ENDPOINTS
# ============================================

@app.post("/predict/{model_name}", response_model=PredictionResponse, tags=["Predictions"])
async def predict(model_name: str, request: PredictionRequest):
    """
    Make a prediction with a specific model
    
    **Parameters:**
    - `model_name`: One of: dummy_baseline, linear_regression, decision_tree, random_forest, xgboost, neural_network, lstm, gru
    
    **Returns:**
    - `model`: Name of the model used
    - `prediction`: Predicted ozone level in ppbv
    - `confidence`: Model confidence (0-1)
    """
    
    try:
        # Validate model name
        if model_name not in settings.ML_MODELS + settings.DL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Choose from: {settings.ML_MODELS + settings.DL_MODELS}"
            )
        
        # Check if model is loaded
        if (model_name not in model_manager.ml_models and 
            model_name not in model_manager.dl_models):
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' is not loaded. Check /health for details."
            )
        
        # Get prediction
        result = prediction_service.predict(model_name, request)
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-all", tags=["Predictions"])
async def predict_all(request: PredictionRequest):
    """
    Get predictions from all available models
    
    Returns predictions from all 8 models (5 ML + 3 DL) in a single request
    """
    
    try:
        results = {
            "ml_models": {},
            "dl_models": {},
            "ensemble": None
        }
        
        # Get predictions from all ML models
        for model_name in settings.ML_MODELS:
            if model_name in model_manager.ml_models:
                try:
                    pred = prediction_service.predict(model_name, request)
                    results["ml_models"][model_name] = pred
                except Exception as e:
                    logger.warning(f"Failed to predict with {model_name}: {str(e)}")
        
        # Get predictions from all DL models
        for model_name in settings.DL_MODELS:
            if model_name in model_manager.dl_models:
                try:
                    pred = prediction_service.predict(model_name, request)
                    results["dl_models"][model_name] = pred
                except Exception as e:
                    logger.warning(f"Failed to predict with {model_name}: {str(e)}")
        
        # Get ensemble prediction
        try:
            results["ensemble"] = prediction_service.predict_ensemble(request)
        except Exception as e:
            logger.warning(f"Failed to compute ensemble: {str(e)}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in predict_all: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-ensemble", tags=["Predictions"])
async def predict_ensemble(request: PredictionRequest):
    """
    Get ensemble prediction from all models
    
    Averages predictions from all available models for robust predictions
    """
    
    try:
        result = prediction_service.predict_ensemble(request)
        return result
    except Exception as e:
        logger.error(f"Ensemble prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 3. BATCH PREDICTION ENDPOINTS
# ============================================

@app.post("/batch-predict/{model_name}", tags=["Predictions"])
async def batch_predict(model_name: str, request: BatchPredictionRequest):
    """
    Make predictions for multiple samples at once
    
    Useful for processing multiple weather conditions in a single request
    """
    
    try:
        if model_name not in settings.ML_MODELS + settings.DL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Choose from: {settings.ML_MODELS + settings.DL_MODELS}"
            )
        
        predictions = []
        for sample in request.samples:
            try:
                pred = prediction_service.predict(model_name, sample)
                predictions.append(PredictionResponse(**pred))
            except Exception as e:
                logger.warning(f"Failed to predict sample: {str(e)}")
        
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
        
        return {
            "total_samples": len(request.samples),
            "successful_predictions": len(predictions),
            "predictions": predictions,
            "average_confidence": round(avg_confidence, 3)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 4. EXPLANATION ENDPOINTS (WITH AI)
# ============================================

@app.post("/explain/{model_name}", response_model=ExplanationResponse, tags=["Explanations"])
async def explain_prediction(model_name: str, request: PredictionRequest):
    """
    Get prediction with AI-powered explanation using Google Gemini
    
    Provides intelligent insight into why the model made this prediction
    """
    
    try:
        # Get prediction
        pred_result = prediction_service.predict(model_name, request)
        prediction = pred_result["prediction"]
        confidence = pred_result["confidence"]
        
        # Get explanation
        explanation = explanation_service.explain_prediction(
            prediction, model_name, request
        )
        
        # Get recommendation
        recommendation = explanation_service.get_recommendation(prediction)
        
        return ExplanationResponse(
            prediction=prediction,
            model_used=model_name,
            explanation=explanation,
            confidence=confidence,
            factors={
                "recommendation": recommendation,
                "severity": "CRITICAL" if prediction > 100 else "HIGH" if prediction > 70 else "MODERATE" if prediction > 50 else "GOOD"
            }
        )
    
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain-ensemble", tags=["Explanations"])
async def explain_ensemble(request: PredictionRequest):
    """
    Get ensemble prediction with comprehensive AI explanation
    
    Shows predictions from all models plus AI insights
    """
    
    try:
        ensemble_result = prediction_service.predict_ensemble(request)
        ensemble_pred = ensemble_result["ensemble_prediction"]
        
        # Get explanation for ensemble
        explanation = explanation_service.explain_prediction(
            ensemble_pred, "ensemble", request
        )
        
        recommendation = explanation_service.get_recommendation(ensemble_pred)
        
        return {
            "ensemble_prediction": ensemble_pred,
            "explanation": explanation,
            "recommendation": recommendation,
            "confidence": ensemble_result["confidence"],
            "models_comparison": {
                "ml_predictions": ensemble_result["ml_predictions"],
                "dl_predictions": ensemble_result["dl_predictions"],
                "std_deviation": ensemble_result["std_deviation"]
            }
        }
    
    except Exception as e:
        logger.error(f"Ensemble explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 5. COMPARISON ENDPOINTS
# ============================================

@app.post("/compare-models", tags=["Comparisons"])
async def compare_models(request: PredictionRequest):
    """
    Compare predictions across all available models
    
    Shows model-by-model predictions for the same input
    """
    
    try:
        comparison = {
            "ml_models": [],
            "dl_models": []
        }
        
        # ML models comparison
        for model_name in settings.ML_MODELS:
            if model_name in model_manager.ml_models:
                try:
                    pred = prediction_service.predict(model_name, request)
                    comparison["ml_models"].append({
                        "model": model_name,
                        **pred
                    })
                except Exception as e:
                    logger.warning(f"Failed to predict with {model_name}: {str(e)}")
        
        # DL models comparison
        for model_name in settings.DL_MODELS:
            if model_name in model_manager.dl_models:
                try:
                    pred = prediction_service.predict(model_name, request)
                    comparison["dl_models"].append({
                        "model": model_name,
                        **pred
                    })
                except Exception as e:
                    logger.warning(f"Failed to predict with {model_name}: {str(e)}")
        
        # Calculate statistics
        all_preds = [m["prediction"] for m in comparison["ml_models"] + comparison["dl_models"]]
        
        return {
            "comparison": comparison,
            "statistics": {
                "min_prediction": round(min(all_preds), 2),
                "max_prediction": round(max(all_preds), 2),
                "avg_prediction": round(sum(all_preds) / len(all_preds), 2),
                "std_deviation": round(np.std(all_preds), 2) if len(all_preds) > 1 else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/", tags=["System"])
async def root():
    """API root - provides quick start guide"""
    return {
        "message": "🌍 Welcome to Ozone Prediction API",
        "version": settings.APP_VERSION,
        "quick_start": {
            "health": "GET /health",
            "info": "GET /info",
            "single_prediction": "POST /predict/{model_name}",
            "all_models": "POST /predict-all",
            "ensemble": "POST /predict-ensemble",
            "with_explanation": "POST /explain/{model_name}",
            "compare": "POST /compare-models"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "models_available": {
            "ml": settings.ML_MODELS,
            "dl": settings.DL_MODELS
        }
    }


# ============================================
# STARTUP & SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    loaded, total = model_manager.get_loaded_models_count()
    logger.info(f"📊 Models loaded: {loaded}/{total}")
    if explanation_service.available:
        logger.info("🤖 Google AI Studio available for explanations")
    else:
        logger.warning("⚠️  Google AI Studio not available (set GOOGLE_API_KEY)")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    logger.info(f"🛑 Shutting down {settings.APP_NAME}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
