# 📋 API Quick Reference

## 🚀 Quick Start

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Google API key

# 2. Run
python app.py

# 3. Test
# Visit: http://localhost:8000/docs
# Or: python test_api.py
```

---

## 🔗 Core Endpoints

### System
```
GET  /              → API welcome & quick info
GET  /health        → Health check
GET  /info          → Detailed API info
```

### Predictions (Single Model)
```
POST /predict/{model_name}           → Single prediction
POST /predict-all                     → All models
POST /predict-ensemble                → Ensemble average
POST /batch-predict/{model_name}      → Batch processing
```

### With AI Explanations
```
POST /explain/{model_name}            → Prediction + explanation
POST /explain-ensemble                → Ensemble + explanation
```

### Comparisons
```
POST /compare-models                  → All models comparison
```

---

## 🎯 Model Names

### ML Models
- `dummy_baseline`
- `linear_regression`
- `decision_tree`
- `random_forest`
- `xgboost` (⭐ best ML)

### DL Models
- `neural_network`
- `lstm` (⭐ best DL)
- `gru`

---

## 📝 Request Format

```json
{
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
```

---

## 🐍 Python Examples

### Single Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/random_forest",
    json=sample_data
)
prediction = response.json()
print(f"Prediction: {prediction['prediction']:.2f} ppbv")
print(f"Confidence: {prediction['confidence']:.2f}")
```

### With Explanation
```python
response = requests.post(
    "http://localhost:8000/explain/lstm",
    json=sample_data
)
data = response.json()
print(data['explanation'])
print(data['factors']['recommendation'])
```

### Ensemble
```python
response = requests.post(
    "http://localhost:8000/predict-ensemble",
    json=sample_data
)
data = response.json()
print(f"Ensemble: {data['ensemble_prediction']:.2f} ppbv")
print(f"Std Dev: {data['std_deviation']:.2f}")
```

---

## 💻 cURL Examples

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict/random_forest" \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# Ensemble
curl -X POST "http://localhost:8000/predict-ensemble" \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# With explanation
curl -X POST "http://localhost:8000/explain/lstm" \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# Compare all models
curl -X POST "http://localhost:8000/compare-models" \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# Health check
curl http://localhost:8000/health
```

---

## 🐳 Docker Commands

```bash
# Build
docker build -t ozone-api:latest .

# Run
docker run -d -p 8000:8000 --env-file .env ozone-api:latest

# Compose
docker-compose up -d
docker-compose logs -f
docker-compose down

# Push to Hub
docker push username/ozone-api:latest
```

---

## 📊 Response Examples

### Prediction Response
```json
{
  "model": "random_forest",
  "prediction": 45.23,
  "confidence": 0.87
}
```

### Explanation Response
```json
{
  "prediction": 45.23,
  "model_used": "random_forest",
  "explanation": "The ozone level is predicted at 45.23 ppbv, which is moderately elevated...",
  "confidence": 0.87,
  "factors": {
    "recommendation": "🟡 MODERATE: Outdoor activities are acceptable...",
    "severity": "MODERATE"
  }
}
```

### Ensemble Response
```json
{
  "ensemble_prediction": 45.30,
  "std_deviation": 2.15,
  "confidence": 0.85,
  "ml_predictions": [45.2, 46.1, 44.8],
  "dl_predictions": [45.5, 44.9],
  "models_used": 5
}
```

### Comparison Response
```json
{
  "comparison": {
    "ml_models": [...],
    "dl_models": [...]
  },
  "statistics": {
    "min_prediction": 42.1,
    "max_prediction": 48.5,
    "avg_prediction": 45.3,
    "std_deviation": 2.1
  }
}
```

---

## ⚙️ Configuration

### .env File
```bash
GOOGLE_API_KEY=your_key_here
DEBUG=False
HOST=0.0.0.0
PORT=8000
ML_MODELS_PATH=./models/
DL_MODELS_PATH=./models/
SCALER_PATH=./models/scaler.pkl
```

### Models Directory
```
models/
├── dummy_baseline.pkl
├── linear_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── xgboost.pkl
├── neural_network_model.h5
├── lstm_model.h5
├── gru_model.h5
└── scaler.pkl
```

---

## 🔍 Debugging

```bash
# Check API status
curl http://localhost:8000/health

# View detailed info
curl http://localhost:8000/info

# Check logs
docker logs ozone-api

# Test connectivity
python test_api.py

# Profile request
time curl -X POST http://localhost:8000/predict/xgboost \
  -H "Content-Type: application/json" -d @sample_request.json
```

---

## 📈 Performance Tips

| Goal | Action |
|------|--------|
| Faster | Use GRU instead of LSTM |
| Most accurate | Use Bidirectional LSTM |
| Robust | Use ensemble prediction |
| Scalable | Use Docker + load balancer |
| Low latency | Deploy on GPU |

---

## 🚀 Deployment Quick Links

- **Local**: `python app.py`
- **Docker**: `docker-compose up -d`
- **AWS**: See DEPLOYMENT.md
- **Google Cloud**: `gcloud run deploy ozone-api --source .`
- **Heroku**: `git push heroku main`

---

## 📚 Documentation
- Full README: `README.md`
- Deployment: `DEPLOYMENT.md`
- API Docs: `http://localhost:8000/docs`

---

**Made with ❤️ for production ML**
