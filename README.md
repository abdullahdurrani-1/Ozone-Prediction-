# Ozone Prediction API

Production-ready FastAPI service for atmospheric ozone forecasting with machine learning, deep learning, and optional AI-generated explanations.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-4CAF50?style=flat-square)](LICENSE)

<p align="center">
  <img src="assets/ozone-hero.svg" alt="Ozone Prediction API Hero Banner" width="100%"/>
</p>

## What this project does

This API predicts ozone concentration (`ppbv`) using:

- 5 classical ML models (`dummy_baseline`, `linear_regression`, `decision_tree`, `random_forest`, `xgboost`)
- 3 DL models (`neural_network`, `lstm`, `gru`)
- Ensemble prediction and model-to-model comparison
- Optional natural-language explanations powered by Google Gemini

It also includes a web dashboard (`index.html`, `dashboard.html`) and Docker support for deployment.

## Visual workflow

### Prediction pipeline (animated)

![Prediction Workflow](assets/workflow-animated.svg)

### MLOps lifecycle (animated)

![MLOps Lifecycle](assets/mlops-lifecycle-animated.svg)

## Features

- FastAPI with typed request/response schemas
- Single-model, all-model, and ensemble prediction endpoints
- Batch prediction endpoint
- AI explanation endpoints
- Health and system info endpoints
- Config via environment variables
- Dockerfile and docker-compose for easy deployment

## Tech stack

- Python 3.11+
- FastAPI + Uvicorn
- scikit-learn, XGBoost, TensorFlow/Keras
- Pydantic + pydantic-settings
- Google Generative AI SDK (optional)

## Project structure

```text
ozone-api/
├── app.py
├── config.py
├── schemas.py
├── model_manager.py
├── services.py
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── index.html
├── dashboard.html
├── train_models.py
├── test_api.py
├── validate_system.py
└── README.md
```

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ozone-api.git
cd ozone-api
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` from template:

```bash
cp .env.example .env
```

Set values in `.env`:

```ini
GOOGLE_API_KEY=your_google_api_key_here
DEBUG=False
HOST=0.0.0.0
PORT=8000
ML_MODELS_PATH=./models/
DL_MODELS_PATH=./models/
SCALER_PATH=./models/scaler.pkl
```

Note: `GOOGLE_API_KEY` is optional unless you use explanation endpoints.

### 3. Prepare models

Place trained model files in `models/` or train them:

```bash
python train_models.py
```

### 4. Run API

```bash
python app.py
```

Or with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API endpoints

### System

- `GET /`
- `GET /health`
- `GET /info`

### Predictions

- `POST /predict/{model_name}`
- `POST /predict-all`
- `POST /predict-ensemble`
- `POST /batch-predict/{model_name}`
- `POST /compare-models`

### Explanations

- `POST /explain/{model_name}`
- `POST /explain-ensemble`

## Example request

```bash
curl -X POST "http://localhost:8000/predict/random_forest" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Docker

Build:

```bash
docker build -t ozone-api:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env ozone-api:latest
```

Compose:

```bash
docker-compose up -d
```

## Testing

```bash
python test_api.py
python validate_system.py
```

## Security and GitHub checklist

- Do not commit real API keys
- Keep `.env` private (already ignored)
- Commit only `.env.example` with placeholders
- Keep large artifacts (models, datasets, logs, notebooks) out of Git unless intentionally versioned

## License

MIT. See [LICENSE](LICENSE).
