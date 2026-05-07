# 🔮 Time Series Forecasting System

A **production-grade**, end-to-end forecasting backend that trains **SARIMA, Prophet, XGBoost, and LSTM** models per US state, automatically selects the best performer, and serves 8-week sales forecasts through a **FastAPI REST API**.

---

## 🏗️ Architecture

```
HTTP Request
     │
     ▼
┌─────────────┐
│  FastAPI    │  /train  /predict  /models/compare  /health
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                  Service Layer                       │
│  TrainingService          PredictionService          │
└───────────┬─────────────────────────┬───────────────┘
            │                         │
            ▼                         ▼
┌───────────────────┐       ┌──────────────────┐
│ ForecastingPipeline│       │  ModelRegistry   │
│  (per-state loop) │       │  (disk + cache)  │
└────────┬──────────┘       └──────────────────┘
         │
    ┌────┴─────────────────────────────┐
    │         StateTrainer             │
    │  fit → evaluate → select → save  │
    └────┬──────┬──────┬────────┬─────┘
         │      │      │        │
      SARIMA Prophet XGBoost  LSTM
```

---

## 📂 Project Structure

```
forecasting_system/
├── app/
│   ├── api/routes/          # FastAPI route handlers
│   ├── core/                # Config, logger, constants
│   ├── data/                # Loader, preprocessor, feature engineering
│   ├── models/              # SARIMA, Prophet, XGBoost, LSTM (all extend BaseForecaster)
│   ├── training/            # Trainer, evaluator, selector, pipeline
│   ├── services/            # Training & prediction service layer
│   ├── schemas/             # Pydantic request/response models
│   ├── utils/               # Metrics, visualization, helpers
│   └── main.py              # FastAPI app factory
├── saved_models/            # Persisted model artefacts (per state/model)
├── logs/                    # Rotating log files
├── plots/                   # Auto-generated forecast & comparison charts
├── tests/                   # Pytest unit tests
├── data/                    # Place your Excel file here
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## 🚀 Quick Start

### 1. Local Setup

```bash
git clone <repo>
cd forecasting_system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Place your Excel file
cp "Forecasting Case-Study.xlsx" data/sales_data.xlsx

# Start the API
uvicorn app.main:app --reload --port 8000
```

### 2. Docker

```bash
cp "Forecasting Case-Study.xlsx" data/sales_data.xlsx
docker-compose up --build
```

API will be live at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "healthy", "version": "1.0.0" }
```

---

### `POST /train`
Trigger full training pipeline for all states.

**Request**
```json
{
  "data_path": null,
  "models": ["xgboost", "prophet", "sarima", "lstm"],
  "parallel": false
}
```

**Response**
```json
{
  "status": "success",
  "states_trained": ["California", "Texas", "..."],
  "results": {
    "California": {
      "best_model": "xgboost",
      "metrics": {
        "xgboost": { "mae": 1200.5, "rmse": 1800.2, "mape": 3.4 },
        "prophet": { "mae": 1500.1, "rmse": 2200.0, "mape": 4.1 }
      }
    }
  }
}
```

---

### `POST /predict`
Forecast next N weeks for a given state.

**Request**
```json
{
  "state": "California",
  "horizon": 8
}
```

**Response**
```json
{
  "state": "California",
  "best_model": "xgboost",
  "forecast_horizon_weeks": 8,
  "forecast": [
    { "week": 1, "prediction": 444000000 },
    { "week": 2, "prediction": 451000000 }
  ]
}
```

---

### `GET /models/compare?state=California`
Return validation metrics for all models.

---

## 🧠 Model Details

| Model | Approach | Strengths |
|-------|----------|-----------|
| **SARIMA** | Statistical, auto order selection via ADF + AIC grid | Interpretable, captures seasonal patterns |
| **Prophet** | Additive decomposition, US holidays | Handles missing dates, changepoints |
| **XGBoost** | Supervised lag features + recursive forecasting | Fast, high accuracy, feature importance |
| **LSTM** | Sliding-window deep learning + recursive | Complex patterns, long-term dependencies |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error ← primary selection metric |
| MAPE | Mean Absolute Percentage Error |

Best model = **lowest RMSE** on hold-out validation window (last 12 weeks).

---

## ⚙️ Configuration (`.env`)

```ini
FORECAST_HORIZON=8      # weeks to forecast
VALIDATION_WEEKS=12     # hold-out window
DATA_PATH=data/sales_data.xlsx
API_PORT=8000
DEBUG=false
```

---

## 🔒 Data Leakage Prevention

- **Chronological split only** — no random shuffling
- **Lag features** use `.shift(N)` — only past observations
- **Rolling features** shift by 1 before windowing
- **Recursive forecasting** appends predictions as pseudo-actuals for next step

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📈 Generated Outputs

After training, find in `plots/`:
- `{State}_{Model}_forecast.png` — historical + 8-week forecast
- `{State}_{Model}_residuals.png` — residual analysis
- `{State}_model_comparison.png` — RMSE bar chart

---

## 🗺️ Dataset Format

| Column | Type | Description |
|--------|------|-------------|
| State | string | US state name |
| Date | date | Week start date |
| Total | numeric | Weekly sales |
| Category | string | Product category (optional) |
