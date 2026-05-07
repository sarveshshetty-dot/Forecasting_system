"""
constants.py - Shared constants used across the system.
"""

MODEL_SARIMA = "sarima"
MODEL_PROPHET = "prophet"
MODEL_XGBOOST = "xgboost"
MODEL_LSTM = "lstm"
MODEL_ENSEMBLE = "ensemble"

ALL_MODELS = [MODEL_SARIMA, MODEL_PROPHET, MODEL_XGBOOST, MODEL_LSTM]

METRIC_MAE = "mae"
METRIC_RMSE = "rmse"
METRIC_MAPE = "mape"

STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_TRAINING = "training"

FILE_EXT_JOBLIB = ".joblib"
FILE_EXT_PICKLE = ".pkl"
FILE_EXT_JSON = ".json"
