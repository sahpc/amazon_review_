from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
import pandas as pd
import joblib
import json
import os

from utils.features import extract_features

# =========================================================
# BASE DIR (IMPORTANTE PARA RENDER)
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Amazon Review Intelligence API",
    description="API NLP + ML para análisis de reseñas Amazon",
    version="1.0.0"
)

# =========================================================
# LOAD MODELS
# =========================================================

models = {}

@app.on_event("startup")
def load_models():
    global models

    MODELS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models"
    )

    models = {
        "logistic_regression": joblib.load(os.path.join(MODELS_PATH, "logistic_regression.pkl")),
        "random_forest": joblib.load(os.path.join(MODELS_PATH, "random_forest.pkl")),
        "xgboost": joblib.load(os.path.join(MODELS_PATH, "xgboost.pkl")),
        "lightgbm": joblib.load(os.path.join(MODELS_PATH, "lightgbm.pkl")),
        "catboost": joblib.load(os.path.join(MODELS_PATH, "catboost.pkl")),
    }

    print("✅ Models loaded:", list(models.keys()))
# =========================================================
# SCHEMAS
# =========================================================

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    score: int = Field(..., ge=1, le=5)

    model_name: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost"
    ] = "random_forest"

# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(models)
    }

# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(review: ReviewRequest):

    try:
        features = extract_features(review.text, review.score)
        X = pd.DataFrame([features])

        model = models[review.model_name]
        prob = model.predict_proba(X)[0][1]

        return {
            "prediction": int(prob >= 0.5),
            "probability": float(prob),
            "model_used": review.model_name,
            "features": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
