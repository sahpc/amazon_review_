from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import os

from utils.features import extract_features

# =========================================================
# PATH BASE
# =========================================================

BASE_DIR = os.path.dirname(__file__)
MODELS_PATH = os.path.join(BASE_DIR, "..", "models")

# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Amazon Review Intelligence API",
    version="1.0.0"
)

# =========================================================
# MODELS
# =========================================================

models = {}

@app.on_event("startup")
def load_models():
    global models

    try:
        models = {
            "logistic_regression": joblib.load(os.path.join(MODELS_PATH, "logistic_regression.pkl")),
            "random_forest": joblib.load(os.path.join(MODELS_PATH, "random_forest.pkl")),
            "xgboost": joblib.load(os.path.join(MODELS_PATH, "xgboost.pkl")),
            "lightgbm": joblib.load(os.path.join(MODELS_PATH, "lightgbm.pkl")),
            "catboost": joblib.load(os.path.join(MODELS_PATH, "catboost.pkl")),
        }

        print("✅ Models loaded:", list(models.keys()))

    except Exception as e:
        print("❌ Error loading models:", str(e))
        models = {}

# =========================================================
# SCHEMA
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

@app.post("/predict")
def predict(review: ReviewRequest):

    if not models:
        raise HTTPException(503, "Models not loaded")

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

# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
