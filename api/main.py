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

    # ==============================
    # 1. VALIDAR MODELOS
    # ==============================
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs."
        )

    if review.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{review.model_name}' not available"
        )

    try:
        # ==============================
        # 2. FEATURE ENGINEERING
        # ==============================
        features = extract_features(review.text, review.score)

        if not features:
            raise ValueError("Feature extraction returned empty result")

        X = pd.DataFrame([features])

        # ==============================
        # 3. LOAD MODEL
        # ==============================
        model = models.get(review.model_name)

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Model not found in memory"
            )

        # ==============================
        # 4. PREDICTION SAFETY
        # ==============================
        if not hasattr(model, "predict_proba"):
            raise HTTPException(
                status_code=500,
                detail="Model does not support predict_proba"
            )

        prob = model.predict_proba(X)

        if prob is None or len(prob) == 0:
            raise ValueError("Model returned invalid probability output")

        probability = float(prob[0][1])
        prediction = int(probability >= 0.5)

        # ==============================
        # 5. RESPONSE
        # ==============================
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_used": review.model_name,
            "features_count": len(features),
            "status": "success"
        }

    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail=f"Data error: {str(ve)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal prediction error: {str(e)}"
        )

# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
