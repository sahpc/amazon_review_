
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import os

from utils.features import extract_features

# =========================================================
# BASE PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(

    title="Amazon Review Intelligence API",

    version="1.0.0"
)

# =========================================================
# MODELS
# =========================================================

models = {}

MODEL_FILES = {

    "logistic_regression":
        "logistic_regression.pkl",

    "random_forest":
        "random_forest.pkl",

    "xgboost":
        "xgboost.pkl",

    "lightgbm":
        "lightgbm.pkl",

    "catboost":
        "catboost.pkl",
}

# =========================================================
# LOAD MODELS
# =========================================================

@app.on_event("startup")
def load_models():

    global models

    print("\n==============================")
    print("Loading ML models...")
    print("==============================\n")

    for model_name, filename in MODEL_FILES.items():

        try:

            model_path = os.path.join(
                MODELS_PATH,
                filename
            )

            # =============================================
            # FILE VALIDATION
            # =============================================

            if not os.path.exists(model_path):

                print(
                    f"❌ File not found: {filename}"
                )

                continue

            # =============================================
            # LOAD MODEL
            # =============================================

            models[model_name] = joblib.load(
                model_path
            )

            print(
                f"✅ Loaded: {model_name}"
            )

        except Exception as e:

            print(
                f"❌ Error loading {model_name}: {e}"
            )

    print("\n==============================")
    print(
        f"Total models loaded: {len(models)}"
    )
    print("==============================\n")

# =========================================================
# REQUEST SCHEMA
# =========================================================

class ReviewRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=10,
        max_length=5000
    )

    score: int = Field(
        ...,
        ge=1,
        le=5
    )

    model_name: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost"
    ] = "random_forest"

# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "Amazon Review Intelligence API Running 🚀"
    }

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "models_loaded":
            len(models),

        "available_models":
            list(models.keys())
    }

# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post("/predict")
def predict(review: ReviewRequest):

    # =====================================================
    # MODELS VALIDATION
    # =====================================================

    if not models:

        raise HTTPException(

            status_code=503,

            detail="Models not loaded"
        )

    # =====================================================
    # MODEL VALIDATION
    # =====================================================

    if review.model_name not in models:

        raise HTTPException(

            status_code=404,

            detail=f"""
Model '{review.model_name}' not available
"""
        )

    try:

        # =================================================
        # FEATURE EXTRACTION
        # =================================================

        features = extract_features(

            review.text,

            review.score
        )

        # =================================================
        # DATAFRAME
        # =================================================

        X = pd.DataFrame([features])

        # =================================================
        # MODEL
        # =================================================

        model = models[
            review.model_name
        ]

        # =================================================
        # PREDICTION
        # =================================================

        prob = model.predict_proba(X)[0][1]

        prediction = int(prob >= 0.5)

        # =================================================
        # RESPONSE
        # =================================================

        return {

            "prediction":
                prediction,

            "probability":
                round(float(prob), 4),

            "model_used":
                review.model_name,

            "features":
                features
        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"""
Prediction error:
{str(e)}
"""
        )

# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api.main:app",

        host="0.0.0.0",

        port=8000,

        reload=True
    )
