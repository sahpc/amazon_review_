
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
import pandas as pd
import joblib
import json
import os

from utils.features import extract_features

# =========================================================
# CONFIGURACION FASTAPI
# =========================================================

app = FastAPI(
    title="Amazon Review Intelligence API",
    description="""
API de NLP y Machine Learning para analizar la utilidad de reseñas Amazon.

Características:
- Predicción de helpfulness
- NLP feature extraction
- Sentiment analysis
- Comparación de modelos
- Explainable AI
- Multi-model inference
""",
    version="1.0.0"
)

# =========================================================
# PATH MODELOS
# =========================================================

MODELS_PATH = "models"

# =========================================================
# CARGAR MODELOS
# =========================================================

models = {
    "logistic_regression": joblib.load(
        os.path.join(MODELS_PATH, "logistic_regression.pkl")
    ),

    "random_forest": joblib.load(
        os.path.join(MODELS_PATH, "random_forest.pkl")
    ),

    "xgboost": joblib.load(
        os.path.join(MODELS_PATH, "xgboost.pkl")
    ),

    "lightgbm": joblib.load(
        os.path.join(MODELS_PATH, "lightgbm.pkl")
    ),

    "catboost": joblib.load(
        os.path.join(MODELS_PATH, "catboost.pkl")
    )
}

# =========================================================
# REQUEST SCHEMA
# =========================================================

class ReviewRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        example="""
This product exceeded my expectations.
Great quality and excellent packaging.
"""
    )

    score: int = Field(
        ...,
        ge=1,
        le=5,
        example=5,
        description="Amazon product rating from 1 to 5"
    )

    model_name: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost"
    ] = "random_forest"


# =========================================================
# RESPONSE SCHEMAS
# =========================================================

class HomeResponse(BaseModel):

    message: str
    version: str


class HealthResponse(BaseModel):

    status: str
    models_loaded: int


class ModelsResponse(BaseModel):

    available_models: List[str]
    default_model: str


class ModelMetric(BaseModel):

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class PredictionResponse(BaseModel):

    prediction: int

    probability_helpful: float

    features: Dict[str, Any]

    model_used: str

    interpretation: List[str]


class ExplainResponse(BaseModel):

    positive_factors: List[str]

    negative_factors: List[str]

    features: Dict[str, Any]


# =========================================================
# HOME
# =========================================================

@app.get(
    "/",
    response_model=HomeResponse,
    tags=["General"]
)
def home():

    return {
        "message": "Amazon Review Intelligence API funcionando",
        "version": "1.0.0"
    }


# =========================================================
# HEALTHCHECK
# =========================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"]
)
def health():

    return {
        "status": "ok",
        "models_loaded": len(models)
    }


# =========================================================
# MODELOS DISPONIBLES
# =========================================================

@app.get(
    "/models",
    response_model=ModelsResponse,
    tags=["Models"]
)
def get_models():

    return {
        "available_models": list(models.keys()),
        "default_model": "random_forest"
    }


# =========================================================
# METRICAS MODELOS
# =========================================================

@app.get(
    "/models/metrics",
    response_model=Dict[str, ModelMetric],
    tags=["Models"]
)
def get_metrics():

    try:

        with open(
            os.path.join(MODELS_PATH, "metrics.json"),
            "r"
        ) as f:

            metrics = json.load(f)

        return metrics

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading metrics: {str(e)}"
        )


# =========================================================
# PREDICCION HELPFULNESS
# =========================================================

@app.post(
    "/reviews/predict_helpfulness",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict_helpfulness(review: ReviewRequest):

    try:

        # ================================================
        # EXTRAER FEATURES NLP
        # ================================================

        features = extract_features(
            review.text,
            review.score
        )

        # ================================================
        # CREAR DATAFRAME
        # ================================================

        X = pd.DataFrame([features])

        # ================================================
        # SELECCIONAR MODELO
        # ================================================

        model = models[review.model_name]

        # ================================================
        # PREDECIR
        # ================================================

        probability = model.predict_proba(X)[0][1]

        prediction = int(probability >= 0.5)

        # ================================================
        # INTERPRETACION SIMPLE
        # ================================================

        interpretation = []

        sentiment = features.get(
            "sentiment_compound",
            0
        )

        word_count = features.get(
            "word_count",
            0
        )

        coherence = features.get(
            "coherence",
            0
        )

        # Sentimiento positivo
        if sentiment > 0.3 and review.score >= 4:

            interpretation.append(
                "El sentimiento coincide con la calificación positiva."
            )

        # Sentimiento negativo
        if sentiment < -0.3 and review.score <= 2:

            interpretation.append(
                "El sentimiento coincide con la calificación negativa."
            )

        # Texto positivo
        if sentiment > 0.5:

            interpretation.append(
                "El texto transmite sentimiento positivo."
            )

        # Texto negativo
        if sentiment < -0.5:

            interpretation.append(
                "El texto transmite sentimiento negativo."
            )

        # Buena longitud
        if word_count > 40:

            interpretation.append(
                "La reseña tiene buena longitud descriptiva."
            )

        # Muy corta
        if word_count < 10:

            interpretation.append(
                "La reseña podría ser demasiado corta."
            )

        # Coherencia
        if coherence == 1:

            interpretation.append(
                "La reseña presenta buena coherencia textual."
            )

        # ================================================
        # RESPUESTA
        # ================================================

        return {

            "prediction": prediction,

            "probability_helpful": round(
                float(probability),
                4
            ),

            "features": features,

            "model_used": review.model_name,

            "interpretation": interpretation
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# =========================================================
# EXPLAINABILITY
# =========================================================

@app.post(
    "/reviews/explain",
    response_model=ExplainResponse,
    tags=["Explainability"]
)
def explain_review(review: ReviewRequest):

    try:

        # ================================================
        # EXTRAER FEATURES
        # ================================================

        features = extract_features(
            review.text,
            review.score
        )

        positive_factors = []

        negative_factors = []

        # ================================================
        # FACTORES POSITIVOS
        # ================================================

        if features.get("word_count", 0) > 40:

            positive_factors.append(
                "Detailed review length"
            )

        if features.get("coherence", 0) == 1:

            positive_factors.append(
                "Consistent writing coherence"
            )

        if features.get("sentiment_compound", 0) > 0.5:

            positive_factors.append(
                "Strong positive sentiment"
            )

        if features.get("sentence_count", 0) >= 3:

            positive_factors.append(
                "Multiple descriptive sentences"
            )

        # ================================================
        # FACTORES NEGATIVOS
        # ================================================

        if features.get("uppercase_ratio", 0) > 0.2:

            negative_factors.append(
                "Too many uppercase letters"
            )

        if features.get("word_count", 0) < 8:

            negative_factors.append(
                "Very short review"
            )

        if features.get("sentiment_negative", 0) > 0.7:

            negative_factors.append(
                "Extremely negative sentiment"
            )

        # ================================================
        # RESPUESTA
        # ================================================

        return {

            "positive_factors": positive_factors,

            "negative_factors": negative_factors,

            "features": features
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Explainability error: {str(e)}"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000
    )
