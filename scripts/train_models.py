
import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils.preprocessing import clean_reviews
from utils.features import extract_features

print("Cargando dataset...")

df = pd.read_csv("data/raw/Reviews.csv")

print("Limpiando datos...")

df = clean_reviews(df)

print("Generando features NLP...")

feature_rows = []

for _, row in df.iterrows():

    feature_rows.append(
        extract_features(
            row['Text'],
            row['Score']
        )
    )

X = pd.DataFrame(feature_rows)

y = df['is_helpful']

print("Separando train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {

    "logistic_regression": LogisticRegression(
        max_iter=2000
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    ),

    "xgboost": XGBClassifier(
        eval_metric='logloss'
    ),

    "lightgbm": LGBMClassifier(),

    "catboost": CatBoostClassifier(
        verbose=0
    )
}

results = {}

for name, model in models.items():

    print(f"Entrenando {name}...")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, preds)
        ),
        "precision": float(
            precision_score(y_test, preds)
        ),
        "recall": float(
            recall_score(y_test, preds)
        ),
        "f1_score": float(
            f1_score(y_test, preds)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probs)
        )
    }

    results[name] = metrics

    joblib.dump(
        model,
        f"models/{name}.pkl"
        os.makedirs("models", exist_ok=True)
    )

with open("models/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("Entrenamiento finalizado.")
print(results)
