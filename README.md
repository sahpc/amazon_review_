
# Amazon Review Intelligence — NLP + ML + FastAPI + Streamlit

Proyecto profesional de Machine Learning para predecir la utilidad de reseñas de Amazon.

## Modelos implementados

- Logistic Regression
- XGBoost
- LightGBM
- CatBoost

## Arquitectura

```text
amazon_review_intelligence_v2/
│
├── api/
├── dashboard/
├── data/
├── models/
├── notebooks/
├── scripts/
└── utils/
```

## Flujo del proyecto

1. Cargar dataset
2. Limpiar datos
3. Crear features NLP
4. Entrenar modelos
5. Guardar modelos
6. Consumir desde FastAPI
7. Visualizar en Streamlit

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar entrenamiento

```bash
python scripts/train_models.py
```

## Ejecutar API

```bash
uvicorn api.main:app --reload
```

## Ejecutar dashboard

```bash
streamlit run dashboard/Home.py
```
