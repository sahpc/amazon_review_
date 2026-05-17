# =========================================================
# REVIEWIQ AI ANALYZER
# ENTERPRISE NLP PLATFORM
# =========================================================

import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Plataforma",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0F172A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: #0F172A;
}

h2, h3 {
    color: #1E293B;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #E2E8F0;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0px 1px 3px rgba(15,23,42,0.04);
}

.insight-box {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 10px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

textarea {
    border-radius: 14px !important;
    border: 1px solid #CBD5E1 !important;
}

.stAlert {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# VALIDATION
# =========================================================

def validate_review(text):

    text = text.strip()

    # =====================================================
    # EMPTY
    # =====================================================

    if not text:
        return False, "La reseña no puede estar vacía."

    # =====================================================
    # WORDS
    # =====================================================

    words = text.split()

    if len(words) < 8:
        return False, "La reseña debe contener mínimo 8 palabras."

    # =====================================================
    # ONLY NUMBERS
    # =====================================================

    if text.isdigit():
        return False, "La reseña no puede contener solo números."

    # =====================================================
    # SPECIAL CHARACTERS
    # =====================================================

    special_ratio = len(
        re.findall(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', text)
    ) / max(len(text), 1)

    if special_ratio > 0.15:
        return False, "Demasiados caracteres especiales detectados."

    # =====================================================
    # REPETITIVE WORDS
    # =====================================================

    lower_words = [
        w.lower()
        for w in words
    ]

    unique_ratio = len(
        set(lower_words)
    ) / max(len(lower_words), 1)

    if unique_ratio < 0.70:
        return False, "Texto repetitivo detectado."

    # =====================================================
    # RANDOM SYMBOLS
    # =====================================================

    if re.search(r'[-_=]{3,}', text):
        return False, "Patrones sospechosos detectados."

    # =====================================================
    # UPPERCASE
    # =====================================================

    uppercase_ratio = sum(
        1 for c in text if c.isupper()
    ) / max(len(text), 1)

    if uppercase_ratio > 0.30:
        return False, "Uso excesivo de mayúsculas."

    # =====================================================
    # SPAM PHRASES
    # =====================================================

    suspicious_patterns = [

        "good good",
        "nice nice",
        "test test",
        "muy bueno good good",
        "aaaa",
        "xxxxx",
        "fake review",
        "asdf",
        "lorem ipsum"

    ]

    text_lower = text.lower()

    for pattern in suspicious_patterns:

        if pattern in text_lower:

            return False, (
                "Posible spam o contenido artificial detectado."
            )

    # =====================================================
    # REPEATED CHARACTERS
    # =====================================================

    if re.search(r'(.)\1{4,}', text.lower()):
        return False, "Patrón repetitivo detectado."

    return True, ""

# =========================================================
# FUNCTIONS
# =========================================================

def sentiment_label(score):

    if score >= 0.6:
        return "Muy Positivo"

    elif score >= 0.2:
        return "Positivo"

    elif score <= -0.6:
        return "Muy Negativo"

    elif score <= -0.2:
        return "Negativo"

    return "Neutral"

def review_quality(probability):

    if probability >= 85:
        return "Excelente"

    elif probability >= 70:
        return "Buena"

    elif probability >= 50:
        return "Regular"

    return "Baja"

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<h1>
ReviewIQ AI Analyzer
</h1>

<p style='font-size:18px;color:#475569;'>
Plataforma Inteligente de Analítica de Reseñas con IA
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.subheader("Configuración IA")

    score = st.slider(
        "Calificación del Producto",
        1,
        5,
        5
    )

    model_name = st.selectbox(
        "Modelo IA",
        [
            "xgboost",
            "lightgbm",
            "catboost",
            "logistic_regression"
        ]
    )

    st.divider()

    st.markdown("""
### Capacidades

- NLP Analytics
- Explainable AI
- Spam Detection
- Machine Learning
- Business Intelligence
""")

# =========================================================
# INPUT
# =========================================================

st.subheader("Análisis de Reseñas")

review_text = st.text_area(
    "Ingrese la reseña del cliente",
    height=220
)

# =========================================================
# BUTTON
# =========================================================

if st.button(
    "Analizar Reseña",
    use_container_width=True
):

    is_valid, message = validate_review(review_text)

    if not is_valid:

        st.error(message)

    else:

        payload = {
            "text": review_text,
            "score": score,
            "model_name": model_name
        }

        try:

            with st.spinner("Procesando análisis IA..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload
                )

                # =====================================================
                # API VALIDATION
                # =====================================================

                if response.status_code != 200:

                    st.error(
                        f"FastAPI Error {response.status_code}: {response.text}"
                    )

                    st.stop()

                data = response.json()

            # =====================================================
            # DATA
            # =====================================================

            probability = (
                data['probability'] * 100
            )

            features = data['features']

            # =====================================================
            # COHERENCE VALIDATION
            # =====================================================

            if features.get("coherence", 1) == 0:

                st.warning(
                    "La IA detectó posible incoherencia textual."
                )

            sentiment = features.get(
                'sentiment_compound',
                0
            )

            word_count = features.get(
                'word_count',
                0
            )

            quality = review_quality(probability)

            sentiment_text = sentiment_label(sentiment)

            # =====================================================
            # AI TRUST SCORE
            # =====================================================

            ai_score = round(
                (
                    probability * 0.6
                    +
                    (word_count * 1.5)
                    +
                    ((sentiment + 1) * 20)
                ) / 2,
                2
            )

            ai_score = min(ai_score, 100)

            # =====================================================
            # SAVE HISTORY
            # =====================================================

            st.session_state.history.append({

                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "review": review_text,

                "probability": round(probability, 2),

                "model": model_name,

                "sentiment": sentiment_text,

                "quality": quality,

                "ai_score": ai_score
            })

            # =====================================================
            # METRICS
            # =====================================================

            st.subheader("Resumen Ejecutivo")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:

                st.metric(
                    "Probabilidad Útil",
                    f"{probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Cantidad de Palabras",
                    word_count
                )

            with col3:

                st.metric(
                    "Sentimiento",
                    sentiment_text
                )

            with col4:

                st.metric(
                    "Calidad",
                    quality
                )

            with col5:

                st.metric(
                    "AI Trust Score",
                    f"{ai_score}%"
                )

            # =====================================================
            # PROGRESS
            # =====================================================

            st.subheader("Score de Calidad")

            progress_value = max(
                0,
                min(int(probability), 100)
            )

            st.progress(progress_value)

            st.caption(
                f"Nivel de utilidad detectado: {progress_value}%"
            )

            # =====================================================
            # FEATURES
            # =====================================================

            st.subheader("Características NLP")

            features_df = pd.DataFrame(
                features.items(),
                columns=["Feature", "Value"]
            )

            st.dataframe(
                features_df,
                use_container_width=True
            )

            # =====================================================
            # INSIGHTS
            # =====================================================

            st.subheader("Explainable AI")

            insights = []

            if sentiment > 0.7 and word_count > 15:

                insights.append(
                    "Sentimiento altamente positivo detectado."
                )

            if word_count > 80:

                insights.append(
                    "La reseña contiene información detallada."
                )

            if features.get('uppercase_ratio', 0) > 0.10:

                insights.append(
                    "Posible patrón spam detectado."
                )

            if probability < 60:

                insights.append(
                    "La reseña presenta baja utilidad informativa."
                )

            if features.get("coherence", 1) == 1:

                insights.append(
                    "La IA detectó coherencia textual adecuada."
                )

            if not insights:

                insights.append(
                    "No se detectaron anomalías importantes."
                )

            for item in insights:

                st.markdown(
                    f"""
<div class="insight-box">
{item}
</div>
""",
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(f"Error general: {e}")

# =========================================================
# HISTORY
# =========================================================

st.divider()

st.subheader("Historial de Análisis")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df.sort_values(
            by="probability",
            ascending=False
        ),
        use_container_width=True
    )

# =========================================================
# SYSTEM STATUS
# =========================================================

st.divider()

st.subheader("Estado del Sistema")

try:

    response = requests.get(
        f"{API_URL}/health"
    )

    if response.status_code == 200:

        health = response.json()

        col1, col2 = st.columns(2)

        with col1:

            st.success("API FastAPI Online")

        with col2:

            st.info(
                f"Modelos cargados: {health['models_loaded']}"
            )

    else:

        st.error("API no disponible.")

except:

    st.error(
        "No se pudo conectar con FastAPI."
    )
