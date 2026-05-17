# =====================================================
# BATCH REVIEW ANALYZER
# ENTERPRISE AI PLATFORM
# =====================================================

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Batch Review Analyzer",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

logo_path = BASE_DIR / "assets" / "logo.png"

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    try:

       st.image(
        "assets/logo.png",
        width=220
    )


    except:
        pass

    st.markdown("# UNIANDES")

    st.markdown("---")

    st.subheader("Proyecto")

    st.markdown("""
Plataforma Inteligente de Analítica de Reseñas basada en:

- NLP
- Machine Learning
- Detección de Spam
- Analítica Empresarial
""")

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://127.0.0.1:8000"

# =====================================================
# CSS
# =====================================================

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

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0px 1px 3px rgba(15,23,42,0.05);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}

.stAlert {
    border-radius: 14px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #0F1B33 100%);
    border-right: 1px solid #1E293B;
}

/* Texto del sidebar */
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Títulos del sidebar */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Divider */
section[data-testid="stSidebar"] hr {
    border-color: #1E3A8A;
}

.js-plotly-plot {
    background: white;
    border-radius: 18px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<h1>
Batch Review Analyzer
</h1>

<p style='font-size:18px;color:#475569;'>
Procesamiento Inteligente Masivo de Reseñas con IA
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# OVERVIEW
# =====================================================

with st.container(border=True):

    st.markdown("""
### Capacidades Inteligentes de la Plataforma

- Análisis NLP de reseñas
- Evaluación automática de calidad
- Detección de spam y fraude
- Análisis de sentimiento
- Machine Learning predictivo
- Procesamiento masivo automatizado
""")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "Subir archivo CSV de reseñas",
    type=["csv"]
)

# =====================================================
# MAIN PROCESS
# =====================================================

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(f"Error leyendo CSV: {e}")

        st.stop()

    st.subheader("Vista Previa del Dataset")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # =====================================================
    # AUTO DETECT COLUMNS
    # =====================================================

    possible_review_cols = [
        "review",
        "review_text",
        "text",
        "comment"
    ]

    possible_score_cols = [
        "score",
        "rating",
        "stars"
    ]

    review_col = None
    score_col = None

    for col in df.columns:

        if col.lower() in possible_review_cols:
            review_col = col

        if col.lower() in possible_score_cols:
            score_col = col

    # =====================================================
    # VALIDATION
    # =====================================================

    if review_col is None:

        st.error(
            "No se detectó columna de reseñas."
        )

        st.stop()

    if score_col is None:

        st.error(
            "No se detectó columna de puntuación."
        )

        st.stop()

    st.success(f"""
Columna de Reseñas: {review_col}

Columna de Rating: {score_col}
""")

    # =====================================================
    # MODEL SELECT
    # =====================================================

    model_name = st.selectbox(

        "Seleccionar Modelo IA",

        [
            
            "xgboost",
            "lightgbm",
            "catboost",
            "logistic_regression"
        ]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # BUTTON
    # =====================================================

    if st.button(
        "Analizar Reseñas",
        use_container_width=True
    ):

        results = []
        errors = []

        progress = st.progress(0)

        total = len(df)

        # =====================================================
        # LOOP
        # =====================================================

        for idx, row in df.iterrows():

            try:

                review_text = str(
                    row[review_col]
                )

                score = int(
                    row[score_col]
                )

                payload = {

                    "text": review_text,

                    "score": score,

                    "model_name": model_name
                }

                # =================================================
                # API REQUEST
                # =================================================

                response = requests.post(

                    f"{API_URL}/predict",

                    json=payload,

                    timeout=30
                )

                # =================================================
                # STATUS VALIDATION
                # =================================================

                if response.status_code != 200:

                    raise Exception(
                        f"""
API Error {response.status_code}

{response.text}
"""
                    )

                # =================================================
                # JSON VALIDATION
                # =================================================

                try:

                    data = response.json()

                except Exception:

                    raise Exception(
                        "Invalid JSON response from API"
                    )

                # =================================================
                # PROBABILITY
                # =================================================

                probability = (
                    data.get(
                        'probability_helpful',
                        0
                    ) * 100
                )

                # =================================================
                # FEATURES
                # =================================================

                features = data.get(
                    'features',
                    {}
                )

                # =================================================
                # QUALITY SCORE
                # =================================================

                if probability >= 85:

                    quality = "Excelente"

                elif probability >= 70:

                    quality = "Buena"

                elif probability >= 50:

                    quality = "Regular"

                else:

                    quality = "Baja"

                # =================================================
                # SPAM DETECTION
                # =================================================

                spam_flag = False

                if (
                    features.get(
                        'uppercase_ratio',
                        0
                    ) > 0.10
                ):

                    spam_flag = True

                if (
                    features.get(
                        'word_count',
                        0
                    ) < 5
                ):

                    spam_flag = True

                # =================================================
                # NLP SCORE
                # =================================================

                nlp_score = 50

                if (
                    features.get(
                        'word_count',
                        0
                    ) > 50
                ):

                    nlp_score += 20

                if (
                    abs(
                        features.get(
                            'sentiment_compound',
                            0
                        )
                    ) > 0.5
                ):

                    nlp_score += 15

                if (
                    features.get(
                        'coherence',
                        0
                    ) == 1
                ):

                    nlp_score += 15

                # =================================================
                # SAVE RESULTS
                # =================================================

                results.append({

                    "review":
                        review_text,

                    "score":
                        score,

                    "helpfulness":
                        round(probability, 2),

                    "quality":
                        quality,

                    "sentiment":
                        round(
                            features.get(
                                'sentiment_compound',
                                0
                            ),
                            3
                        ),

                    "word_count":
                        features.get(
                            'word_count',
                            0
                        ),

                    "spam":
                        spam_flag,

                    "nlp_score":
                        nlp_score
                })

            except Exception as e:

                errors.append({

                    "row": idx,

                    "error": str(e)
                })

            progress.progress(
                (idx + 1) / total
            )

        # =====================================================
        # RESULTS DF
        # =====================================================

        results_df = pd.DataFrame(results)

        # =====================================================
        # EMPTY VALIDATION
        # =====================================================

        if results_df.empty:

            st.error(
                "No se pudieron procesar reseñas válidas."
            )

            if errors:

                st.subheader(
                    "Errores de Procesamiento"
                )

                error_df = pd.DataFrame(errors)

                st.dataframe(
                    error_df,
                    use_container_width=True
                )

                st.write(error_df)

            st.stop()

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # EXECUTIVE KPIs
        # =====================================================

        st.subheader("Executive KPIs")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Reviews",
                len(results_df)
            )

        with col2:

            avg_helpfulness = round(
                results_df[
                    'helpfulness'
                ].mean(),
                2
            )

            st.metric(
                "Avg Helpfulness",
                f"{avg_helpfulness}%"
            )

        with col3:

            spam_pct = round(

                (
                    len(
                        results_df[
                            results_df[
                                'spam'
                            ] == True
                        ]
                    )
                    /
                    len(results_df)
                ) * 100,
                2
            )

            st.metric(
                "Spam %",
                f"{spam_pct}%"
            )

        with col4:

            avg_nlp = round(
                results_df[
                    'nlp_score'
                ].mean(),
                2
            )

            st.metric(
                "Avg NLP Score",
                avg_nlp
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # RESULTS TABLE
        # =====================================================

        st.subheader("Resultados del Análisis")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # HISTOGRAM
        # =====================================================

        st.subheader(
            "Distribución de Helpfulness"
        )

        fig = px.histogram(

            results_df,

            x='helpfulness',

            nbins=20,

            color_discrete_sequence=["#2563EB"]
        )

        fig.update_layout(

            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================================
        # DOWNLOAD
        # =====================================================

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(

            "Descargar CSV Analizado",

            csv,

            file_name=f"""
batch_analysis_{
datetime.now().strftime('%Y%m%d_%H%M%S')
}.csv
""",

            mime="text/csv"
        )
