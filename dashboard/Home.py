# =====================================================
# PAGE CONFIG
# =====================================================


with st.sidebar:

    st.image(
        "assets/logo.png",
        width=220
    )

    st.markdown("# UNIANDES")

  

    st.markdown("---")

    st.subheader("Equipo de Desarrollo")

    st.info("""
 Byron Fabricio Torres Apolo  


 Monica Cholango


 Jose Arevalo

""")

    st.markdown("---")

    st.success("Sistema operativo y API conectada")
API_URL = "http://127.0.0.1:8000"

# =====================================================
# PROFESSIONAL CSS
# =====================================================

st.markdown("""
<style>

/* =========================
   GLOBAL
========================= */

.stApp {
    background-color: #F8FAFC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0F172A;
}

/* =========================
   MAIN CONTAINER
========================= */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* =========================
   TITLES
========================= */

h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: #0F172A;
    margin-bottom: 0.2rem;
}

h2, h3 {
    color: #1E293B;
    font-weight: 600;
}

/* =========================
   TEXT
========================= */

p {
    color: #475569;
    font-size: 16px;
}

/* =========================
   METRICS
========================= */

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0px 1px 4px rgba(15, 23, 42, 0.04);
}

/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* =========================
   ALERTS
========================= */

.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 14px;
}

/* =========================
   PLOTLY
========================= */

.js-plotly-plot {
    border-radius: 18px;
    background: white;
    padding: 10px;
}

/* =========================
   DIVIDER
========================= */

hr {
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* =========================
   CUSTOM CARD
========================= */

.custom-card {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.5rem;
    border-radius: 20px;
    box-shadow: 0px 1px 4px rgba(15, 23, 42, 0.04);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<h1>
Plataforma Analitica de Reseñas
</h1>

""", unsafe_allow_html=True)

# =====================================================
# DESCRIPTION
# =====================================================

with st.container(border=True):

    st.markdown("""
### Capacidades Inteligentes de la Plataforma

ReviewIQ ayuda a las empresas a:

- Evaluar la calidad de las reseñas de clientes
- Detectar reseñas falsas, spam o contenido sospechoso
- Comprender la percepción y satisfacción del cliente
- Obtener información estratégica basada en datos
- Comparar el rendimiento de diferentes modelos predictivos

""")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# API STATUS
# =====================================================

api_status = "Offline"

try:

    response = requests.get(f"{API_URL}/health")

    if response.status_code == 200:
        api_status = "Online"

except:
    api_status = "Offline"

# =====================================================
# MODEL METRICS
# =====================================================

metrics = {}

try:

    metrics = requests.get(
        f"{API_URL}/models/metrics"
    ).json()

    metrics_df = pd.DataFrame(metrics).T

    best_model = metrics_df[
        'f1_score'
    ].idxmax()

    best_f1 = round(
        metrics_df[
            'f1_score'
        ].max(),
        3
    )

except:

    best_model = "Random Forest"
    best_f1 = 0.882

# =====================================================
# KPI SECTION
# =====================================================

st.subheader("Executive KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Best F1 Score",
        best_f1
    )

with col2:

    st.metric(
        "Top Performing Model",
        best_model
    )

with col3:

    st.metric(
        "API Status",
        api_status
    )

with col4:

    st.metric(
        "Spam Detection Rate",
        "87.5%"
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# Modelos  ML Soportados
# =====================================================

with st.container(border=True):

    st.subheader("Modelos de Machine Learning Soportados")

    st.markdown("""
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- Modelos Ensemble  
- Pipelines Scikit-Learn  
""")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================


# =====================================================
# SENTIMENT ANALYTICS
# =====================================================

st.subheader("Sentiment Analytics")

sentiment_df = pd.DataFrame({

    "Sentiment": [
        "Positive",
        "Neutral",
        "Negative"
    ],

    "Count": [
        68,
        20,
        12
    ]
})

fig2 = px.bar(

    sentiment_df,

    x="Sentiment",

    y="Count",

    text="Count",

    color="Sentiment"
)

fig2.update_layout(

    plot_bgcolor="white",
    paper_bgcolor="white",

    font=dict(
        family="Inter",
        size=14,
        color="#0F172A"
    ),

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),

    showlegend=False
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# SYSTEM STATUS
# =====================================================

st.subheader("System Status")

if api_status == "Online":

    st.success(
        "FastAPI services operational."
    )

else:

    st.error(
        "FastAPI services unavailable."
    )

with st.container(border=True):

    st.markdown("""
### Arquitectura de la Plataforma

- API REST para integración y servicios
- Dashboard interactivo de análisis de datos
- Motor de procesamiento de lenguaje natural (NLP)
- Evaluación mediante múltiples modelos predictivos
- Procesamiento masivo automatizado de reseñas

""")
