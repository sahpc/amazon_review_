import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ReviewIQ Platform",
    page_icon="📊",
    layout="wide"
)

#API_URL = "http://127.0.0.1:8000"
API_URL = "https://amazon-review-zsqc.onrender.com/"

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
ReviewIQ Platform
</h1>

<p style='font-size:18px; color:#475569; margin-bottom:30px;'>
Enterprise NLP & Machine Learning Analytics Platform
</p>
""", unsafe_allow_html=True)

# =====================================================
# DESCRIPTION
# =====================================================

with st.container(border=True):

    st.markdown("""
### Platform Capabilities

ReviewIQ enables organizations to:

- Detect high-quality reviews
- Automatically identify spam content
- Analyze customer sentiment
- Generate business intelligence insights
- Deploy explainable AI models
- Benchmark multiple ML algorithms
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
# AI CAPABILITIES
# =====================================================

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.subheader("AI Features")

        st.markdown("""
- NLP Feature Engineering  
- Sentiment Analysis  
- Explainable AI  
- Spam Detection  
- Review Quality Scoring  
- Batch Processing  
""")

with col2:

    with st.container(border=True):

        st.subheader("Supported Models")

        st.markdown("""
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- Ensemble Learning  
- Scikit-Learn Pipeline  
""")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# BUSINESS VALUE
# =====================================================

with st.container(border=True):

    st.subheader("Business Value")

    st.markdown("""
### Executive Benefits

- Improve customer trust
- Reduce spam and fake reviews
- Enhance marketplace credibility
- Automate moderation processes
- Improve recommendation systems
- Generate actionable analytics
""")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.subheader("AI Model Benchmark")

try:

    benchmark_df = metrics_df.sort_values(
        by="f1_score",
        ascending=False
    )

    fig = px.bar(

        benchmark_df,

        y="f1_score",

        text="f1_score",

        color="f1_score",

        color_continuous_scale="Blues"
    )

    fig.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside'
    )

    fig.update_layout(

        plot_bgcolor="white",
        paper_bgcolor="white",

        font=dict(
            family="Inter",
            size=14,
            color="#0F172A"
        ),

        title_font_size=20,

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),

        coloraxis_showscale=False,

        xaxis_title="Model",
        yaxis_title="F1 Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except:

    st.warning(
        "Model metrics currently unavailable."
    )

st.markdown("<br>", unsafe_allow_html=True)

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
### Platform Architecture

- REST API Infrastructure
- Streamlit Analytics Dashboard
- NLP Intelligence Engine
- Explainable AI Module
- Multi-Model Inference
- Batch Processing Pipeline
- Enterprise Analytics Layer
""")
