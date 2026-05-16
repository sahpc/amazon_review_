# =========================================================
# REVIEWIQ AI ANALYZER
# ENTERPRISE NLP PLATFORM
# =========================================================

import streamlit as st
import requests
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ReviewIQ Analyzer",
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
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
===================================================== */

.stApp {
    background-color: #F8FAFC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0F172A;
}

/* =====================================================
   MAIN CONTAINER
===================================================== */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* =====================================================
   TITLES
===================================================== */

h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: #0F172A;
}

h2, h3 {
    color: #1E293B;
    font-weight: 600;
}

/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #E2E8F0;
}

/* =====================================================
   METRICS
===================================================== */

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0px 1px 3px rgba(15,23,42,0.04);
}

/* =====================================================
   CONTAINERS
===================================================== */

.custom-card {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.5rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}

/* =====================================================
   REVIEW CARDS
===================================================== */

.review-excellent {
    background: #ECFDF5;
    border: 1px solid #BBF7D0;
    border-left: 6px solid #22C55E;
    padding: 1.2rem;
    border-radius: 16px;
    margin-bottom: 15px;
}

.review-good {
    background: #FEFCE8;
    border: 1px solid #FDE68A;
    border-left: 6px solid #F59E0B;
    padding: 1.2rem;
    border-radius: 16px;
    margin-bottom: 15px;
}

.review-bad {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 6px solid #EF4444;
    padding: 1.2rem;
    border-radius: 16px;
    margin-bottom: 15px;
}

/* =====================================================
   INSIGHTS
===================================================== */

.insight-box {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 10px;
}

/* =====================================================
   BUTTON
===================================================== */

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

/* =====================================================
   TEXT AREA
===================================================== */

textarea {
    border-radius: 14px !important;
    border: 1px solid #CBD5E1 !important;
}

/* =====================================================
   PLOTLY
===================================================== */

.js-plotly-plot {
    background: white;
    border-radius: 18px;
    padding: 10px;
}

/* =====================================================
   ALERTS
===================================================== */

.stAlert {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# NLP VALIDATION
# =========================================================

def validate_review(text):

    text = text.strip()

    if not text:
        return False, "Review cannot be empty."

    if len(text.split()) < 5:
        return False, "Review is too short."

    if text.isdigit():
        return False, "Review cannot contain only numbers."

    special_ratio = len(
        re.findall(r'[^a-zA-Z0-9\\s]', text)
    ) / max(len(text), 1)

    if special_ratio > 0.40:
        return False, "Too many special characters detected."

    words = text.lower().split()

    unique_ratio = len(set(words)) / max(len(words), 1)

    if unique_ratio < 0.30:
        return False, "Potential spam or repetitive content."

    uppercase_ratio = sum(
        1 for c in text if c.isupper()
    ) / max(len(text), 1)

    if uppercase_ratio > 0.40:
        return False, "Excessive uppercase usage."

    return True, ""

# =========================================================
# SENTIMENT LABEL
# =========================================================

def sentiment_label(score):

    if score >= 0.6:
        return "Very Positive"

    elif score >= 0.2:
        return "Positive"

    elif score <= -0.6:
        return "Very Negative"

    elif score <= -0.2:
        return "Negative"

    return "Neutral"

# =========================================================
# REVIEW QUALITY
# =========================================================

def review_quality(probability):

    if probability >= 85:
        return "Excellent"

    elif probability >= 70:
        return "Good"

    elif probability >= 50:
        return "Average"

    return "Low"

# =========================================================
# NLP SCORE
# =========================================================

def calculate_nlp_score(features, sentiment, text):

    score = 50

    word_count = features['word_count']

    if word_count > 50:
        score += 20

    elif word_count > 25:
        score += 10

    else:
        score -= 10

    if features['coherence'] == 1:
        score += 15

    if abs(sentiment) > 0.5:
        score += 10

    if features['uppercase_ratio'] > 0.10:
        score -= 10

    patterns = [
        "i used",
        "my experience",
        "after using",
        "for two weeks",
        "tested",
        "i have been"
    ]

    text_lower = text.lower()

    if any(p in text_lower for p in patterns):
        score += 15

    return max(0, min(100, score))

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<h1>
ReviewIQ AI Analyzer
</h1>

<p style='font-size:18px;color:#475569;'>
Enterprise NLP & Explainable AI Review Analysis Platform
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

metrics = {}

try:

    metrics = requests.get(
        f"{API_URL}/models/metrics"
    ).json()

    metrics_df = pd.DataFrame(metrics).T

    best_model = metrics_df['f1_score'].idxmax()

except:

    best_model = "random_forest"

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.subheader("AI Configuration")

    score = st.slider(
        "Product Rating",
        1,
        5,
        5
    )

    model_name = st.selectbox(
        "AI Model",
        [
            "random_forest",
            "xgboost",
            "lightgbm",
            "logistic_regression"
        ]
    )

    st.divider()

    st.success(
        f"Top Model: {best_model}"
    )

    st.markdown("""
### Platform Modules

- NLP Analytics
- Explainable AI
- Spam Detection
- Multi-Model Inference
- Business Intelligence
""")

# =========================================================
# REVIEW INPUT
# =========================================================

st.subheader("Review Analysis")

review_text = st.text_area(
    "Enter customer review",
    height=220,
    placeholder="""
This product exceeded my expectations.
Excellent packaging and premium quality.
I have used it for two weeks and performance has been very stable.
"""
)

button_disabled = len(
    review_text.strip()
) == 0

# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "Analyze Review",
    use_container_width=True,
    disabled=button_disabled
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

            with st.spinner(
                "Running AI inference..."
            ):

                response = requests.post(
                    f"{API_URL}/reviews/predict_helpfulness",
                    json=payload
                )

                data = response.json()

            probability = (
                data['probability_helpful'] * 100
            )

            features = data['features']

            sentiment = (
                features['sentiment_compound']
            )

            word_count = features['word_count']

            nlp_score = calculate_nlp_score(
                features,
                sentiment,
                review_text
            )

            quality = review_quality(probability)

            sentiment_text = sentiment_label(sentiment)

            # =====================================================
            # HISTORY
            # =====================================================

            st.session_state.history.append({

                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "review": review_text,

                "probability": probability,

                "model": model_name,

                "sentiment": sentiment,

                "quality": quality,

                "nlp_score": nlp_score
            })

            st.markdown("<br>", unsafe_allow_html=True)

            # =====================================================
            # KPI METRICS
            # =====================================================

            st.subheader("Executive Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Helpfulness",
                    f"{probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Word Count",
                    word_count
                )

            with col3:
                st.metric(
                    "Sentiment",
                    sentiment_text
                )

            with col4:
                st.metric(
                    "NLP Score",
                    f"{nlp_score}/100"
                )

            # =====================================================
            # PROGRESS
            # =====================================================

            st.subheader("Review Quality Score")

            st.progress(
                min(int(probability), 100)
            )

            # =====================================================
            # FEATURES
            # =====================================================

            st.subheader("NLP Features")

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

            if sentiment >= 0.3 and score >= 4:
                insights.append(
                    "Positive sentiment aligns with product rating."
                )

            if word_count > 50:
                insights.append(
                    "Review contains detailed product experience."
                )

            if features['uppercase_ratio'] > 0.10:
                insights.append(
                    "Potential spam pattern detected from uppercase usage."
                )

            if sentiment > 0.5:
                insights.append(
                    "Highly positive emotional language detected."
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

            st.error(
                f"API Error: {e}"
            )

# =========================================================
# HISTORY
# =========================================================

st.divider()

st.subheader("Review Intelligence History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    excellent_reviews = history_df[
        history_df["probability"] >= 85
    ]

    good_reviews = history_df[
        (history_df["probability"] >= 70)
        &
        (history_df["probability"] < 85)
    ]

    low_reviews = history_df[
        history_df["probability"] < 70
    ]

    tab1, tab2, tab3 = st.tabs([
        "Excellent",
        "Good",
        "Low Quality"
    ])

    # =====================================================
    # EXCELLENT
    # =====================================================

    with tab1:

        if not excellent_reviews.empty:

            for _, item in excellent_reviews.iterrows():

                st.markdown(
                    f"""
<div class="review-excellent">

<b>Date:</b> {item['time']}<br>
<b>Model:</b> {item['model']}<br>
<b>Helpfulness:</b> {item['probability']:.2f}%<br>
<b>NLP Score:</b> {item['nlp_score']}/100<br><br>

{item['review'][:400]}

</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No excellent reviews detected."
            )

    # =====================================================
    # GOOD
    # =====================================================

    with tab2:

        if not good_reviews.empty:

            for _, item in good_reviews.iterrows():

                st.markdown(
                    f"""
<div class="review-good">

<b>Date:</b> {item['time']}<br>
<b>Model:</b> {item['model']}<br>
<b>Helpfulness:</b> {item['probability']:.2f}%<br>
<b>NLP Score:</b> {item['nlp_score']}/100<br><br>

{item['review'][:400]}

</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No good reviews detected."
            )

    # =====================================================
    # LOW QUALITY
    # =====================================================

    with tab3:

        if not low_reviews.empty:

            for _, item in low_reviews.iterrows():

                st.markdown(
                    f"""
<div class="review-bad">

<b>Date:</b> {item['time']}<br>
<b>Model:</b> {item['model']}<br>
<b>Helpfulness:</b> {item['probability']:.2f}%<br>
<b>NLP Score:</b> {item['nlp_score']}/100<br><br>

{item['review'][:400]}

</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No low-quality reviews detected."
            )

# =========================================================
# MODEL ANALYTICS
# =========================================================

st.divider()

st.subheader("Model Performance Analytics")

try:

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

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

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),

        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except:

    st.warning(
        "Start FastAPI service to load analytics."
    )