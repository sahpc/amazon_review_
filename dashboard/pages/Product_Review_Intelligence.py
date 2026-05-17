# =====================================================
# PRODUCT REVIEW INTELLIGENCE
# ENTERPRISE AI PLATFORM
# =====================================================

import streamlit as st
import requests
import pandas as pd
import re

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Product Intelligence",
    page_icon="📊",
    layout="wide"
)

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

/* =========================================================
   MODIFICACIÓN: DROPDOWN CON FONDO ROJO
   ========================================================= */
div[data-baseweb="popover"] ul {
    background-color: #FF0000 !important;
}

div[data-baseweb="popover"] li {
    background-color: #FF0000 !important;
    color: #FFFFFF !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #CC0000 !important;
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# FUNCTIONS
# =====================================================

def review_quality(probability):

    if probability >= 85:
        return "Excellent"

    elif probability >= 70:
        return "Good"

    elif probability >= 50:
        return "Average"

    return "Low"

def sentiment_label(sentiment):

    if sentiment >= 0.6:
        return "Very Positive"

    elif sentiment >= 0.2:
        return "Positive"

    elif sentiment <= -0.6:
        return "Very Negative"

    elif sentiment <= -0.2:
        return "Negative"

    return "Neutral"

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<h1>
Product Intelligence Platform
</h1>

<p style='font-size:18px;color:#475569;'>
Enterprise AI System for Product & Customer Review Analytics
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# OVERVIEW
# =====================================================

with st.container(border=True):

    st.markdown("""
### Platform Capabilities

This platform enables organizations to:

- Analyze review quality
- Detect spam patterns
- Measure customer sentiment
- Evaluate product-review consistency
- Extract product intelligence insights
- Improve marketplace trust
- Generate explainable AI analytics
""")

# =====================================================
# INPUTS
# =====================================================

st.subheader("Product Information")

col1, col2 = st.columns(2)

with col1:

    product_name = st.text_input(
        "Product Name"
    )

    category = st.selectbox(
        "Product Category",
        [
            "Electronics",
            "Food",
            "Books",
            "Beauty",
            "Home",
            "Sports",
            "Clothing",
            "Health"
        ]
    )

with col2:

    brand = st.text_input(
        "Brand"
    )

    score = st.slider(
        "Product Rating",
        1,
        5,
        5
    )

# =====================================================
# DESCRIPTION
# =====================================================

product_description = st.text_area(
    "Product Description",
    height=150
)

# =====================================================
# REVIEW
# =====================================================

review_text = st.text_area(
    "Customer Review",
    height=220
)

# =====================================================
# MODEL
# =====================================================

model_name = st.selectbox(
    "AI Model",
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
    "Analyze Product Review",
    use_container_width=True
):

    # =================================================
    # VALIDATION
    # =================================================

    if not product_name.strip():

        st.error("Product name is required.")
        st.stop()

    if not review_text.strip():

        st.error("Customer review is required.")
        st.stop()

    try:

        # =================================================
        # PAYLOAD
        # =================================================

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
            json=payload
        )

        # =================================================
        # VALIDATE RESPONSE
        # =================================================

        if response.status_code != 200:

            st.error(
                f"API Error {response.status_code}: {response.text}"
            )

            st.stop()

        data = response.json()

        # =================================================
        # FIXED
        # =================================================

        probability = (
            data['probability'] * 100
        )

        features = data['features']

        sentiment = features.get(
            'sentiment_compound',
            0
        )

        word_count = features.get(
            'word_count',
            0
        )

        uppercase_ratio = features.get(
            'uppercase_ratio',
            0
        )

        coherence = features.get(
            'coherence',
            1
        )

        # =================================================
        # NLP ADJUSTMENT
        # =================================================

        if word_count < 20:
            probability *= 0.75

        elif word_count < 40:
            probability *= 0.90

        probability = min(probability, 100)

        # =================================================
        # CONSISTENCY
        # =================================================

        desc_words = set(
            re.findall(
                r'\w+',
                product_description.lower()
            )
        )

        review_words = set(
            re.findall(
                r'\w+',
                review_text.lower()
            )
        )

        overlap = len(
            desc_words.intersection(review_words)
        )

        if overlap >= 10:
            consistency_score = 95

        elif overlap >= 6:
            consistency_score = 80

        elif overlap >= 3:
            consistency_score = 65

        else:
            consistency_score = 40

        # =================================================
        # SPAM RISK
        # =================================================

        spam_risk = "Low"

        if uppercase_ratio > 0.10:
            spam_risk = "Medium"

        if coherence == 0:
            spam_risk = "High"

        # =================================================
        # QUALITY
        # =================================================

        quality = review_quality(probability)

        # =================================================
        # SENTIMENT
        # =================================================

        sentiment_text = sentiment_label(sentiment)

        # =================================================
        # REVIEW CLASSIFICATION
        # =================================================

        review_rank = "Average Review"

        if (
            probability >= 90
            and consistency_score >= 80
            and coherence == 1
        ):

            review_rank = "Excellent Review"

        elif probability >= 75:

            review_rank = "Good Review"

        elif probability < 50:

            review_rank = "Low Quality Review"

        if spam_risk == "High":

            review_rank = "Potential Spam Review"

        # =================================================
        # MARKETPLACE PRIORITY
        # =================================================

        marketplace_priority = "Low"

        if review_rank == "Excellent Review":
            marketplace_priority = "High"

        elif review_rank == "Good Review":
            marketplace_priority = "Medium"

        # =================================================
        # METRICS
        # =================================================

        st.subheader("Executive AI Summary")

        colA, colB, colC, colD = st.columns(4)

        with colA:

            st.metric(
                "Helpfulness",
                f"{probability:.2f}%"
            )

        with colB:

            st.metric(
                "Sentiment",
                sentiment_text
            )

        with colC:

            st.metric(
                "Spam Risk",
                spam_risk
            )

        with colD:

            st.metric(
                "Review Quality",
                quality
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # CLASSIFICATION
        # =================================================

        st.subheader("AI Review Classification")

        if "Excellent" in review_rank:

            st.success(review_rank)

        elif "Good" in review_rank:

            st.info(review_rank)

        elif "Spam" in review_rank:

            st.error(review_rank)

        else:

            st.warning(review_rank)

        # =================================================
        # PRODUCT INTELLIGENCE
        # =================================================

        st.subheader("Product Intelligence")

        col1, col2 = st.columns(2)

        with col1:

            st.info(f"""
Product: {product_name}

Brand: {brand}

Category: {category}
""")

        with col2:

            st.info(f"""
Consistency Score: {consistency_score}%

Word Count: {word_count}

Marketplace Priority: {marketplace_priority}
""")

        # =================================================
        # FEATURES
        # =================================================

        st.subheader("NLP Features")

        features_df = pd.DataFrame(
            features.items(),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            features_df,
            use_container_width=True
        )

        # =================================================
        # EXPLAINABLE AI
        # =================================================

        st.subheader("Explainable AI")

        insights = []

        if consistency_score >= 80:

            insights.append(
                "High product-review consistency detected."
            )

        else:

            insights.append(
                "Review may lack product-specific details."
            )

        if probability >= 85:

            insights.append(
                "Review contains high informational value."
            )

        if sentiment > 0.5:

            insights.append(
                "Strong positive customer perception detected."
            )

        if sentiment < -0.5:

            insights.append(
                "Strong negative customer perception detected."
            )

        if spam_risk == "High":

            insights.append(
                "Potential spam indicators detected."
            )

        if word_count < 20:

            insights.append(
                "Review length is limited."
            )

        if coherence == 1:

            insights.append(
                "Text coherence successfully detected."
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
            f"FastAPI Error: {e}"
        )
