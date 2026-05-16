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

# =====================================================
# PROFESSIONAL CSS
# =====================================================

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
   CONTAINER
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
   INPUTS
===================================================== */

textarea, input {
    border-radius: 14px !important;
    border: 1px solid #CBD5E1 !important;
}

/* =====================================================
   BUTTON
===================================================== */

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

/* =====================================================
   METRICS
===================================================== */

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0px 1px 3px rgba(15,23,42,0.05);
}

/* =====================================================
   CARDS
===================================================== */

.custom-card {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 1.5rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}

/* =====================================================
   INSIGHTS
===================================================== */

.insight-box {
    background: white;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #2563EB;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 12px;
}

/* =====================================================
   STATUS BOXES
===================================================== */

.good-box {
    background: #ECFDF5;
    border: 1px solid #BBF7D0;
    border-left: 5px solid #22C55E;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 12px;
}

.warning-box {
    background: #FEFCE8;
    border: 1px solid #FDE68A;
    border-left: 5px solid #F59E0B;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 12px;
}

.bad-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 5px solid #EF4444;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 12px;
}

/* =====================================================
   DATAFRAME
===================================================== */

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}

/* =====================================================
   ALERTS
===================================================== */

.stAlert {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

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

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# INPUTS
# =====================================================

st.subheader("Product Information")

col1, col2 = st.columns(2)

with col1:

    product_name = st.text_input(
        "Product Name",
        placeholder="Organic Protein Snack Bar"
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
        "Brand",
        placeholder="NatureFit"
    )

    score = st.slider(
        "Product Rating",
        1,
        5,
        5
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PRODUCT DESCRIPTION
# =====================================================

product_description = st.text_area(
    "Product Description",
    height=160,
    placeholder="""
Healthy organic protein snack made with natural ingredients,
high fiber, gluten free, low sugar and premium packaging.
Ideal for athletes and healthy lifestyles.
"""
)

# =====================================================
# REVIEW
# =====================================================

review_text = st.text_area(
    "Customer Review",
    height=220,
    placeholder="""
I have been using these protein bars for two weeks after workouts.
The flavor tastes natural and not overly sweet.
Packaging quality was excellent and ingredients feel premium.
Very good option for healthy snacks.
"""
)

# =====================================================
# MODEL
# =====================================================

model_name = st.selectbox(
    "AI Model",
    [
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "logistic_regression"
    ]
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# ANALYZE BUTTON
# =====================================================

if st.button(
    "Analyze Product Review",
    use_container_width=True
):

    # =================================================
    # VALIDATION
    # =================================================

    if not product_name.strip():

        st.error(
            "Product name is required."
        )

        st.stop()

    if not review_text.strip():

        st.error(
            "Customer review is required."
        )

        st.stop()

    try:

        # =================================================
        # API REQUEST
        # =================================================

        payload = {

            "text": review_text,

            "score": score,

            "model_name": model_name
        }

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

        # =================================================
        # NLP ADJUSTMENTS
        # =================================================

        if word_count < 20:

            probability *= 0.75

        elif word_count < 40:

            probability *= 0.90

        probability = min(probability, 100)

        # =================================================
        # PRODUCT CONSISTENCY
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

        if features['uppercase_ratio'] > 0.10:

            spam_risk = "Medium"

        if features['coherence'] == 0:

            spam_risk = "High"

        # =================================================
        # QUALITY
        # =================================================

        if probability >= 85:

            quality = "Excellent"

        elif probability >= 70:

            quality = "Good"

        elif probability >= 50:

            quality = "Average"

        else:

            quality = "Low"

        # =================================================
        # REVIEW CLASSIFICATION
        # =================================================

        review_rank = "Average Review"

        if (
            probability >= 90
            and consistency_score >= 80
            and features['coherence'] == 1
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
        # EXECUTIVE SUMMARY
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
                round(sentiment, 2)
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
        # REVIEW CLASSIFICATION
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

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # PRODUCT INTELLIGENCE
        # =================================================

        st.subheader("Product Intelligence")

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.markdown(f"""
### Product Metadata

**Product:** {product_name}

**Brand:** {brand}

**Category:** {category}
""")

        with col2:

            with st.container(border=True):

                st.markdown(f"""
### AI Metrics

**Consistency Score:** {consistency_score}%

**Word Count:** {word_count}

**Marketplace Priority:** {marketplace_priority}
""")

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # NLP FEATURES
        # =================================================

        st.subheader("NLP Features")

        features_df = pd.DataFrame(
            features.items(),
            columns=['Feature', 'Value']
        )

        st.dataframe(
            features_df,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

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

        if features['coherence'] == 1:

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

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # BUSINESS IMPACT
        # =================================================

        with st.container(border=True):

            st.subheader(
                "Executive Business Impact"
            )

            st.markdown("""
### Strategic Value

This module enables:

- Product perception analytics
- AI-driven moderation
- Marketplace trust optimization
- Product intelligence extraction
- Customer sentiment analytics
- Spam review reduction
- Explainable AI for e-commerce

### Enterprise Impact

- Better customer trust
- Improved recommendation systems
- Reduced fraudulent reviews
- Enhanced marketplace quality
- AI-powered product analytics
- Enterprise scalable architecture
""")

    except Exception as e:

        st.error(
            f"FastAPI Error: {e}"
        )