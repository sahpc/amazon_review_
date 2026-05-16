# =====================================================
# BATCH REVIEW ANALYZER
# ENTERPRISE AI PLATFORM
# =====================================================

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Batch Review Analyzer",
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
   BUTTONS
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

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    margin-bottom: 1rem;
    box-shadow: 0px 1px 3px rgba(15,23,42,0.04);
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

/* =====================================================
   UPLOAD
===================================================== */

section[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #CBD5E1;
    border-radius: 18px;
    padding: 1rem;
}

/* =====================================================
   PLOTLY
===================================================== */

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
Enterprise AI Batch Processing for Customer Review Intelligence
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# OVERVIEW
# =====================================================

with st.container(border=True):

    st.markdown("""
### Platform Capabilities

This module enables large-scale AI analysis for customer reviews using:

- NLP Analytics
- Multi-Model Machine Learning
- Sentiment Analysis
- Spam Detection
- Explainable AI
- Review Quality Scoring
- Batch Intelligence Processing
""")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Reviews CSV",
    type=["csv"]
)

# =====================================================
# PROCESS
# =====================================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # =====================================================
    # AUTO DETECTION
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
            "No review column detected."
        )

        st.stop()

    if score_col is None:

        st.error(
            "No score column detected."
        )

        st.stop()

    st.success(f"""
Review Column Detected: {review_col}

Score Column Detected: {score_col}
""")

    # =====================================================
    # MODEL
    # =====================================================

    model_name = st.selectbox(

        "Select AI Model",

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
    # BUTTON
    # =====================================================

    if st.button(
        "Analyze Batch Reviews",
        use_container_width=True
    ):

        results = []
        errors = []

        progress = st.progress(0)

        total = len(df)

        # =================================================
        # PROCESS LOOP
        # =================================================

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

                response = requests.post(

                    f"{API_URL}/reviews/predict_helpfulness",

                    json=payload
                )

                data = response.json()

                probability = (
                    data[
                        'probability_helpful'
                    ] * 100
                )

                features = data['features']

                # =========================================
                # QUALITY
                # =========================================

                if probability >= 85:

                    quality = "Excellent"

                elif probability >= 70:

                    quality = "Good"

                elif probability >= 50:

                    quality = "Average"

                else:

                    quality = "Low"

                # =========================================
                # SPAM DETECTION
                # =========================================

                spam_flag = False

                if (
                    features['uppercase_ratio']
                    > 0.10
                ):

                    spam_flag = True

                if (
                    features['word_count']
                    < 5
                ):

                    spam_flag = True

                # =========================================
                # NLP SCORE
                # =========================================

                nlp_score = 50

                if (
                    features['word_count']
                    > 50
                ):

                    nlp_score += 20

                if (
                    abs(
                        features[
                            'sentiment_compound'
                        ]
                    ) > 0.5
                ):

                    nlp_score += 15

                if (
                    features['coherence']
                    == 1
                ):

                    nlp_score += 15

                # =========================================
                # SAVE RESULTS
                # =========================================

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
                            features[
                                'sentiment_compound'
                            ],
                            3
                        ),

                    "word_count":
                        features[
                            'word_count'
                        ],

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
        # RESULTS DATAFRAME
        # =====================================================

        results_df = pd.DataFrame(results)

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

        st.subheader("Batch Analysis Results")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # HELPFULNESS DISTRIBUTION
        # =====================================================

        st.subheader(
            "Helpfulness Distribution"
        )

        fig = px.histogram(

            results_df,

            x='helpfulness',

            nbins=20,

            color_discrete_sequence=["#2563EB"]
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
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================================
        # QUALITY DISTRIBUTION
        # =====================================================

        st.subheader(
            "Quality Distribution"
        )

        quality_counts = (
            results_df[
                'quality'
            ].value_counts()
        )

        fig2 = px.pie(

            names=quality_counts.index,

            values=quality_counts.values
        )

        fig2.update_layout(

            paper_bgcolor="white",

            font=dict(
                family="Inter",
                size=14,
                color="#0F172A"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # TOP REVIEWS
        # =====================================================

        st.subheader(
            "Top Useful Reviews"
        )

        top_reviews = results_df.sort_values(
            by="helpfulness",
            ascending=False
        ).head(5)

        st.dataframe(
            top_reviews,
            use_container_width=True
        )

        # =====================================================
        # LOW QUALITY REVIEWS
        # =====================================================

        st.subheader(
            "Low Quality Reviews"
        )

        low_reviews = results_df.sort_values(
            by="helpfulness",
            ascending=True
        ).head(5)

        st.dataframe(
            low_reviews,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # EXECUTIVE INSIGHTS
        # =====================================================

        st.subheader(
            "Executive AI Insights"
        )

        if avg_helpfulness >= 80:

            st.success(
                "Dataset contains high-quality customer feedback."
            )

        else:

            st.warning(
                "Large volume of low-value reviews detected."
            )

        if spam_pct > 20:

            st.error(
                "High spam probability detected in uploaded dataset."
            )

        else:

            st.success(
                "Spam levels appear controlled."
            )

        # =====================================================
        # ERRORS
        # =====================================================

        if errors:

            st.subheader(
                "Processing Errors"
            )

            st.dataframe(
                pd.DataFrame(errors),
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # DOWNLOAD
        # =====================================================

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(

            "Download Analysis CSV",

            csv,

            file_name=f"""
batch_analysis_{
datetime.now().strftime('%Y%m%d_%H%M%S')
}.csv
""",

            mime="text/csv"
        )