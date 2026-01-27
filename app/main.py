"""
CitaPred: Citation Predictor for AUB Research Papers
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Page configuration
st.set_page_config(
    page_title="CitaPred - Citation Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and feature data
@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        models_dir = project_root / "models"

        with open(models_dir / "classification" / "lightgbm.pkl", "rb") as f:
            clf_model = pickle.load(f)

        with open(models_dir / "regression" / "random_forest.pkl", "rb") as f:
            reg_model = pickle.load(f)

        return clf_model, reg_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_feature_importance():
    """Load feature importance data."""
    try:
        # Load the feature names from saved features
        features_dir = project_root / "data" / "features"
        X = pd.read_pickle(features_dir / "X_all.pkl")
        feature_names = X.columns.tolist()

        return feature_names
    except Exception as e:
        st.error(f"Error loading features: {e}")
        return []

def main():
    """Main application."""

    # Header
    st.title("📊 CitaPred: Citation Impact Predictor")
    st.markdown("### Predict Research Impact for AUB Publications")
    st.markdown("---")

    # Sidebar navigation
    with st.sidebar:
        st.header("📍 Navigation")
        page = st.radio(
            "Select a page:",
            ["🏠 Home", "🎯 Make Prediction", "📊 Model Performance", "🔍 Feature Importance", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("### 📈 Quick Stats")
        st.metric("Total Papers", "14,832")
        st.metric("Train (2015-2017)", "2,545")
        st.metric("Test (2018-2020)", "3,573")
        st.metric("Total Features", "5,019")
        st.metric("Best F1 Score", "68.2%")
        st.metric("Best R² Score", "48.2%")

    # Route to different pages
    if page == "🏠 Home":
        show_home()
    elif page == "🎯 Make Prediction":
        show_prediction()
    elif page == "📊 Model Performance":
        show_performance()
    elif page == "🔍 Feature Importance":
        show_feature_importance()
    else:
        show_about()

def show_home():
    """Display home page."""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Welcome to CitaPred!")
        st.markdown("""
        CitaPred is a machine learning system that predicts the citation impact of research publications
        from the American University of Beirut (AUB).

        **What can you do with CitaPred?**
        - 🎯 **Predict** whether a paper will be high-impact (top 25% by citations)
        - 📈 **Estimate** the expected number of citations
        - 🔍 **Understand** what factors drive citation impact
        - 📊 **Explore** model performance and validation results

        **How does it work?**

        The system analyzes **5,019 features** from three categories:
        - **Text Features** (5,000): TF-IDF analysis of paper abstracts
        - **Venue Features** (9): Journal prestige metrics (SNIP, SJR, CiteScore, percentiles)
        - **Author Features** (10): Collaboration patterns and team composition

        **Key Findings:**
        - 📰 **Venue Prestige** (SJR, CiteScore) is a strong predictor of citation impact
        - ✍️ **Abstract Content** provides significant predictive power
        - 📊 **Journal Metrics** (SNIP, CiteScore percentiles) matter greatly
        - 👥 **Author Collaboration** has minimal direct effect
        """)

    with col2:
        st.info("""
        **Model Performance**

        **Classification:**
        - F1 Score: 68.2%
        - ROC-AUC: 83.7%
        - Accuracy: 76.5%

        **Regression:**
        - R² Score: 48.2%
        - Spearman: 67.5%
        - MAE: 0.69 (log scale)

        **Validation:**
        - Temporal split (2015-2017 train, 2018-2020 test)
        - No data leakage
        - Scientifically valid
        """)

        st.success("""
        **✅ Ready to use!**

        Navigate to **Make Prediction**
        to predict citation impact for
        a new research paper.
        """)

def show_prediction():
    """Display prediction page."""

    st.subheader("🎯 Predict Citation Impact")
    st.markdown("Enter paper details below to get predictions:")

    st.info("⚠️ **Note**: This is a demo interface. For full predictions, you need the complete feature set including TF-IDF vectors from the abstract text processing pipeline.")

    # Input form
    with st.form("prediction_form"):
        st.markdown("### Paper Information")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("📄 Title", placeholder="Enter paper title...")
            abstract = st.text_area("📝 Abstract", placeholder="Enter paper abstract...", height=150)
            year = st.number_input("📅 Publication Year", min_value=2015, max_value=2025, value=2023)

        with col2:
            st.markdown("### Venue Information")
            citescore = st.number_input("📊 CiteScore", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
            sjr = st.number_input("📈 SJR", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
            snip = st.number_input("📉 SNIP", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

        st.markdown("### Author Information")
        col3, col4 = st.columns(2)

        with col3:
            num_authors = st.number_input("👥 Number of Authors", min_value=1, max_value=100, value=3)
            num_institutions = st.number_input("🏛️ Number of Institutions", min_value=1, max_value=50, value=1)

        with col4:
            is_international = st.checkbox("🌍 International Collaboration")
            is_multi_institution = st.checkbox("🏢 Multi-Institution")

        submitted = st.form_submit_button("🚀 Predict Citation Impact", use_container_width=True)

        if submitted:
            if not title or not abstract:
                st.error("❌ Please enter both title and abstract!")
            else:
                # Show predictions (simplified demo)
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                # Simple heuristic for demo (not actual model prediction)
                score = (
                    citescore * 25 +
                    sjr * 35 +
                    snip * 15 +
                    num_authors * 3
                )

                # Normalize to probability
                probability = min(score / 250, 0.95)
                predicted_citations = int(score * 0.9)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "High-Impact Prediction",
                        "YES ✅" if probability > 0.5 else "NO ❌",
                        f"{probability*100:.1f}% confidence"
                    )

                with col2:
                    st.metric(
                        "Predicted Citations",
                        f"~{predicted_citations}",
                        "in 3-5 years"
                    )

                with col3:
                    impact_category = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
                    st.metric(
                        "Impact Category",
                        impact_category,
                        f"Top {int((1-probability)*100)}%"
                    )

                # Show feature contribution
                st.markdown("### 🔍 Key Factors")

                factors_data = {
                    "Factor": ["SJR", "CiteScore", "SNIP", "Authors"],
                    "Value": [sjr, citescore, snip, num_authors],
                    "Importance": [35, 32, 28, 5]
                }
                factors_df = pd.DataFrame(factors_data)

                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(factors_df["Factor"], factors_df["Importance"], color='steelblue')
                ax.set_xlabel("Importance (%)")
                ax.set_title("Feature Contribution to Prediction")
                st.pyplot(fig)

                st.warning("""
                **⚠️ Demo Limitation**: This is a simplified demonstration.
                The actual model uses 5,019 features including full TF-IDF analysis of the abstract text.
                For production use, integrate with the full feature engineering pipeline.
                """)

def show_performance():
    """Display model performance page."""

    st.subheader("📊 Model Performance")

    tab1, tab2 = st.tabs(["Classification", "Regression"])

    with tab1:
        st.markdown("### Classification Model Performance")
        st.markdown("**Task:** Identify high-impact papers (top 25% by citations, threshold: 26 citations)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", "76.5%")
        with col2:
            st.metric("F1 Score", "68.2%")
        with col3:
            st.metric("Precision", "58.6%")
        with col4:
            st.metric("ROC-AUC", "83.7%")

        st.markdown("---")

        st.markdown("**Model Comparison:**")

        perf_data = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
            "F1 Score": [0.682, 0.651, 0.676, 0.679],
            "ROC-AUC": [0.832, 0.813, 0.814, 0.823],
            "Accuracy": [0.755, 0.743, 0.731, 0.750]
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)

        st.markdown("**Classification Errors:**")
        col1, col2 = st.columns(2)
        with col1:
            st.error("**False Positives:** 560 papers  \nPredicted high-impact but got 14.4 avg citations")
        with col2:
            st.warning("**False Negatives:** 297 papers  \nPredicted low-impact but got 44.6 avg citations")

    with tab2:
        st.markdown("### Regression Model Performance")
        st.markdown("**Task:** Predict citation counts (log-transformed)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("R² Score", "48.2%")
        with col2:
            st.metric("Spearman ρ", "67.5%")
        with col3:
            st.metric("MAE", "0.69", help="Mean Absolute Error (log scale)")
        with col4:
            st.metric("RMSE", "0.88", help="Root Mean Squared Error (log scale)")

        st.markdown("---")

        st.markdown("**Model Comparison:**")

        reg_data = {
            "Model": ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"],
            "R² Score": [-332.78, 0.482, 0.474, 0.478],
            "Spearman": [0.102, 0.675, 0.669, 0.671],
            "MAE": [3.927, 0.692, 0.699, 0.696]
        }
        reg_df = pd.DataFrame(reg_data)
        st.dataframe(reg_df, use_container_width=True)

        st.markdown("**Error Distribution by Citation Range:**")

        error_data = {
            "Citation Range": ["0-5", "6-10", "11-25", "26-50", "51-100", "101-1000", "1000+"],
            "Mean Error": [0.944, 0.521, 0.425, 0.567, 0.774, 1.190, 2.007],
            "Papers": [752, 582, 1022, 602, 299, 160, 7]
        }
        error_df = pd.DataFrame(error_data)
        st.dataframe(error_df, use_container_width=True)

        st.info("""
        **Key Insights:**
        - Best performance for mid-range citations (11-50)
        - Struggles with very low (0-5) and very high (1000+) citations
        - Linear regression completely failed (negative R²)
        - Tree-based models perform best
        """)
        st.info("👈 Navigate to **Model Performance** page to view metrics!")

def show_feature_importance():
    """Display feature importance page."""

    st.subheader("🔍 Feature Importance Analysis")

    st.markdown("""
    Understanding what drives citation impact based on trained models.
    """)

    tab1, tab2, tab3 = st.tabs(["Top Features", "By Category", "Insights"])

    with tab1:
        st.markdown("### Top 20 Most Important Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Classification (LightGBM)**")
            clf_features = [
                ("citescore", 45, "Venue"),
                ("sjr", 44, "Venue"),
                ("venue_score_composite", 34, "Venue"),
                ("snip", 18, "Venue"),
                ("study", 17, "Text"),
                ("num_authors", 17, "Author"),
                ("avg_venue_percentile", 16, "Venue"),
                ("authors_per_institution", 15, "Author"),
                ("citescore_percentile", 14, "Venue"),
                ("results", 13, "Text")
            ]
            clf_df = pd.DataFrame(clf_features, columns=["Feature", "Importance", "Category"])
            st.dataframe(clf_df, use_container_width=True)

        with col2:
            st.markdown("**Regression (Random Forest)**")
            reg_features = [
                ("sjr", 0.150, "Venue"),
                ("citescore", 0.055, "Venue"),
                ("venue_score_composite", 0.040, "Venue"),
                ("avg_venue_percentile", 0.025, "Venue"),
                ("snip", 0.015, "Venue"),
                ("num_authors", 0.008, "Author"),
                ("2015", 0.006, "Text"),
                ("review", 0.005, "Text"),
                ("study", 0.004, "Text"),
                ("results", 0.004, "Text")
            ]
            reg_df = pd.DataFrame(reg_features, columns=["Feature", "Importance", "Category"])
            st.dataframe(reg_df, use_container_width=True)

    with tab2:
        st.markdown("### Importance by Category")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Classification**")
            cat_clf = {
                "Category": ["Text", "Venue", "Other", "Author"],
                "Importance": [711, 226, 134, 49],
                "Percentage": [63.5, 20.2, 12.0, 4.4]
            }
            cat_clf_df = pd.DataFrame(cat_clf)
            st.dataframe(cat_clf_df, use_container_width=True)

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.bar(cat_clf_df["Category"], cat_clf_df["Percentage"], color=['steelblue', 'coral', 'gray', 'green'])
            ax1.set_ylabel("Importance (%)")
            ax1.set_title("Classification Feature Categories")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Regression**")
            cat_reg = {
                "Category": ["Text", "Venue", "Other", "Author"],
                "Importance": [0.397, 0.311, 0.276, 0.017],
                "Percentage": [39.7, 31.1, 27.6, 1.7]
            }
            cat_reg_df = pd.DataFrame(cat_reg)
            st.dataframe(cat_reg_df, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(cat_reg_df["Category"], cat_reg_df["Percentage"], color=['steelblue', 'coral', 'gray', 'green'])
            ax2.set_ylabel("Importance (%)")
            ax2.set_title("Regression Feature Categories")
            st.pyplot(fig2)

    with tab3:
        st.markdown("### 💡 Key Insights")

        st.success("""
        **🔥 #1 Predictor: Venue Prestige**

        The most important features are journal prestige metrics (CiteScore, SJR).
        Publishing in high-quality venues significantly increases citation potential.
        """)

        st.info("""
        **📰 Venue Prestige Matters**

        Publishing in high-prestige journals (high SJR, CiteScore, SNIP) significantly increases citation potential.
        Venue features account for 20-31% of predictive power.
        """)

        st.warning("""
        **✍️ Content Has Moderate Impact**

        Abstract content (TF-IDF features) provides moderate predictive power (39-63%).
        Specific keywords like "study", "results", "review" appear in high-impact papers.
        """)

        st.error("""
        **👥 Collaboration Has Minimal Direct Effect**

        Author collaboration features (team size, international collaboration) have surprisingly low importance (1.7-4.4%).
        The effect may be indirect through other factors.
        """)

def show_about():
    """Display about page."""

    st.subheader("ℹ️ About CitaPred")

    st.markdown("""
    ### Project Overview

    **CitaPred** (Citation Predictor) is a machine learning system developed to predict the citation impact
    of research publications from the American University of Beirut (AUB).

    ### Methodology

    **Data Sources:**
    - **Scopus**: 15,748 publications with abstracts
    - **SciVal**: 15,493 publications with citation metrics
    - **Merged Dataset**: 14,832 clean publications (2015-2020)

    **Feature Engineering:**
    - **Text Features** (5,000): TF-IDF vectorization of abstracts (1-2 grams, min_df=5, max_df=0.8)
    - **Venue Features** (9): SNIP, SJR, CiteScore + percentiles, composite scores
    - **Author Features** (10): Team size, collaboration metrics, institutional diversity

    **Models:**
    - **Classification**: Logistic Regression, Random Forest, XGBoost, LightGBM
    - **Regression**: Random Forest, XGBoost, LightGBM (Linear Regression failed)
    - **Target**: Top 25% papers (≥26 citations) for classification, log(citations+1) for regression

    **Validation:**
    - **Temporal Split**: Train on 2015-2017 (2,545 papers), test on 2018-2020 (3,573 papers)
    - **No Data Leakage**: Removed citation-derived features (field_weighted_citation_impact, etc.)
    - **Cross-Validation**: 5-fold stratified CV for classification

    ### Performance

    **Best Classification Model: Logistic Regression**
    - F1 Score: 68.2%
    - ROC-AUC: 83.7%
    - Accuracy: 76.5%

    **Best Regression Model: Random Forest**
    - R² Score: 48.2%
    - Spearman Correlation: 67.5%
    - MAE: 0.69 (log scale)

    ### Limitations

    - Cannot predict "viral" papers or "sleeper hits"
    - Performance degrades for extreme citation counts (0-5, 1000+)
    - Highly skewed citation distribution (median: 10, max: 26,135)
    - Limited to AUB publications (may not generalize)

    ### Technology Stack

    - **ML Libraries**: scikit-learn, XGBoost, LightGBM
    - **Data Processing**: pandas, numpy
    - **NLP**: TfidfVectorizer, NLTK
    - **Visualization**: matplotlib, seaborn
    - **Deployment**: Streamlit

    ### Team

    **Student**: Mario Chartouni (AUB ID: 202575069)
    **Project**: MSBA Capstone Project
    **Institution**: American University of Beirut
    **Year**: 2024-2025

    ### Citation

    If you use this tool in your research, please cite:

    ```
    Chartouni, M. (2025). CitaPred: A Machine Learning System for Predicting
    Citation Impact of Research Publications. MSBA Capstone Project,
    American University of Beirut.
    ```

    ### Contact

    For questions or feedback, contact: mario.chartouni@aub.edu.lb
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>CitaPred v1.0 | Built with ❤️ using Streamlit | © 2025 AUB</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
