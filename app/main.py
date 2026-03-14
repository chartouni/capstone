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
        st.metric("Total Features", "5,023")
        st.metric("Best F1 Score", "~62.6% (AUB-only)")
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

        The system analyzes **5,023 features** from four categories:
        - **Text Features** (5,000): TF-IDF analysis of paper abstracts
        - **Venue Features** (5): Journal prestige percentiles (SNIP, CiteScore, SJR percentiles + derived)
        - **Author Features** (10): Collaboration patterns and team composition
        - **Metadata Features** (8): Open access, topic prominence, publication/source type

        **Key Findings:**
        - 📰 **Venue Prestige** (percentile scores) is a strong predictor of citation impact
        - ✍️ **Abstract Content** provides significant predictive power
        - 📊 **Topic Prominence Percentile** is the single most important non-text feature
        - 👥 **Author Collaboration** has minimal direct effect
        """)

    with col2:
        st.info("""
        **Model Performance**

        **Classification (AUB-only baseline):**
        - F1 Score: ~62.6%
        - ROC-AUC: ~81%
        - Features: percentile-only venue scores

        **Regression:**
        - R² Score: 48.2%
        - Spearman: 67.5%
        - MAE: 0.69 (log scale)

        **Validation:**
        - Temporal split (2015-2017 train, 2018-2020 test)
        - Percentile-only venue features (no leakage)
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
            st.caption("Enter percentile scores (0–100). Find these in SciVal under journal metrics.")
            citescore_pct = st.slider("📊 CiteScore Percentile", min_value=0, max_value=100, value=50)
            sjr_pct = st.slider("📈 SJR Percentile", min_value=0, max_value=100, value=50)
            snip_pct = st.slider("📉 SNIP Percentile", min_value=0, max_value=100, value=50)

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
                avg_pct = (citescore_pct + sjr_pct + snip_pct) / 3
                score = avg_pct * 0.7 + num_authors * 1.5

                # Normalize to probability
                probability = min(score / 100, 0.95)
                predicted_citations = int(avg_pct * 0.4)

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
                    "Factor": ["SJR Percentile", "CiteScore Percentile", "SNIP Percentile", "Authors"],
                    "Value": [sjr_pct, citescore_pct, snip_pct, num_authors],
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

        st.info("ℹ️ Numbers below are from the AUB-only baseline with clean (percentile-only) venue features. Re-run nb42 to get final model results.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", "~73%")
        with col2:
            st.metric("F1 Score", "~62.6%")
        with col3:
            st.metric("Precision", "~53%")
        with col4:
            st.metric("ROC-AUC", "~81%")

        st.markdown("---")

        st.markdown("**Model Comparison:**")

        st.caption("Placeholder — replace with actual values after re-running nb30 with clean features.")
        perf_data = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
            "F1 Score": ["~62.6%", "TBD", "TBD", "TBD"],
            "ROC-AUC": ["~81%", "TBD", "TBD", "TBD"],
            "Accuracy": ["~73%", "TBD", "TBD", "TBD"]
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

        st.caption("Feature importance below is from models trained with clean (percentile-only) venue features. Re-run nb40b for updated values.")
        with col1:
            st.markdown("**Classification (Logistic Regression)**")
            clf_features = [
                ("topic_prominence", 45, "Metadata"),
                ("avg_venue_percentile", 38, "Venue"),
                ("citescore_percentile", 28, "Venue"),
                ("snip_percentile", 22, "Venue"),
                ("sjr_percentile", 20, "Venue"),
                ("study", 17, "Text"),
                ("num_authors", 15, "Author"),
                ("is_top_journal", 14, "Venue"),
                ("authors_per_institution", 13, "Author"),
                ("results", 12, "Text")
            ]
            clf_df = pd.DataFrame(clf_features, columns=["Feature", "Importance", "Category"])
            st.dataframe(clf_df, use_container_width=True)

        with col2:
            st.markdown("**Regression (Random Forest)**")
            reg_features = [
                ("avg_venue_percentile", 0.085, "Venue"),
                ("citescore_percentile", 0.055, "Venue"),
                ("sjr_percentile", 0.040, "Venue"),
                ("topic_prominence", 0.030, "Metadata"),
                ("snip_percentile", 0.015, "Venue"),
                ("num_authors", 0.008, "Author"),
                ("review", 0.006, "Text"),
                ("study", 0.005, "Text"),
                ("results", 0.004, "Text"),
                ("2015", 0.004, "Text")
            ]
            reg_df = pd.DataFrame(reg_features, columns=["Feature", "Importance", "Category"])
            st.dataframe(reg_df, use_container_width=True)

    with tab2:
        st.markdown("### Importance by Category")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Classification**")
            cat_clf = {
                "Category": ["Text", "Venue", "Metadata", "Author"],
                "Importance": [711, 180, 110, 49],
                "Percentage": [67.5, 17.1, 10.5, 4.7]
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
                "Category": ["Text", "Venue", "Metadata", "Author"],
                "Importance": [0.397, 0.250, 0.140, 0.017],
                "Percentage": [49.5, 31.2, 17.5, 2.1]
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
    - **Venue Features** (5): SNIP, CiteScore, SJR **percentiles only** + avg_venue_percentile + is_top_journal
    - **Author Features** (10): Team size, collaboration metrics, institutional diversity
    - **Metadata Features** (8): Open access, topic prominence percentile, publication type, source type

    **Models:**
    - **Classification**: Logistic Regression, Random Forest, XGBoost, LightGBM
    - **Regression**: Random Forest, XGBoost, LightGBM (Linear Regression failed)
    - **Target**: Top 25% papers (≥26 citations) for classification, log(citations+1) for regression

    **Validation:**
    - **Temporal Split**: Train on 2015-2017 (2,545 papers), test on 2018-2020 (3,573 papers)
    - **No Data Leakage**: Removed citation-derived features (field_weighted_citation_impact, etc.)
    - **Cross-Validation**: 5-fold stratified CV for classification

    ### Performance

    **Best Classification Model: Logistic Regression (AUB-only baseline)**
    - F1 Score: ~62.6% (clean features, re-run nb42 for final result)
    - ROC-AUC: ~81%
    - Features: percentile-only venue scores (no temporal leakage)

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
