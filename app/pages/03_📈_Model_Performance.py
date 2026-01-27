"""
Model Performance Page

Display model performance metrics and evaluation results.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_metrics():
    """Load model performance metrics from reports."""
    metrics_dir = project_root / "reports" / "metrics"

    metrics = {}

    # Load classification metrics
    cls_metrics_path = metrics_dir / "classification_results.csv"
    if cls_metrics_path.exists():
        metrics['classification'] = pd.read_csv(cls_metrics_path, index_col=0)
    else:
        metrics['classification'] = None

    # Load regression metrics
    reg_metrics_path = metrics_dir / "regression_results.csv"
    if reg_metrics_path.exists():
        metrics['regression'] = pd.read_csv(reg_metrics_path, index_col=0)
    else:
        metrics['regression'] = None

    return metrics


def main():
    st.set_page_config(
        page_title="Model Performance - Citation Predictor",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Model Performance")
    st.markdown("""
    View performance metrics for trained classification and regression models.
    """)

    # Load metrics
    metrics = load_metrics()

    # Check if metrics are available
    if metrics['classification'] is None and metrics['regression'] is None:
        st.warning("""
        ⚠️ No performance metrics found.

        Metrics are generated when you run the model training notebooks (30-31).
        They are saved in `reports/metrics/`.
        """)
        return

    # Display classification metrics
    if metrics['classification'] is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Models (High-Impact Prediction)")

        st.markdown("""
        Classification models predict whether a publication will be in the **top 25%** most cited papers.
        """)

        cls_df = metrics['classification']

        # Display metrics table
        st.dataframe(
            cls_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0),
            use_container_width=True
        )

        # Best models
        col1, col2 = st.columns(2)

        with col1:
            if 'Test_ROC_AUC' in cls_df.columns:
                best_roc_model = cls_df['Test_ROC_AUC'].idxmax()
                best_roc_score = cls_df['Test_ROC_AUC'].max()
                st.metric(
                    "Best ROC-AUC",
                    f"{best_roc_score:.4f}",
                    delta=best_roc_model
                )

        with col2:
            if 'Test_F1' in cls_df.columns:
                best_f1_model = cls_df['Test_F1'].idxmax()
                best_f1_score = cls_df['Test_F1'].max()
                st.metric(
                    "Best F1 Score",
                    f"{best_f1_score:.4f}",
                    delta=best_f1_model
                )

        # Explanation
        with st.expander("📚 Understanding Classification Metrics"):
            st.markdown("""
            **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
            - Measures the model's ability to distinguish between high-impact and regular papers
            - Range: 0.5 (random) to 1.0 (perfect)
            - Higher is better

            **F1 Score**
            - Harmonic mean of precision and recall
            - Range: 0.0 to 1.0
            - Balances false positives and false negatives
            - Higher is better

            **Cross-Validation (CV) vs Test**
            - CV metrics: Average performance across 5-fold cross-validation on training data
            - Test metrics: Performance on held-out temporal test set (2018-2020)
            """)

    # Display regression metrics
    if metrics['regression'] is not None:
        st.markdown("---")
        st.subheader("📊 Regression Models (Citation Count Prediction)")

        st.markdown("""
        Regression models predict the expected number of citations a publication will receive.
        """)

        reg_df = metrics['regression']

        # Display metrics table
        st.dataframe(
            reg_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0, vmin=0),
            use_container_width=True
        )

        # Best models
        col1, col2 = st.columns(2)

        with col1:
            if 'Test_R2' in reg_df.columns:
                best_r2_model = reg_df['Test_R2'].idxmax()
                best_r2_score = reg_df['Test_R2'].max()
                st.metric(
                    "Best R² Score",
                    f"{best_r2_score:.4f}",
                    delta=best_r2_model
                )

        with col2:
            if 'Test_MAE' in reg_df.columns:
                best_mae_model = reg_df['Test_MAE'].idxmin()
                best_mae_score = reg_df['Test_MAE'].min()
                st.metric(
                    "Best MAE (Lower is Better)",
                    f"{best_mae_score:.4f}",
                    delta=best_mae_model
                )

        # Explanation
        with st.expander("📚 Understanding Regression Metrics"):
            st.markdown("""
            **R² (R-squared / Coefficient of Determination)**
            - Measures how well the model explains variance in citation counts
            - Range: -∞ to 1.0
            - Higher is better (1.0 = perfect predictions)

            **MAE (Mean Absolute Error)**
            - Average absolute difference between predicted and actual citations
            - Lower is better
            - In log-space for this model

            **RMSE (Root Mean Squared Error)**
            - Similar to MAE but penalizes larger errors more heavily
            - Lower is better
            """)

    # Model Information
    st.markdown("---")
    st.subheader("ℹ️ Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Classification Models:**
        - Logistic Regression
        - Random Forest
        - XGBoost
        - LightGBM

        **Target:** Top 25% citations (binary)
        """)

    with col2:
        st.markdown("""
        **Regression Models:**
        - Random Forest Regressor
        - XGBoost Regressor
        - LightGBM Regressor

        **Target:** Log-transformed citation count
        """)

    # Training Information
    with st.expander("🔧 Training Details"):
        st.markdown("""
        **Temporal Validation:**
        - Training Set: Publications from 2015-2017
        - Test Set: Publications from 2018-2020

        **Features:**
        - Text Features: TF-IDF vectors (5000 features)
        - Venue Features: Prestige scores, average citations
        - Author Features: H-index statistics, collaboration metrics

        **Class Imbalance Handling:**
        - Class weights applied to handle top 25% imbalance
        - Stratified cross-validation for classification

        **Hyperparameters:**
        - Tuned using cross-validation
        - See notebooks 30-31 for details
        """)


if __name__ == "__main__":
    main()
