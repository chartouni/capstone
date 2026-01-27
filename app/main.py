"""
Main Streamlit application for Citation Predictor.

This is the entry point for the web application.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="Citation Predictor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("📊 Citation Predictor")
    st.markdown("""
    ### Predict Research Impact for AUB Publications

    This tool uses machine learning to predict citation counts and identify
    high-impact research papers based on publication metadata.
    """)

    # Sidebar
    with st.sidebar:
        st.header("📋 Quick Navigation")
        st.markdown("""
        Navigate using the sidebar menu to:
        - 🎯 **Single Prediction** - Predict for one paper
        - 📁 **Batch Prediction** - Predict for multiple papers
        - 📈 **Model Performance** - View model metrics

        👈 Use the menu on the left to get started!
        """)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This tool was developed as part of a capstone project in collaboration
        with the American University of Beirut (AUB) to predict citation impact
        of research publications.

        Built with scikit-learn, XGBoost, LightGBM, and Streamlit.
        """)

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Single Prediction")
        st.markdown("""
        Make predictions for a single publication by entering its metadata:
        - Title and abstract
        - Author information
        - Venue details
        - Publication year
        """)
        st.info("👈 Navigate to **Single Prediction** page using the sidebar menu to get started!")

    with col2:
        st.subheader("📁 Batch Prediction")
        st.markdown("""
        Upload a CSV file with multiple publications to get predictions
        for all of them at once.

        Download the results as a CSV file.
        """)
        st.info("👈 Navigate to **Batch Prediction** page using the sidebar menu to upload your CSV!")

    # Additional sections
    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📈 Model Performance")
        st.markdown("""
        View detailed performance metrics for trained models:
        - Classification accuracy and ROC curves
        - Regression predictions vs. actual
        - Temporal validation results
        """)
        st.info("👈 Navigate to **Model Performance** page to view metrics!")

    with col4:
        st.subheader("🛠️ Setup Requirements")
        st.markdown("""
        To use predictions, you need:
        - ✅ Trained models in `models/`
        - ✅ TF-IDF vectorizer in `data/features/`
        - ✅ Feature statistics

        See the prediction pages for detailed setup instructions.
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Citation Predictor v1.0 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
