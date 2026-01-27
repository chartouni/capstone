"""
Batch Prediction Page

Make predictions for multiple publications from CSV file.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import io

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.deployment.prediction_service import CitationPredictor


# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load and initialize the prediction service."""
    predictor = CitationPredictor(
        models_dir=str(project_root / "models"),
        features_dir=str(project_root / "data" / "features")
    )
    status = predictor.initialize()
    return predictor, status


def create_template_csv():
    """Create a template CSV file for download."""
    template = pd.DataFrame({
        'Title': ['Example Paper 1', 'Example Paper 2'],
        'Abstract': [
            'This is an example abstract about machine learning...',
            'Another example abstract about neural networks...'
        ],
        'Scopus Source title': ['Nature', 'IEEE Transactions'],
        'Authors': ['Smith, John; Doe, Jane', 'Johnson, Bob'],
        'Authors H-index': ['15;20', '25'],
        'Year': [2023, 2022]
    })
    return template


def main():
    st.set_page_config(
        page_title="Batch Prediction - Citation Predictor",
        page_icon="📁",
        layout="wide"
    )

    st.title("📁 Batch Publication Prediction")
    st.markdown("""
    Upload a CSV file with multiple publications to get predictions for all of them at once.
    """)

    # Load predictor
    try:
        predictor, init_status = load_predictor()

        # Check if all required artifacts are available
        if not all(init_status.values()):
            st.warning("⚠️ Some prediction components are missing:")
            for component, status in init_status.items():
                if not status:
                    st.markdown(f"- ❌ {component}")

            st.info("""
            **To enable predictions, you need:**
            1. Trained models in `models/classification/` and `models/regression/`
            2. TF-IDF vectorizer in `data/features/tfidf_vectorizer.pkl`
            3. (Optional) Venue statistics in `data/features/venue_statistics.pkl`
            """)
            return

        # Get available models
        available_models = predictor.get_available_models()

        if not available_models['classification'] or not available_models['regression']:
            st.error("❌ No trained models found. Please train models using the notebooks first.")
            return

    except Exception as e:
        st.error(f"Error loading prediction service: {e}")
        return

    # Instructions
    with st.expander("📋 Instructions & CSV Format", expanded=True):
        st.markdown("""
        **Required Columns:**
        - `Title`: Publication title
        - `Abstract`: Full abstract text
        - `Scopus Source title`: Venue/journal name
        - `Authors`: Authors separated by semicolons (e.g., "Smith, John; Doe, Jane")
        - `Year`: Publication year

        **Optional Columns:**
        - `Authors H-index`: H-index values separated by semicolons (e.g., "15;20;10")
        - `EID`: Publication identifier (for tracking)

        **Note:** All text fields should be properly escaped if they contain commas or quotes.
        """)

        # Download template
        template = create_template_csv()
        csv_buffer = io.StringIO()
        template.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Template CSV",
            data=csv_data,
            file_name="citation_prediction_template.csv",
            mime="text/csv",
            help="Download a template CSV file with the correct format"
        )

    # File upload
    st.markdown("---")
    st.subheader("📤 Upload CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with publication data"
    )

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ File uploaded successfully! Found {len(df)} publications.")

            # Show preview
            with st.expander("👀 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Validate required columns
            required_cols = ['Title', 'Abstract', 'Scopus Source title', 'Authors', 'Year']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return

            # Model selection
            st.markdown("---")
            st.subheader("🤖 Model Selection")

            col1, col2 = st.columns(2)

            with col1:
                classification_model = st.selectbox(
                    "Classification Model",
                    options=available_models['classification'],
                    help="Model for high-impact prediction"
                )

            with col2:
                regression_model = st.selectbox(
                    "Regression Model",
                    options=available_models['regression'],
                    help="Model for citation count prediction"
                )

            # Predict button
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                try:
                    with st.spinner(f"Making predictions for {len(df)} publications..."):
                        results = predictor.predict_batch(
                            df,
                            classification_model=classification_model,
                            regression_model=regression_model
                        )

                    st.success("✅ Predictions completed!")

                    # Display summary statistics
                    st.markdown("---")
                    st.subheader("📊 Summary Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        high_impact_count = results['is_high_impact'].sum()
                        st.metric(
                            "High-Impact Papers",
                            f"{high_impact_count}",
                            f"{high_impact_count/len(results)*100:.1f}%"
                        )

                    with col2:
                        avg_citations = results['predicted_citations'].mean()
                        st.metric(
                            "Avg. Predicted Citations",
                            f"{avg_citations:.0f}"
                        )

                    with col3:
                        max_citations = results['predicted_citations'].max()
                        st.metric(
                            "Max Predicted Citations",
                            f"{max_citations:.0f}"
                        )

                    with col4:
                        avg_probability = results['high_impact_probability'].mean()
                        st.metric(
                            "Avg. High-Impact Prob.",
                            f"{avg_probability:.1%}"
                        )

                    # Display results
                    st.markdown("---")
                    st.subheader("📋 Prediction Results")

                    # Select columns to display
                    display_cols = ['Title', 'Year', 'Scopus Source title',
                                    'is_high_impact', 'high_impact_probability',
                                    'predicted_citations']

                    # Add EID if it exists
                    if 'EID' in results.columns:
                        display_cols = ['EID'] + display_cols

                    st.dataframe(
                        results[display_cols].style.format({
                            'high_impact_probability': '{:.1%}',
                            'predicted_citations': '{:.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

                    # Download results
                    st.markdown("---")
                    st.subheader("💾 Download Results")

                    # Prepare download
                    csv_buffer = io.StringIO()
                    results.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv_data,
                            file_name="citation_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:
                        st.info("The CSV file contains all original columns plus the prediction results.")

                    # Distribution plots
                    st.markdown("---")
                    st.subheader("📈 Prediction Distributions")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.bar_chart(
                            results['is_high_impact'].value_counts(),
                            use_container_width=True
                        )
                        st.caption("High-Impact Distribution (0=No, 1=Yes)")

                    with col2:
                        st.bar_chart(
                            pd.cut(results['predicted_citations'],
                                   bins=[0, 10, 25, 50, 100, float('inf')],
                                   labels=['0-10', '11-25', '26-50', '51-100', '100+']).value_counts().sort_index(),
                            use_container_width=True
                        )
                        st.caption("Predicted Citation Count Distribution")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.exception(e)

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            st.info("Please ensure your CSV file is properly formatted.")


if __name__ == "__main__":
    main()
