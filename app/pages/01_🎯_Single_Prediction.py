"""
Single Prediction Page

Make predictions for individual publications.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

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


def main():
    st.set_page_config(
        page_title="Single Prediction - Citation Predictor",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Single Publication Prediction")
    st.markdown("""
    Enter the details of a publication to predict its citation impact.
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

            These files should be generated during the training phase using the notebooks.
            """)
            return

        # Get available models
        available_models = predictor.get_available_models()

        if not available_models['classification'] or not available_models['regression']:
            st.error("❌ No trained models found. Please train models using the notebooks first.")
            return

    except Exception as e:
        st.error(f"Error loading prediction service: {e}")
        st.info("""
        **To enable predictions:**
        1. Run the feature engineering notebooks (20-23) to create the TF-IDF vectorizer
        2. Run the model training notebooks (30-31) to train models
        3. Ensure models are saved in the `models/` directory
        """)
        return

    # Create form
    with st.form("prediction_form"):
        st.subheader("Publication Details")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input(
                "Title *",
                placeholder="Enter the publication title",
                help="Full title of the research paper"
            )

            venue = st.text_input(
                "Venue/Journal *",
                placeholder="e.g., Nature, IEEE Transactions on...",
                help="Name of the journal or conference"
            )

            year = st.number_input(
                "Publication Year *",
                min_value=2000,
                max_value=2030,
                value=2023,
                help="Year of publication"
            )

        with col2:
            authors = st.text_area(
                "Authors *",
                placeholder="Last, First; Last, First; ...",
                help="Authors separated by semicolons",
                height=100
            )

            h_indices = st.text_input(
                "H-indices",
                placeholder="10;15;20;...",
                help="H-index values separated by semicolons (same order as authors)"
            )

        abstract = st.text_area(
            "Abstract *",
            placeholder="Enter the publication abstract...",
            help="Full abstract of the research paper",
            height=200
        )

        # Metadata features
        st.subheader("Publication Metadata")
        st.caption("These 8 features contribute 26.9% of the model's predictive power.")

        col5, col6 = st.columns(2)

        with col5:
            is_open_access = st.checkbox(
                "Open Access",
                value=False,
                help="Check if the paper is published open access"
            )

            topic_prominence = st.slider(
                "Topic Prominence Percentile",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                help="Scopus Topic Prominence Percentile (0-100). This is the #1 most important feature. Find it in Scopus under the paper's topic details."
            )

        with col6:
            publication_type = st.selectbox(
                "Publication Type",
                options=["Article", "Review"],
                index=0,
                help="Select whether this is a research article or a review paper"
            )

            source_type = st.selectbox(
                "Source Type",
                options=["Journal", "Conference Proceeding", "Book", "Book Series"],
                index=0,
                help="Select the type of publication venue"
            )

        # Model selection
        st.subheader("Model Selection")
        col3, col4 = st.columns(2)

        with col3:
            classification_model = st.selectbox(
                "Classification Model",
                options=available_models['classification'],
                help="Model for predicting high-impact (top 25%) classification"
            )

        with col4:
            regression_model = st.selectbox(
                "Regression Model",
                options=available_models['regression'],
                help="Model for predicting citation count"
            )

        # Submit button
        submit = st.form_submit_button("🔮 Predict Citation Impact", use_container_width=True)

    # Make prediction
    if submit:
        # Validate inputs
        if not title or not abstract or not venue or not authors:
            st.error("❌ Please fill in all required fields (marked with *)")
            return

        try:
            with st.spinner("Making predictions..."):
                result = predictor.predict_single(
                    title=title,
                    abstract=abstract,
                    venue=venue,
                    authors=authors,
                    h_indices=h_indices if h_indices else "",
                    year=year,
                    is_open_access=is_open_access,
                    topic_prominence=topic_prominence,
                    publication_type=publication_type,
                    source_type=source_type,
                    classification_model=classification_model,
                    regression_model=regression_model
                )

            # Display results
            st.success("✅ Predictions completed!")

            st.markdown("---")
            st.subheader("📊 Prediction Results")

            # Create columns for results
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "High-Impact Prediction",
                    "Yes ✨" if result['is_high_impact'] else "No",
                    help="Top 25% citation prediction"
                )

                probability = result['high_impact_probability']
                st.metric(
                    "High-Impact Probability",
                    f"{probability:.1%}",
                    help="Probability of being in top 25% most cited"
                )

            with col2:
                citations = result['predicted_citations']
                st.metric(
                    "Predicted Citation Count",
                    f"{citations:.0f}",
                    help="Estimated number of citations"
                )

                # Citation range (rough estimate)
                lower = max(0, citations * 0.7)
                upper = citations * 1.3
                st.metric(
                    "Estimated Range",
                    f"{lower:.0f} - {upper:.0f}",
                    help="Approximate citation range (±30%)"
                )

            # Interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")

            if result['is_high_impact']:
                st.success(f"""
                **High Impact Predicted!** 🎉

                This publication is predicted to be in the **top 25%** of most cited papers
                with a confidence of **{probability:.1%}**.

                Expected citations: **~{citations:.0f}**
                """)
            else:
                st.info(f"""
                **Moderate Impact Predicted**

                This publication is predicted to receive **~{citations:.0f}** citations,
                which is below the top 25% threshold.

                Probability of high impact: **{probability:.1%}**
                """)

            # Show input summary
            with st.expander("📝 Input Summary"):
                st.markdown(f"""
                **Title:** {title}

                **Venue:** {venue}

                **Year:** {year}

                **Authors:** {authors}

                **H-indices:** {h_indices if h_indices else "Not provided"}

                **Open Access:** {"Yes" if is_open_access else "No"}

                **Topic Prominence:** {topic_prominence:.1f} / 100

                **Publication Type:** {publication_type}

                **Source Type:** {source_type}

                **Abstract:** {abstract[:200]}{'...' if len(abstract) > 200 else ''}
                """)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
