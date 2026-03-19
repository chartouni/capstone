"""
Single Prediction Page

Make predictions for individual publications.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

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


@st.cache_data
def load_merged_dataset():
    """
    Load the merged dataset for random paper sampling.
    Returns a DataFrame or None if not available.
    """
    candidates = [
        project_root / "data" / "processed" / "merged.csv",
        project_root / "data" / "processed" / "merged_data.csv",
        project_root / "data" / "processed" / "merged_data.pkl",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".pkl":
                return pd.read_pickle(path)
            else:
                return pd.read_csv(path, low_memory=False)
    return None


def _get_field(row, *col_names, default=None):
    """Return first non-null value from a list of candidate column names."""
    for col in col_names:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def _parse_paper(row):
    """Map a raw dataset row to the form field dictionary."""
    # Open access: True if the 'Open Access' column has a non-null/non-empty value
    oa_val = _get_field(row, "Open Access", "open_access", "is_open_access")
    if isinstance(oa_val, (int, float, bool)):
        is_oa = bool(oa_val)
    elif isinstance(oa_val, str):
        is_oa = oa_val.strip().lower() not in ("", "nan", "none", "no", "0", "false")
    else:
        is_oa = False

    # Publication type: standardise to 'Article' or 'Review'
    raw_pubtype = str(_get_field(row, "Publication type", "Publication Type", default="Article"))
    if "review" in raw_pubtype.lower():
        pub_type = "Review"
    else:
        pub_type = "Article"

    # Source type: standardise to known options
    raw_srctype = str(_get_field(row, "Source type", "source_type", default="Journal"))
    src_map = {
        "journal": "Journal",
        "conference": "Conference Proceeding",
        "book series": "Book Series",
        "book": "Book",
    }
    src_type = "Journal"
    for key, val in src_map.items():
        if key in raw_srctype.lower():
            src_type = val
            break

    topic = _get_field(row, "Topic Prominence Percentile", "topic_prominence", default=50.0)
    try:
        topic = float(topic)
    except (TypeError, ValueError):
        topic = 50.0
    topic = max(0.0, min(100.0, topic))

    year = _get_field(row, "Year", "year", default=2020)
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = 2020

    actual_citations = _get_field(row, "Citations", "citations", "Times Cited Count", default=None)
    try:
        actual_citations = int(float(actual_citations)) if actual_citations is not None else None
    except (TypeError, ValueError):
        actual_citations = None

    return {
        "title": str(_get_field(row, "Title", "title", default="")),
        "abstract": str(_get_field(row, "Abstract", "abstract", default="")),
        "venue": str(_get_field(row, "Scopus Source title", "Source title", "venue", default="")),
        "authors": str(_get_field(row, "Authors", "authors", default="")),
        "h_indices": str(_get_field(row, "Authors H-index", "h_indices", default="")),
        "year": year,
        "is_open_access": is_oa,
        "topic_prominence": topic,
        "publication_type": pub_type,
        "source_type": src_type,
        "actual_citations": actual_citations,
    }


def main():
    st.set_page_config(
        page_title="Single Prediction - Citation Predictor",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Single Publication Prediction")
    st.markdown("Enter the details of a publication to predict its citation impact.")

    # Load predictor
    try:
        predictor, init_status = load_predictor()

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

    # ---------- Session state defaults ----------
    if "paper_defaults" not in st.session_state:
        st.session_state.paper_defaults = {}
    if "sampled_paper_actual" not in st.session_state:
        st.session_state.sampled_paper_actual = None

    defaults = st.session_state.paper_defaults

    # ---------- Random Paper button ----------
    dataset = load_merged_dataset()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        sample_clicked = st.button(
            "🎲 Random Paper from Dataset",
            use_container_width=True,
            help="Pick a random paper from the merged dataset and auto-fill the form"
        )

    if sample_clicked:
        if dataset is None:
            st.warning(
                "⚠️ Merged dataset not found. Run the data pipeline notebooks first "
                "(expected at `data/processed/merged.csv`)."
            )
        else:
            # Filter to papers with a non-empty abstract
            has_abstract = dataset["Abstract"].notna() if "Abstract" in dataset.columns else pd.Series([True] * len(dataset))
            pool = dataset[has_abstract]
            if len(pool) == 0:
                pool = dataset
            row = pool.sample(1, random_state=None).iloc[0]
            parsed = _parse_paper(row)
            st.session_state.paper_defaults = parsed
            st.session_state.sampled_paper_actual = parsed.get("actual_citations")
            defaults = parsed
            st.rerun()

    with col_info:
        if dataset is not None:
            st.caption(f"Dataset loaded: **{len(dataset):,} papers** available for sampling.")
        else:
            st.caption("Dataset not found — manual entry only.")

    st.markdown("---")

    # ---------- Prediction form ----------
    with st.form("prediction_form"):
        st.subheader("Publication Details")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input(
                "Title *",
                value=defaults.get("title", ""),
                placeholder="Enter the publication title",
                help="Full title of the research paper"
            )

            venue = st.text_input(
                "Venue/Journal *",
                value=defaults.get("venue", ""),
                placeholder="e.g., Nature, IEEE Transactions on...",
                help="Name of the journal or conference"
            )

            year = st.number_input(
                "Publication Year *",
                min_value=2000,
                max_value=2030,
                value=int(defaults.get("year", 2023)),
                help="Year of publication"
            )

        with col2:
            authors = st.text_area(
                "Authors *",
                value=defaults.get("authors", ""),
                placeholder="Last, First; Last, First; ...",
                help="Authors separated by semicolons",
                height=100
            )

            h_indices = st.text_input(
                "H-indices",
                value=defaults.get("h_indices", ""),
                placeholder="10;15;20;...",
                help="H-index values separated by semicolons (same order as authors)"
            )

        abstract = st.text_area(
            "Abstract *",
            value=defaults.get("abstract", ""),
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
                value=bool(defaults.get("is_open_access", False)),
                help="Check if the paper is published open access"
            )

            topic_prominence = st.slider(
                "Topic Prominence Percentile",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("topic_prominence", 50.0)),
                step=0.1,
                help="Scopus Topic Prominence Percentile (0-100). This is the #1 most important feature."
            )

        with col6:
            pub_type_options = ["Article", "Review"]
            default_pub = defaults.get("publication_type", "Article")
            pub_type_idx = pub_type_options.index(default_pub) if default_pub in pub_type_options else 0
            publication_type = st.selectbox(
                "Publication Type",
                options=pub_type_options,
                index=pub_type_idx,
                help="Select whether this is a research article or a review paper"
            )

            src_type_options = ["Journal", "Conference Proceeding", "Book", "Book Series"]
            default_src = defaults.get("source_type", "Journal")
            src_type_idx = src_type_options.index(default_src) if default_src in src_type_options else 0
            source_type = st.selectbox(
                "Source Type",
                options=src_type_options,
                index=src_type_idx,
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

        submit = st.form_submit_button("🔮 Predict Citation Impact", use_container_width=True)

    # ---------- Prediction results ----------
    if submit:
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

            st.success("✅ Predictions completed!")
            st.markdown("---")

            # ---- Side-by-side: Model prediction vs Actual (if sampled) ----
            actual_citations = st.session_state.sampled_paper_actual

            if actual_citations is not None:
                pred_col, actual_col = st.columns(2)
            else:
                pred_col = st.container()
                actual_col = None

            with pred_col:
                st.subheader("📊 Model Prediction")

                c1, c2 = st.columns(2)
                probability = result['high_impact_probability']
                citations = result['predicted_citations']

                with c1:
                    st.metric(
                        "High-Impact Prediction",
                        "Yes ✨" if result['is_high_impact'] else "No",
                        help="Top 25% citation prediction"
                    )
                    st.metric(
                        "High-Impact Probability",
                        f"{probability:.1%}",
                        help="Probability of being in top 25% most cited"
                    )

                with c2:
                    st.metric(
                        "Predicted Citation Count",
                        f"{citations:.0f}",
                        help="Estimated number of citations"
                    )
                    lower = max(0, citations * 0.7)
                    upper = citations * 1.3
                    st.metric(
                        "Estimated Range",
                        f"{lower:.0f} – {upper:.0f}",
                        help="Approximate citation range (±30%)"
                    )

                if result['is_high_impact']:
                    st.success(f"Predicted **high impact** — top 25% (confidence: {probability:.1%})")
                else:
                    st.info(f"Predicted **moderate impact** — ~{citations:.0f} citations (probability: {probability:.1%})")

            if actual_col is not None:
                with actual_col:
                    st.subheader("📋 Actual Status (from Dataset)")

                    # Compute high-impact threshold from dataset (top 25%)
                    if dataset is not None and "Citations" in dataset.columns:
                        threshold = dataset["Citations"].quantile(0.75)
                    else:
                        threshold = 26  # fallback from project documentation

                    actual_high_impact = actual_citations >= threshold

                    c3, c4 = st.columns(2)
                    with c3:
                        st.metric(
                            "Actually High-Impact",
                            "Yes ✨" if actual_high_impact else "No",
                            help=f"Top 25% threshold: ≥{threshold:.0f} citations"
                        )
                        st.metric(
                            "Citation Threshold (top 25%)",
                            f"≥ {threshold:.0f}",
                        )
                    with c4:
                        st.metric(
                            "Actual Citation Count",
                            f"{actual_citations:,}",
                        )
                        delta = actual_citations - citations
                        st.metric(
                            "Prediction Error",
                            f"{abs(delta):.0f} citations",
                            delta=f"{'over' if delta < 0 else 'under'} by {abs(delta):.0f}",
                            delta_color="off"
                        )

                    # Correct / wrong indicator
                    correct = result['is_high_impact'] == actual_high_impact
                    if correct:
                        st.success("✅ Classification **correct** — model and ground truth agree.")
                    else:
                        st.error("❌ Classification **wrong** — model and ground truth disagree.")

            # Input summary expander
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
