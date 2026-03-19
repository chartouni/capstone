"""
Prediction service for citation prediction.

This module provides a unified interface for making predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.text_features import extract_text_features
from src.features.venue_features import extract_venue_features
from src.features.author_features import extract_author_features
from src.deployment.model_loader import ModelLoader


class CitationPredictor:
    """Unified prediction service for citation prediction."""

    def __init__(self, models_dir: str = "models", features_dir: str = "data/features"):
        """
        Initialize the prediction service.

        Args:
            models_dir: Directory containing trained models
            features_dir: Directory containing feature preprocessing artifacts
        """
        self.model_loader = ModelLoader(models_dir, features_dir)
        self.tfidf_vectorizer = None
        self.venue_stats = None
        self.citation_threshold = None  # p75 threshold for regression-derived classification
        self._initialized = False

    def initialize(self) -> Dict[str, bool]:
        """
        Initialize the service by loading required artifacts.

        Returns:
            Dictionary indicating which artifacts were loaded successfully
        """
        status = {}

        # Load TF-IDF vectorizer
        try:
            self.tfidf_vectorizer = self.model_loader.load_tfidf_vectorizer()
            status['tfidf_vectorizer'] = True
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            status['tfidf_vectorizer'] = False

        # Load venue statistics (optional)
        try:
            self.venue_stats = self.model_loader.load_venue_statistics()
            status['venue_stats'] = self.venue_stats is not None
        except Exception as e:
            print(f"Warning: Could not load venue statistics: {e}")
            status['venue_stats'] = False

        # Load citation threshold for regression-derived classification (optional)
        try:
            self.citation_threshold = self.model_loader.load_citation_threshold()
            status['citation_threshold'] = self.citation_threshold is not None
        except Exception as e:
            print(f"Warning: Could not load citation threshold: {e}")
            status['citation_threshold'] = False

        self._initialized = True
        return status

    def _extract_scopus_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Scopus journal metrics from raw dataframe columns when available.

        At inference (single prediction) these columns won't exist, so the method
        returns neutral defaults: 50 for percentiles, 0 for raw scores.
        At batch/training time the real Scopus values are used.
        """
        features = pd.DataFrame(index=df.index)

        col_map = {
            'citescore': ['CiteScore', 'citescore'],
            'sjr':       ['SJR', 'sjr'],
            'snip':      ['SNIP', 'snip'],
            'citescore_pct': ['CiteScore Percentile', 'citescore_pct', 'Percentile (CiteScore)'],
            'sjr_pct':   ['SJR Percentile', 'sjr_pct', 'Percentile (SJR)'],
            'snip_pct':  ['SNIP Percentile', 'snip_pct', 'Percentile (SNIP)'],
        }

        found = {}
        for feat, candidates in col_map.items():
            for col in candidates:
                if col in df.columns:
                    found[feat] = pd.to_numeric(df[col], errors='coerce')
                    break

        # Raw scores: 0 when not available
        for feat in ['citescore', 'sjr', 'snip']:
            features[feat] = found.get(feat, pd.Series(0.0, index=df.index)).fillna(0.0)

        # Percentiles: 50 (neutral midpoint) when not available
        for feat in ['citescore_pct', 'sjr_pct', 'snip_pct']:
            features[feat] = found.get(feat, pd.Series(50.0, index=df.index)).fillna(50.0)

        features['avg_venue_percentile'] = features[['citescore_pct', 'sjr_pct', 'snip_pct']].mean(axis=1)
        features['venue_score_composite'] = (
            0.4 * features['citescore_pct'] +
            0.3 * features['sjr_pct'] +
            0.3 * features['snip_pct']
        ) / 100.0

        return features

    def prepare_features(
        self,
        df: pd.DataFrame,
        abstract_col: str = 'Abstract',
        venue_col: str = 'Scopus Source title',
        authors_col: str = 'Authors',
        h_index_col: str = 'Authors H-index'
    ) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            df: DataFrame with publication metadata
            abstract_col: Name of abstract column
            venue_col: Name of venue column
            authors_col: Name of authors column
            h_index_col: Name of H-index column

        Returns:
            DataFrame with all features ready for prediction
        """
        if not self._initialized:
            self.initialize()

        features_list = []

        # Extract text features
        if abstract_col in df.columns and self.tfidf_vectorizer is not None:
            text_features = extract_text_features(df[abstract_col], self.tfidf_vectorizer)
            features_list.append(text_features)
        else:
            print("Warning: Text features not available")

        # Extract venue features
        if venue_col in df.columns:
            venue_features = extract_venue_features(
                df[venue_col],
                venue_stats=self.venue_stats,
                training_mode=False
            )
            features_list.append(venue_features)
        else:
            print("Warning: Venue features not available")

        # Extract author features
        author_features = extract_author_features(df, authors_col, h_index_col)
        features_list.append(author_features)

        # Extract metadata features (8 features critical for model performance)
        metadata_features = self._extract_metadata_features(df)
        features_list.append(metadata_features)

        # Extract Scopus journal metrics (real values in batch/training, neutral defaults in single prediction)
        scopus_features = self._extract_scopus_metrics(df)
        features_list.append(scopus_features)

        # Combine all features
        if len(features_list) == 0:
            raise ValueError("No features could be extracted")

        X = pd.concat(features_list, axis=1)
        return X

    def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 8 metadata features used in the final model.

        Features:
            - is_open_access: Open access status (binary)
            - topic_prominence: Scopus topic prominence percentile (0-100)
            - pubtype_Article: Publication type is Article (binary)
            - pubtype_Review: Publication type is Review (binary)
            - sourcetype_Journal: Source type is Journal (binary)
            - sourcetype_Conference Proceeding: Source type is Conference (binary)
            - sourcetype_Book: Source type is Book (binary)
            - sourcetype_Book Series: Source type is Book Series (binary)

        Args:
            df: DataFrame with publication metadata

        Returns:
            DataFrame with 8 metadata features
        """
        idx = df.index
        meta = pd.DataFrame(index=idx)

        # Open access status
        if 'is_open_access' in df.columns:
            meta['is_open_access'] = df['is_open_access'].fillna(0).astype(int)
        else:
            meta['is_open_access'] = 0

        # Topic prominence percentile (Scopus-provided, 0-100)
        if 'topic_prominence' in df.columns:
            tp = pd.to_numeric(df['topic_prominence'], errors='coerce')
            meta['topic_prominence'] = tp.fillna(tp.median() if tp.notna().any() else 50.0)
        else:
            meta['topic_prominence'] = 50.0  # Default: median percentile

        # Publication type (one-hot)
        pub_type = df.get('Publication Type', pd.Series(['Article'] * len(df), index=idx))
        meta['pubtype_Article'] = (pub_type == 'Article').astype(int)
        meta['pubtype_Review'] = (pub_type == 'Review').astype(int)

        # Source type (one-hot)
        src_type = df.get('Source type', pd.Series(['Journal'] * len(df), index=idx))
        meta['sourcetype_Journal'] = (src_type == 'Journal').astype(int)
        meta['sourcetype_Conference Proceeding'] = (src_type == 'Conference Proceeding').astype(int)
        meta['sourcetype_Book'] = (src_type == 'Book').astype(int)
        meta['sourcetype_Book Series'] = (src_type == 'Book Series').astype(int)

        return meta

    @staticmethod
    def _get_model_feature_names(model) -> Optional[List[str]]:
        """Return the feature names the model was trained on, or None if unavailable."""
        # XGBoost
        try:
            names = model.get_booster().feature_names
            if names:
                return names
        except Exception:
            pass
        # sklearn / LightGBM
        for attr in ('feature_names_in_', 'feature_name_'):
            names = getattr(model, attr, None)
            if names is not None:
                return list(names)
        return None

    @staticmethod
    def _align_features(X: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
        """
        Reindex X to match the model's expected feature names.

        Fill strategy for missing columns:
        - Features ending in '_percentile' or '_pct': fill with 50 (neutral midpoint)
        - All other missing features: fill with 0
        """
        if feature_names is None:
            return X
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            print(f"Info: filling {len(missing)} missing feature(s): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        X = X.reindex(columns=feature_names, fill_value=0)
        # Override fill for percentile features — 0 means "worst venue", 50 is neutral
        for col in missing:
            if col.endswith('_percentile') or col.endswith('_pct'):
                X[col] = 50.0
        return X

    def _classify_from_regression(self, predicted_citations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive binary high-impact labels and soft probabilities from regression output.

        Uses a sigmoid centred on the p75 citation threshold so that:
          - predicted == threshold  → 50 % probability
          - predicted == 2×threshold → ~95 % probability
          - predicted == 0           → ~5 % probability
        """
        thr = self.citation_threshold
        labels = (predicted_citations >= thr).astype(int)
        distance = (predicted_citations - thr) / max(thr, 1.0)
        probabilities = 1.0 / (1.0 + np.exp(-3.0 * distance))
        return labels, probabilities

    def predict_classification(
        self,
        df: pd.DataFrame,
        model_name: str = 'lightgbm'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict high-impact classification (top 25% citations).

        Args:
            df: DataFrame with publication metadata
            model_name: Name of the model to use

        Returns:
            Tuple of (predictions, probabilities)
        """
        # Prepare features
        X = self.prepare_features(df)

        # Load model
        model = self.model_loader.load_classification_model(model_name)

        # Align feature columns to what the model was trained on
        X = self._align_features(X, self._get_model_feature_names(model))

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def predict_regression(
        self,
        df: pd.DataFrame,
        model_name: str = 'lightgbm'
    ) -> np.ndarray:
        """
        Predict citation counts (log-transformed).

        Args:
            df: DataFrame with publication metadata
            model_name: Name of the model to use

        Returns:
            Array of predicted log citation counts
        """
        # Prepare features
        X = self.prepare_features(df)

        # Load model
        model = self.model_loader.load_regression_model(model_name)

        # Align feature columns to what the model was trained on
        X = self._align_features(X, self._get_model_feature_names(model))

        # Make predictions
        log_predictions = model.predict(X)

        # Transform back to original scale
        predictions = np.expm1(log_predictions)

        return predictions

    def predict_single(
        self,
        title: str,
        abstract: str,
        venue: str,
        authors: str,
        h_indices: str,
        year: int,
        is_open_access: bool = False,
        topic_prominence: float = 50.0,
        publication_type: str = 'Article',
        source_type: str = 'Journal',
        classification_model: str = 'lightgbm',
        regression_model: str = 'lightgbm'
    ) -> Dict:
        """
        Make predictions for a single publication.

        Args:
            title: Publication title
            abstract: Publication abstract
            venue: Venue/journal name
            authors: Authors string
            h_indices: H-index values
            year: Publication year
            is_open_access: Whether the paper is open access
            topic_prominence: Scopus topic prominence percentile (0-100)
            publication_type: Publication type ('Article' or 'Review')
            source_type: Source type ('Journal', 'Conference Proceeding', 'Book', 'Book Series')
            classification_model: Model for classification
            regression_model: Model for regression

        Returns:
            Dictionary with prediction results
        """
        # Create DataFrame
        df = pd.DataFrame([{
            'Title': title,
            'Abstract': abstract,
            'Scopus Source title': venue,
            'Authors': authors,
            'Authors H-index': h_indices,
            'Year': year,
            'is_open_access': int(is_open_access),
            'topic_prominence': topic_prominence,
            'Publication Type': publication_type,
            'Source type': source_type
        }])

        # Regression prediction (always computed)
        reg_pred = self.predict_regression(df, regression_model)

        # Derive classification from regression when threshold is available;
        # fall back to the standalone classifier otherwise.
        if self.citation_threshold is not None:
            cls_pred, cls_prob = self._classify_from_regression(reg_pred)
        else:
            cls_pred, cls_prob = self.predict_classification(df, classification_model)

        return {
            'is_high_impact': bool(cls_pred[0]),
            'high_impact_probability': float(cls_prob[0]),
            'predicted_citations': float(reg_pred[0]),
            'classification_model': 'regression-derived' if self.citation_threshold is not None else classification_model,
            'regression_model': regression_model
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        classification_model: str = 'lightgbm',
        regression_model: str = 'lightgbm'
    ) -> pd.DataFrame:
        """
        Make predictions for multiple publications.

        Args:
            df: DataFrame with publication metadata
            classification_model: Model for classification
            regression_model: Model for regression

        Returns:
            DataFrame with original data plus predictions
        """
        reg_pred = self.predict_regression(df, regression_model)

        if self.citation_threshold is not None:
            cls_pred, cls_prob = self._classify_from_regression(reg_pred)
        else:
            cls_pred, cls_prob = self.predict_classification(df, classification_model)

        result = df.copy()
        result['is_high_impact'] = cls_pred
        result['high_impact_probability'] = cls_prob
        result['predicted_citations'] = reg_pred

        return result

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models.

        Returns:
            Dictionary with available classification and regression models
        """
        return self.model_loader.list_available_models()
