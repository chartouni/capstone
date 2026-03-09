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

        self._initialized = True
        return status

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
            meta['topic_prominence'] = df['topic_prominence'].fillna(df['topic_prominence'].median() if 'topic_prominence' in df.columns else 50.0)
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

        # Make predictions
        cls_pred, cls_prob = self.predict_classification(df, classification_model)
        reg_pred = self.predict_regression(df, regression_model)

        return {
            'is_high_impact': bool(cls_pred[0]),
            'high_impact_probability': float(cls_prob[0]),
            'predicted_citations': float(reg_pred[0]),
            'classification_model': classification_model,
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
        # Make predictions
        cls_pred, cls_prob = self.predict_classification(df, classification_model)
        reg_pred = self.predict_regression(df, regression_model)

        # Add predictions to dataframe
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
