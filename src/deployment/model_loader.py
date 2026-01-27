"""
Model loading utilities for deployment.

This module provides functions to load trained models and preprocessing artifacts.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ModelLoader:
    """Handles loading of trained models and preprocessing artifacts."""

    def __init__(self, models_dir: str = "models", features_dir: str = "data/features"):
        """
        Initialize ModelLoader.

        Args:
            models_dir: Directory containing trained models
            features_dir: Directory containing feature preprocessing artifacts
        """
        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)
        self._models_cache = {}
        self._artifacts_cache = {}

    def load_classification_model(self, model_name: str) -> Any:
        """
        Load a classification model.

        Args:
            model_name: Name of the model (e.g., 'random_forest', 'xgboost')

        Returns:
            Loaded model object
        """
        cache_key = f"classification_{model_name}"

        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        model_path = self.models_dir / "classification" / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Classification model not found: {model_path}\n"
                f"Available models should be in: {self.models_dir / 'classification'}"
            )

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self._models_cache[cache_key] = model
        return model

    def load_regression_model(self, model_name: str) -> Any:
        """
        Load a regression model.

        Args:
            model_name: Name of the model (e.g., 'random_forest', 'xgboost')

        Returns:
            Loaded model object
        """
        cache_key = f"regression_{model_name}"

        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        model_path = self.models_dir / "regression" / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Regression model not found: {model_path}\n"
                f"Available models should be in: {self.models_dir / 'regression'}"
            )

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self._models_cache[cache_key] = model
        return model

    def load_tfidf_vectorizer(self) -> Any:
        """
        Load the trained TF-IDF vectorizer.

        Returns:
            Trained TfidfVectorizer
        """
        if 'tfidf_vectorizer' in self._artifacts_cache:
            return self._artifacts_cache['tfidf_vectorizer']

        vectorizer_path = self.features_dir / "tfidf_vectorizer.pkl"

        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found: {vectorizer_path}\n"
                f"Please ensure the vectorizer is saved during training."
            )

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        self._artifacts_cache['tfidf_vectorizer'] = vectorizer
        return vectorizer

    def load_venue_statistics(self) -> Optional[Dict]:
        """
        Load venue statistics computed during training.

        Returns:
            Dictionary of venue statistics or None if not found
        """
        if 'venue_stats' in self._artifacts_cache:
            return self._artifacts_cache['venue_stats']

        stats_path = self.features_dir / "venue_statistics.pkl"

        if not stats_path.exists():
            return None

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self._artifacts_cache['venue_stats'] = stats
        return stats

    def list_available_models(self) -> Dict[str, list]:
        """
        List all available trained models.

        Returns:
            Dictionary with 'classification' and 'regression' model lists
        """
        available = {
            'classification': [],
            'regression': []
        }

        # Check classification models
        cls_dir = self.models_dir / "classification"
        if cls_dir.exists():
            available['classification'] = [
                f.stem for f in cls_dir.glob("*.pkl")
            ]

        # Check regression models
        reg_dir = self.models_dir / "regression"
        if reg_dir.exists():
            available['regression'] = [
                f.stem for f in reg_dir.glob("*.pkl")
            ]

        return available

    def check_required_artifacts(self) -> Dict[str, bool]:
        """
        Check which required artifacts are available.

        Returns:
            Dictionary indicating availability of each artifact
        """
        return {
            'tfidf_vectorizer': (self.features_dir / "tfidf_vectorizer.pkl").exists(),
            'venue_statistics': (self.features_dir / "venue_statistics.pkl").exists(),
            'classification_models': len(self.list_available_models()['classification']) > 0,
            'regression_models': len(self.list_available_models()['regression']) > 0
        }
