"""
Text feature extraction utilities for citation prediction.

This module provides functions to extract TF-IDF features from publication abstracts.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
from typing import Union, List


def preprocess_text(text: Union[str, float]) -> str:
    """
    Preprocess text for TF-IDF vectorization.

    Args:
        text: Input text (abstract)

    Returns:
        Preprocessed text (lowercase)
    """
    if pd.isna(text):
        return ""
    return str(text).lower()


def load_tfidf_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """
    Load a trained TF-IDF vectorizer from disk.

    Args:
        vectorizer_path: Path to the pickled vectorizer

    Returns:
        Trained TfidfVectorizer
    """
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


def extract_text_features(
    abstracts: Union[pd.Series, List[str]],
    vectorizer: TfidfVectorizer
) -> pd.DataFrame:
    """
    Extract TF-IDF features from abstracts using a trained vectorizer.

    Args:
        abstracts: Series or list of abstract texts
        vectorizer: Trained TfidfVectorizer

    Returns:
        DataFrame with TF-IDF features
    """
    # Preprocess texts
    if isinstance(abstracts, pd.Series):
        processed = abstracts.apply(preprocess_text)
    else:
        processed = [preprocess_text(text) for text in abstracts]

    # Transform using the trained vectorizer
    tfidf_matrix = vectorizer.transform(processed)

    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{feat}' for feat in feature_names]
    )

    return tfidf_df


def train_tfidf_vectorizer(
    abstracts: pd.Series,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 5,
    max_df: float = 0.8
) -> tuple:
    """
    Train a new TF-IDF vectorizer on abstracts.

    Args:
        abstracts: Series of abstract texts
        max_features: Maximum number of features to extract
        ngram_range: N-gram range (min, max)
        min_df: Minimum document frequency
        max_df: Maximum document frequency

    Returns:
        Tuple of (tfidf_df, vectorizer)
    """
    # Preprocess
    processed = abstracts.apply(preprocess_text)

    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(processed)

    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{feat}' for feat in feature_names],
        index=abstracts.index
    )

    return tfidf_df, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """
    Save a trained vectorizer to disk.

    Args:
        vectorizer: Trained TfidfVectorizer
        path: Output file path
    """
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
