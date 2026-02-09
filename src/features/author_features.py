"""
Author feature extraction utilities for citation prediction.

This module provides functions to extract author-related features.
"""

import pandas as pd
import numpy as np
from typing import List, Union


def parse_author_count(authors: Union[str, float]) -> int:
    """
    Parse the number of authors from the authors string.

    Args:
        authors: Authors string (comma-separated or semicolon-separated)

    Returns:
        Number of authors
    """
    if pd.isna(authors):
        return 0

    authors_str = str(authors).strip()

    # Unambiguous separators first: " and ", " & ", ";"
    for sep in [' and ', ' & ', ';']:
        if sep in authors_str:
            return len([a for a in authors_str.split(sep) if a.strip()])

    # Comma: ambiguous - could be "Last, First" (1 author) or "A, B, C" (multiple)
    if ',' in authors_str:
        parts = [p.strip() for p in authors_str.split(',') if p.strip()]
        # Heuristic: exactly 2 parts often means "LastName, FirstName"
        if len(parts) == 2:
            return 1
        return len(parts) if parts else 0

    return 1 if authors_str else 0


def parse_h_indices(h_index_str: Union[str, float]) -> List[int]:
    """
    Parse H-index values from string.

    Args:
        h_index_str: String containing H-index values (e.g., "10;15;20")

    Returns:
        List of H-index values
    """
    if pd.isna(h_index_str):
        return []

    h_index_str = str(h_index_str)

    # Try different separators
    for separator in [';', ',', '|']:
        if separator in h_index_str:
            try:
                values = [int(float(x.strip())) for x in h_index_str.split(separator) if x.strip()]
                return values
            except (ValueError, AttributeError):
                continue

    # Try single value
    try:
        return [int(float(h_index_str.strip()))]
    except (ValueError, AttributeError):
        return []


def extract_author_features(
    df: pd.DataFrame,
    authors_col: str = 'Authors',
    h_index_col: str = 'Authors H-index'
) -> pd.DataFrame:
    """
    Extract author-related features.

    Args:
        df: DataFrame with author information
        authors_col: Name of the authors column
        h_index_col: Name of the H-index column

    Returns:
        DataFrame with author features
    """
    features = pd.DataFrame(index=df.index)

    # Number of authors
    if authors_col in df.columns:
        features['author_count'] = df[authors_col].apply(parse_author_count)
    else:
        features['author_count'] = 0

    # H-index features
    if h_index_col in df.columns:
        h_indices = df[h_index_col].apply(parse_h_indices)

        features['h_index_max'] = h_indices.apply(
            lambda x: max(x) if len(x) > 0 else 0
        )
        features['h_index_mean'] = h_indices.apply(
            lambda x: np.mean(x) if len(x) > 0 else 0
        )
        features['h_index_sum'] = h_indices.apply(
            lambda x: sum(x) if len(x) > 0 else 0
        )
        features['h_index_min'] = h_indices.apply(
            lambda x: min(x) if len(x) > 0 else 0
        )
    else:
        features['h_index_max'] = 0
        features['h_index_mean'] = 0
        features['h_index_sum'] = 0
        features['h_index_min'] = 0

    # Collaboration features
    features['is_single_author'] = (features['author_count'] == 1).astype(int)
    features['is_large_team'] = (features['author_count'] >= 5).astype(int)

    # Author reputation score (combination of H-index metrics)
    features['author_reputation_score'] = (
        0.5 * features['h_index_max'] +
        0.3 * features['h_index_mean'] +
        0.2 * (features['h_index_sum'] / (features['author_count'] + 1))
    )

    return features


def extract_single_paper_author_features(
    authors: str,
    h_indices: Union[str, List[int]]
) -> pd.DataFrame:
    """
    Extract author features for a single paper (useful for single predictions).

    Args:
        authors: Authors string
        h_indices: H-index string or list of values

    Returns:
        DataFrame with one row of author features
    """
    df = pd.DataFrame({
        'Authors': [authors],
        'Authors H-index': [h_indices]
    })

    return extract_author_features(df)
