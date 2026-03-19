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

    authors_str = str(authors)

    # Try different separators
    if ';' in authors_str:
        return len([a for a in authors_str.split(';') if a.strip()])
    elif ',' in authors_str:
        # Be careful with commas - they might be in names
        # Count by assuming format: "Last, First; Last, First"
        return len([a for a in authors_str.split(';') if a.strip()])
    else:
        # Single author or unknown format
        return 1 if authors_str.strip() else 0


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
    Extract author-related features aligned with trained model expectations.

    Args:
        df: DataFrame with author information
        authors_col: Name of the authors column
        h_index_col: Name of the H-index column

    Returns:
        DataFrame with author features
    """
    features = pd.DataFrame(index=df.index)

    # Number of authors (model expects 'num_authors')
    if authors_col in df.columns:
        num_authors = df[authors_col].apply(parse_author_count)
    else:
        num_authors = pd.Series(1, index=df.index)
    features['num_authors'] = num_authors

    # Team size buckets (model expects binary flags)
    features['is_single_author'] = (num_authors == 1).astype(int)
    features['team_size_small'] = (num_authors <= 3).astype(int)
    features['team_size_large'] = (num_authors >= 8).astype(int)

    # Number of countries — use Scopus column if available, else default to 1
    country_cols = ['Number of Countries/Regions', 'num_countries', 'Number of Countries']
    num_countries = None
    for col in country_cols:
        if col in df.columns:
            num_countries = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
            break
    if num_countries is None:
        num_countries = pd.Series(1, index=df.index)
    features['num_countries'] = num_countries

    # Number of institutions — use Scopus column if available, else default to 1
    inst_cols = ['Number of Institutions', 'num_institutions', 'Affiliations']
    num_institutions = None
    for col in inst_cols:
        if col in df.columns:
            num_institutions = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
            break
    if num_institutions is None:
        num_institutions = pd.Series(1, index=df.index)
    features['num_institutions'] = num_institutions

    # International collaboration flag
    features['is_international_collab'] = (num_countries > 1).astype(int)

    # H-index features (kept for any model that uses them)
    if h_index_col in df.columns:
        h_indices = df[h_index_col].apply(parse_h_indices)
        features['h_index_max'] = h_indices.apply(lambda x: max(x) if x else 0)
        features['h_index_mean'] = h_indices.apply(lambda x: float(np.mean(x)) if x else 0.0)
        features['h_index_sum'] = h_indices.apply(lambda x: sum(x) if x else 0)
        features['h_index_min'] = h_indices.apply(lambda x: min(x) if x else 0)
    else:
        features['h_index_max'] = 0
        features['h_index_mean'] = 0.0
        features['h_index_sum'] = 0
        features['h_index_min'] = 0

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
