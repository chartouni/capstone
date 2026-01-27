"""
Venue feature extraction utilities for citation prediction.

This module provides functions to extract venue-related features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def extract_venue_features(
    venues: pd.Series,
    venue_stats: Optional[Dict] = None,
    training_mode: bool = False
) -> pd.DataFrame:
    """
    Extract venue-related features.

    Args:
        venues: Series of venue names (journal/conference titles)
        venue_stats: Dictionary of pre-computed venue statistics (for inference)
        training_mode: If True, compute statistics from data; if False, use provided stats

    Returns:
        DataFrame with venue features
    """
    features = pd.DataFrame(index=venues.index)

    if not training_mode and venue_stats is None:
        # This should be called during training with the full dataset
        raise ValueError("venue_stats required when training_mode is False")

    if training_mode:
        # Compute venue statistics from the training data
        # This would typically be done in the training pipeline
        venue_counts = venues.value_counts()
        features['venue_paper_count'] = venues.map(venue_counts).fillna(0)

        # Placeholder for average citations per venue
        # In real training, this would use actual citation data
        features['venue_avg_citations'] = 0
        features['venue_prestige_score'] = 0
        features['is_top_venue'] = 0
    else:
        # Use pre-computed statistics for inference
        if venue_stats is None:
            # Default features if no statistics available
            features['venue_paper_count'] = 0
            features['venue_avg_citations'] = 0
            features['venue_prestige_score'] = 0
            features['is_top_venue'] = 0
        else:
            features['venue_paper_count'] = venues.map(
                venue_stats.get('paper_counts', {})
            ).fillna(0)
            features['venue_avg_citations'] = venues.map(
                venue_stats.get('avg_citations', {})
            ).fillna(venue_stats.get('global_avg_citations', 0))
            features['venue_prestige_score'] = venues.map(
                venue_stats.get('prestige_scores', {})
            ).fillna(0)
            features['is_top_venue'] = venues.map(
                venue_stats.get('is_top_venue', {})
            ).fillna(0).astype(int)

    return features


def compute_venue_statistics(
    df: pd.DataFrame,
    venue_col: str = 'Scopus Source title',
    citation_col: str = 'Citations',
    top_n: int = 50
) -> Dict:
    """
    Compute venue statistics from training data.

    Args:
        df: Training dataframe
        venue_col: Name of the venue column
        citation_col: Name of the citations column
        top_n: Number of top venues to identify

    Returns:
        Dictionary of venue statistics
    """
    venue_groups = df.groupby(venue_col)[citation_col]

    paper_counts = df[venue_col].value_counts().to_dict()
    avg_citations = venue_groups.mean().to_dict()
    global_avg_citations = df[citation_col].mean()

    # Prestige score: combination of paper count and average citations
    venue_df = pd.DataFrame({
        'paper_count': df[venue_col].value_counts(),
        'avg_citations': venue_groups.mean(),
        'median_citations': venue_groups.median(),
        'total_citations': venue_groups.sum()
    })

    # Normalize and combine
    venue_df['prestige_score'] = (
        0.3 * (venue_df['paper_count'] / venue_df['paper_count'].max()) +
        0.5 * (venue_df['avg_citations'] / venue_df['avg_citations'].max()) +
        0.2 * (venue_df['median_citations'] / venue_df['median_citations'].max())
    )

    prestige_scores = venue_df['prestige_score'].to_dict()

    # Identify top venues
    top_venues = venue_df.nlargest(top_n, 'prestige_score').index.tolist()
    is_top_venue = {venue: 1 if venue in top_venues else 0 for venue in paper_counts.keys()}

    return {
        'paper_counts': paper_counts,
        'avg_citations': avg_citations,
        'global_avg_citations': global_avg_citations,
        'prestige_scores': prestige_scores,
        'is_top_venue': is_top_venue,
        'top_venues': top_venues
    }
