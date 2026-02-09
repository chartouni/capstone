"""Tests for venue feature extraction."""

import pandas as pd
import pytest

from src.features.venue_features import (
    extract_venue_features,
    compute_venue_statistics,
)


class TestComputeVenueStatistics:
    """Tests for compute_venue_statistics."""

    def test_basic_computation(self):
        df = pd.DataFrame({
            "Scopus Source title": ["Journal A", "Journal A", "Journal B"],
            "Citations": [10, 20, 5],
        })
        stats = compute_venue_statistics(df)
        assert "paper_counts" in stats
        assert stats["paper_counts"]["Journal A"] == 2
        assert stats["paper_counts"]["Journal B"] == 1
        assert stats["avg_citations"]["Journal A"] == 15.0
        assert "global_avg_citations" in stats


class TestExtractVenueFeatures:
    """Tests for extract_venue_features."""

    def test_inference_mode_with_stats(self):
        venues = pd.Series(["Journal A", "Journal B", "Unknown"])
        venue_stats = {
            "paper_counts": {"Journal A": 10, "Journal B": 5},
            "avg_citations": {"Journal A": 20.0, "Journal B": 10.0},
            "global_avg_citations": 15.0,
            "prestige_scores": {"Journal A": 0.9, "Journal B": 0.5},
            "is_top_venue": {"Journal A": 1, "Journal B": 0},
        }
        features = extract_venue_features(venues, venue_stats=venue_stats, training_mode=False)
        assert "venue_paper_count" in features.columns
        assert features["venue_paper_count"].iloc[0] == 10
        assert features["venue_avg_citations"].iloc[0] == 20.0
        assert features["venue_paper_count"].iloc[2] == 0  # Unknown venue

    def test_training_mode_requires_stats_when_false(self):
        venues = pd.Series(["A", "B"])
        with pytest.raises(ValueError, match="venue_stats required"):
            extract_venue_features(venues, training_mode=False)

    def test_training_mode_with_df(self):
        df = pd.DataFrame({
            "Scopus Source title": ["J1", "J1", "J2"],
            "Citations": [10, 20, 5],
        })
        venues = df["Scopus Source title"]
        features = extract_venue_features(
            venues, training_mode=True, df=df, citation_col="Citations"
        )
        assert features["venue_paper_count"].iloc[0] == 2
        assert features["venue_avg_citations"].iloc[0] == 15.0
