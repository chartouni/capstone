"""Tests for author feature extraction."""

import pandas as pd
import pytest

from src.features.author_features import (
    parse_author_count,
    parse_h_indices,
    extract_author_features,
)


class TestParseAuthorCount:
    """Tests for parse_author_count."""

    def test_semicolon_separated(self):
        assert parse_author_count("Author1; Author2; Author3") == 3
        assert parse_author_count("A; B") == 2

    def test_and_separated(self):
        assert parse_author_count("Author1 and Author2") == 2
        assert parse_author_count("A & B") == 2

    def test_comma_last_first(self):
        # "LastName, FirstName" format - treat as 1 author
        assert parse_author_count("Smith, John") == 1

    def test_comma_multiple(self):
        # Multiple comma-separated authors
        assert parse_author_count("A, B, C, D") == 4

    def test_single_author(self):
        assert parse_author_count("Single Author") == 1
        assert parse_author_count("  ") == 0

    def test_nan_empty(self):
        assert parse_author_count(float("nan")) == 0
        assert parse_author_count("") == 0


class TestParseHIndices:
    """Tests for parse_h_indices."""

    def test_semicolon_separated(self):
        assert parse_h_indices("10; 15; 20") == [10, 15, 20]

    def test_comma_separated(self):
        assert parse_h_indices("5, 10, 15") == [5, 10, 15]

    def test_single_value(self):
        assert parse_h_indices("42") == [42]
        assert parse_h_indices(" 12 ") == [12]

    def test_nan_empty(self):
        assert parse_h_indices(float("nan")) == []
        assert parse_h_indices("") == []

    def test_invalid_returns_empty(self):
        assert parse_h_indices("not a number") == []


class TestExtractAuthorFeatures:
    """Tests for extract_author_features."""

    def test_basic_extraction(self):
        df = pd.DataFrame({
            "Authors": ["Author1; Author2; Author3"],
            "Authors H-index": ["10; 15; 20"],
        })
        features = extract_author_features(df)
        assert "author_count" in features.columns
        assert features["author_count"].iloc[0] == 3
        assert features["h_index_max"].iloc[0] == 20
        assert features["h_index_mean"].iloc[0] == 15.0

    def test_missing_columns_fills_zeros(self):
        df = pd.DataFrame({"Other": [1]})
        features = extract_author_features(df)
        assert features["author_count"].iloc[0] == 0
        assert features["h_index_max"].iloc[0] == 0
