"""Tests for data loading utilities."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_scopus_data, load_scival_data, load_processed_data


class TestLoadScopusData:
    """Tests for load_scopus_data."""

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            pd.DataFrame({"EID": ["1", "2"], "Title": ["A", "B"]}).to_csv(f.name, index=False)
            df = load_scopus_data(f.name)
            assert len(df) == 2
            assert "EID" in df.columns
        Path(f.name).unlink()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_scopus_data("nonexistent.csv")

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            with pytest.raises(ValueError, match="Unsupported"):
                load_scopus_data(f.name)
        Path(f.name).unlink()


class TestLoadProcessedData:
    """Tests for load_processed_data."""

    def test_load_pickle(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pd.DataFrame({"a": [1, 2]}).to_pickle(f.name)
            df = load_processed_data(f.name)
            assert len(df) == 2
        Path(f.name).unlink()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_processed_data("nonexistent.pkl")
