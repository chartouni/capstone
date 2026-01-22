"""
Data loading utilities for citation prediction project.

This module provides functions to load Scopus and SciVal data files.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.logger import logger


def load_scopus_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Scopus data file.

    Args:
        file_path: Path to Scopus data file (CSV or Excel)

    Returns:
        DataFrame containing Scopus data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Scopus file not found: {file_path}")

    logger.info(f"Loading Scopus data from: {file_path}")

    # Determine file format and load
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(df)} records from Scopus file")

    # Validate required columns
    if 'EID' not in df.columns:
        logger.warning("EID column not found in Scopus data")

    return df


def load_scival_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load SciVal data file.

    Args:
        file_path: Path to SciVal data file (CSV or Excel)

    Returns:
        DataFrame containing SciVal data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"SciVal file not found: {file_path}")

    logger.info(f"Loading SciVal data from: {file_path}")

    # Determine file format and load
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(df)} records from SciVal file")

    # Validate required columns
    if 'EID' not in df.columns:
        logger.warning("EID column not found in SciVal data")

    return df


def load_processed_data(
    file_path: Union[str, Path],
    file_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load processed data file.

    Args:
        file_path: Path to processed data file
        file_format: File format ('csv', 'parquet', 'pickle').
                    If None, inferred from file extension.

    Returns:
        DataFrame containing processed data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")

    # Infer format from extension if not provided
    if file_format is None:
        file_format = file_path.suffix.lower()[1:]  # Remove leading dot

    logger.info(f"Loading processed data from: {file_path}")

    # Load based on format
    if file_format == 'csv':
        df = pd.read_csv(file_path, low_memory=False)
    elif file_format == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_format in ['pkl', 'pickle']:
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    logger.info(f"Loaded {len(df)} records from processed file")

    return df
