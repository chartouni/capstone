"""
Data merging utilities for citation prediction project.

This module provides functions to merge Scopus and SciVal datasets
using the EID column as the key.
"""

from typing import List, Optional

import pandas as pd

from src.utils.logger import logger


def merge_datasets(
    scopus_df: pd.DataFrame,
    scival_df: pd.DataFrame,
    on: str = 'EID',
    how: str = 'inner',
    validate: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge Scopus and SciVal datasets.

    Args:
        scopus_df: Scopus DataFrame
        scival_df: SciVal DataFrame
        on: Column name to merge on (default: 'EID')
        how: Type of merge ('inner', 'outer', 'left', 'right')
        validate: Validation mode for merge operation

    Returns:
        Merged DataFrame

    Raises:
        ValueError: If merge column not found in either DataFrame
    """
    # Validate merge column exists
    if on not in scopus_df.columns:
        raise ValueError(f"Column '{on}' not found in Scopus data")
    if on not in scival_df.columns:
        raise ValueError(f"Column '{on}' not found in SciVal data")

    logger.info(f"Merging datasets on column: {on}")
    logger.info(f"Scopus records: {len(scopus_df)}")
    logger.info(f"SciVal records: {len(scival_df)}")

    # Perform merge
    merged_df = pd.merge(
        scopus_df,
        scival_df,
        on=on,
        how=how,
        validate=validate,
        suffixes=('_scopus', '_scival')
    )

    logger.info(f"Merged records: {len(merged_df)}")

    # Log merge statistics
    if how == 'inner':
        lost_scopus = len(scopus_df) - len(merged_df)
        lost_scival = len(scival_df) - len(merged_df)
        logger.info(f"Records lost from Scopus: {lost_scopus}")
        logger.info(f"Records lost from SciVal: {lost_scival}")

    return merged_df


def add_abstracts_to_scival(
    scival_df: pd.DataFrame,
    scopus_df: pd.DataFrame,
    eid_column: str = 'EID',
    abstract_column: str = 'Abstract'
) -> pd.DataFrame:
    """
    Add abstracts from Scopus data to SciVal data.

    This is the main use case mentioned in the README:
    "add the abstracts from the Scopus file to the corresponding
    entries in the SciVal file"

    Args:
        scival_df: SciVal DataFrame
        scopus_df: Scopus DataFrame containing abstracts
        eid_column: Name of the EID column for matching
        abstract_column: Name of the abstract column in Scopus data

    Returns:
        SciVal DataFrame with abstracts added

    Raises:
        ValueError: If required columns not found
    """
    # Validate columns
    if eid_column not in scival_df.columns:
        raise ValueError(f"Column '{eid_column}' not found in SciVal data")
    if eid_column not in scopus_df.columns:
        raise ValueError(f"Column '{eid_column}' not found in Scopus data")
    if abstract_column not in scopus_df.columns:
        raise ValueError(f"Column '{abstract_column}' not found in Scopus data")

    logger.info("Adding abstracts from Scopus to SciVal data")

    # Extract only EID and abstract from Scopus
    scopus_abstracts = scopus_df[[eid_column, abstract_column]].copy()

    # Merge with SciVal data
    result_df = scival_df.merge(
        scopus_abstracts,
        on=eid_column,
        how='left'
    )

    # Count matches
    matched = result_df[abstract_column].notna().sum()
    total = len(result_df)

    logger.info(f"Matched abstracts: {matched}/{total} ({matched/total*100:.2f}%)")

    return result_df


def resolve_duplicate_columns(
    df: pd.DataFrame,
    priority: str = 'scopus',
    suffixes: tuple = ('_scopus', '_scival')
) -> pd.DataFrame:
    """
    Resolve duplicate columns after merge.

    When merging, some columns may exist in both datasets with different
    values. This function resolves conflicts by choosing values based on
    priority.

    Args:
        df: Merged DataFrame with duplicate columns
        priority: Which dataset to prioritize ('scopus' or 'scival')
        suffixes: Suffixes used in merge operation

    Returns:
        DataFrame with duplicate columns resolved
    """
    logger.info(f"Resolving duplicate columns with priority: {priority}")

    df_cleaned = df.copy()

    # Find columns with suffixes
    cols_with_suffixes = [col for col in df.columns if col.endswith(suffixes)]

    # Group by base column name
    base_cols = set()
    for col in cols_with_suffixes:
        for suffix in suffixes:
            if col.endswith(suffix):
                base_col = col[:-len(suffix)]
                base_cols.add(base_col)

    # Resolve each duplicate
    for base_col in base_cols:
        scopus_col = f"{base_col}{suffixes[0]}"
        scival_col = f"{base_col}{suffixes[1]}"

        if scopus_col in df.columns and scival_col in df.columns:
            # Choose column based on priority
            if priority == 'scopus':
                df_cleaned[base_col] = df_cleaned[scopus_col].fillna(
                    df_cleaned[scival_col]
                )
            else:
                df_cleaned[base_col] = df_cleaned[scival_col].fillna(
                    df_cleaned[scopus_col]
                )

            # Drop duplicate columns
            df_cleaned = df_cleaned.drop(columns=[scopus_col, scival_col])

            logger.info(f"Resolved duplicate column: {base_col}")

    return df_cleaned
