# Data Directory

This directory contains all data files for the citation prediction project.

## Structure

- **raw/**: Original, immutable data files from Scopus and SciVal
  - Store the original files here without modification
  - Files should include EID column for matching between datasets

- **interim/**: Intermediate data during processing
  - Temporary files during data transformation
  - Partially cleaned data

- **processed/**: Final cleaned and merged datasets
  - Ready for feature engineering
  - Includes merged Scopus + SciVal data with abstracts

- **features/**: Feature matrices ready for modeling
  - Engineered features saved as pickle or parquet files
  - Includes train/test splits

## Data Files

The project expects two main input files:
1. **Scopus file**: Contains abstracts and publication metadata
2. **SciVal file**: Contains citation counts and additional metrics

Both files must contain the **EID** column for matching.

## Usage

1. Place raw data files in `data/raw/`
2. Run data preprocessing scripts to generate processed data
3. Run feature engineering scripts to generate feature matrices
4. Use feature matrices for model training

## Notes

- Large data files (>.csv, .json, .parquet) are excluded from git via .gitignore
- Maintain data documentation in this directory
- Never commit raw data files to version control
