"""
Data Quality Check for Cleaned Dataset
Checks for common issues: missing values, duplicates, outliers, inconsistencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("="*80)
print("CLEANED DATASET QUALITY CHECK")
print("="*80)

# Load cleaned data
df = pd.read_pickle('data/processed/cleaned_data.pkl')
print(f"\n✓ Loaded cleaned_data.pkl: {df.shape[0]} rows × {df.shape[1]} columns")

# Load feature files
X = pd.read_pickle('data/features/X_all.pkl')
y_class = pd.read_pickle('data/features/y_classification.pkl')
y_reg = pd.read_pickle('data/features/y_regression.pkl')

print("\n" + "="*80)
print("1. BASIC DATA INTEGRITY")
print("="*80)

# Check for duplicates
duplicates = df.duplicated(subset='EID')
print(f"\nDuplicate EIDs: {duplicates.sum()}")
if duplicates.sum() > 0:
    print(f"  ⚠️  WARNING: Found {duplicates.sum()} duplicate papers")
else:
    print(f"  ✓ No duplicates found")

# Check index alignment
print(f"\nIndex alignment:")
print(f"  df index: {len(df)}")
print(f"  X index: {len(X)}")
print(f"  y_class index: {len(y_class)}")
print(f"  y_reg index: {len(y_reg)}")

if len(df) == len(X) == len(y_class) == len(y_reg):
    print(f"  ✓ All indices aligned")
else:
    print(f"  ⚠️  WARNING: Index mismatch detected!")

print("\n" + "="*80)
print("2. MISSING VALUES")
print("="*80)

# Check missing values in key columns
key_columns = ['EID', 'Title', 'Year', 'Abstract', 'Citations',
               'Number of Authors', 'Number of Institutions', 'Number of Countries/Regions',
               'SNIP (publication year)', 'CiteScore (publication year)', 'SJR (publication year)']

print("\nMissing values in key columns:")
for col in key_columns:
    if col in df.columns:
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        if missing > 0:
            print(f"  ⚠️  {col}: {missing} ({missing_pct:.1f}%)")
        else:
            print(f"  ✓ {col}: 0 (0.0%)")

# Check missing values in features
print(f"\nMissing values in feature matrix X:")
missing_features = X.isnull().sum()
if missing_features.sum() > 0:
    print(f"  ⚠️  WARNING: {missing_features.sum()} missing values found")
    print(f"  Columns with missing values:")
    for col in missing_features[missing_features > 0].index:
        print(f"    - {col}: {missing_features[col]}")
else:
    print(f"  ✓ No missing values in feature matrix")

print("\n" + "="*80)
print("3. CITATION STATISTICS")
print("="*80)

print(f"\nCitation distribution:")
print(f"  Count: {len(df)}")
print(f"  Min: {df['Citations'].min()}")
print(f"  Max: {df['Citations'].max()}")
print(f"  Mean: {df['Citations'].mean():.2f}")
print(f"  Median: {df['Citations'].median():.0f}")
print(f"  Std: {df['Citations'].std():.2f}")

# Check for negative citations
negative_citations = (df['Citations'] < 0).sum()
if negative_citations > 0:
    print(f"  ⚠️  WARNING: {negative_citations} papers with negative citations!")
else:
    print(f"  ✓ No negative citations")

# Check for extreme outliers (>99.9 percentile)
p999 = df['Citations'].quantile(0.999)
extreme_outliers = (df['Citations'] > p999).sum()
print(f"\nExtreme outliers (>99.9 percentile = {p999:.0f} citations):")
print(f"  Count: {extreme_outliers}")
if extreme_outliers > 0:
    print(f"  Max outlier: {df['Citations'].max()}")

print("\n" + "="*80)
print("4. YEAR DISTRIBUTION")
print("="*80)

print(f"\nPapers by year:")
year_counts = df['Year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count} papers")

# Check for unexpected years
expected_years = [2015, 2016, 2017, 2018, 2019, 2020]
unexpected_years = df[~df['Year'].isin(expected_years)]
if len(unexpected_years) > 0:
    print(f"  ⚠️  WARNING: {len(unexpected_years)} papers with unexpected years:")
    print(f"    {unexpected_years['Year'].unique()}")
else:
    print(f"  ✓ All years within expected range (2015-2020)")

print("\n" + "="*80)
print("5. AUTHOR FEATURES")
print("="*80)

print(f"\nNumber of authors:")
print(f"  Min: {df['Number of Authors'].min()}")
print(f"  Max: {df['Number of Authors'].max()}")
print(f"  Mean: {df['Number of Authors'].mean():.2f}")
print(f"  Median: {df['Number of Authors'].median():.0f}")

# Check for zero authors
zero_authors = (df['Number of Authors'] == 0).sum()
if zero_authors > 0:
    print(f"  ⚠️  WARNING: {zero_authors} papers with 0 authors!")
else:
    print(f"  ✓ No papers with 0 authors")

print(f"\nNumber of institutions:")
print(f"  Min: {df['Number of Institutions'].min()}")
print(f"  Max: {df['Number of Institutions'].max()}")
print(f"  Mean: {df['Number of Institutions'].mean():.2f}")

print(f"\nNumber of countries:")
print(f"  Min: {df['Number of Countries/Regions'].min()}")
print(f"  Max: {df['Number of Countries/Regions'].max()}")
print(f"  Mean: {df['Number of Countries/Regions'].mean():.2f}")

# Check for logical inconsistencies
inconsistent = df[df['Number of Institutions'] > df['Number of Authors']]
if len(inconsistent) > 0:
    print(f"  ⚠️  WARNING: {len(inconsistent)} papers with more institutions than authors!")
else:
    print(f"  ✓ No papers with more institutions than authors")

print("\n" + "="*80)
print("6. VENUE FEATURES")
print("="*80)

venue_cols = ['SNIP (publication year)', 'CiteScore (publication year)', 'SJR (publication year)']

for col in venue_cols:
    if col in df.columns:
        print(f"\n{col}:")
        vals = pd.to_numeric(df[col], errors='coerce')
        valid = vals.notna().sum()
        print(f"  Non-null (numeric): {valid} ({valid/len(df)*100:.1f}%)")
        if valid > 0:
            print(f"  Min: {vals.min():.2f}")
            print(f"  Max: {vals.max():.2f}")
            print(f"  Mean: {vals.mean():.2f}")

            # Check for negative values
            negative = (vals < 0).sum()
            if negative > 0:
                print(f"  ⚠️  WARNING: {negative} negative values!")

print("\n" + "="*80)
print("7. TEXT FEATURES")
print("="*80)

# Check abstract lengths
df['abstract_length'] = df['Abstract'].fillna('').str.len()
print(f"\nAbstract lengths:")
print(f"  Min: {df['abstract_length'].min()}")
print(f"  Max: {df['abstract_length'].max()}")
print(f"  Mean: {df['abstract_length'].mean():.0f}")
print(f"  Median: {df['abstract_length'].median():.0f}")

empty_abstracts = (df['abstract_length'] == 0).sum()
if empty_abstracts > 0:
    print(f"  ⚠️  WARNING: {empty_abstracts} papers with empty abstracts!")
else:
    print(f"  ✓ No empty abstracts")

# Check for suspiciously short abstracts
short_abstracts = (df['abstract_length'] < 50).sum()
if short_abstracts > 0:
    print(f"  ℹ️  {short_abstracts} papers with very short abstracts (<50 chars)")

print("\n" + "="*80)
print("8. FEATURE MATRIX VALIDATION")
print("="*80)

# Check for infinite values
inf_values = np.isinf(X).sum().sum()
if inf_values > 0:
    print(f"  ⚠️  WARNING: {inf_values} infinite values in feature matrix!")
    inf_cols = X.columns[np.isinf(X).any()]
    print(f"  Columns with inf values: {list(inf_cols)}")
else:
    print(f"  ✓ No infinite values")

# Check for NaN values
nan_values = X.isnull().sum().sum()
if nan_values > 0:
    print(f"  ⚠️  WARNING: {nan_values} NaN values in feature matrix!")
else:
    print(f"  ✓ No NaN values")

# Check for constant features (zero variance)
constant_features = X.columns[X.std() == 0]
if len(constant_features) > 0:
    print(f"  ⚠️  WARNING: {len(constant_features)} constant features (zero variance):")
    for col in constant_features[:10]:
        print(f"    - {col}")
    if len(constant_features) > 10:
        print(f"    ... and {len(constant_features) - 10} more")
else:
    print(f"  ✓ No constant features")

print("\n" + "="*80)
print("9. TARGET VARIABLE VALIDATION")
print("="*80)

# Classification target
print(f"\nClassification target (y_classification):")
print(f"  High-impact (1): {y_class.sum()} ({y_class.mean()*100:.1f}%)")
print(f"  Low-impact (0): {(y_class == 0).sum()} ({(1-y_class.mean())*100:.1f}%)")

threshold = df['Citations'].quantile(0.75)
print(f"  Threshold: {threshold:.0f} citations (75th percentile)")

# Check if target makes sense
expected_high = (df['Citations'] >= threshold).sum()
if y_class.sum() != expected_high:
    print(f"  ⚠️  WARNING: Target mismatch! Expected {expected_high}, got {y_class.sum()}")
else:
    print(f"  ✓ Target aligned with 75th percentile threshold")

# Regression target
print(f"\nRegression target (y_regression):")
print(f"  Min: {y_reg.min():.2f}")
print(f"  Max: {y_reg.max():.2f}")
print(f"  Mean: {y_reg.mean():.2f}")
print(f"  Median: {y_reg.median():.2f}")

# Check if it matches original citations
if not y_reg.equals(df['Citations']):
    print(f"  ⚠️  WARNING: Regression target doesn't match Citations column!")
else:
    print(f"  ✓ Regression target matches Citations column")

print("\n" + "="*80)
print("10. TEMPORAL SPLIT VALIDATION")
print("="*80)

X_train = pd.read_pickle('data/features/X_train_temporal.pkl')
X_test = pd.read_pickle('data/features/X_test_temporal.pkl')

print(f"\nTemporal split:")
print(f"  Train (2015-2017): {len(X_train)} papers")
print(f"  Test (2018-2020): {len(X_test)} papers")
print(f"  Total: {len(X_train) + len(X_test)} papers")

# Check if train + test = total
if len(X_train) + len(X_test) == len(X):
    print(f"  ✓ Train + Test = Total")
else:
    print(f"  ⚠️  WARNING: Train + Test ≠ Total ({len(X_train)} + {len(X_test)} ≠ {len(X)})")

# Check for data leakage (train/test overlap)
train_indices = set(X_train.index)
test_indices = set(X_test.index)
overlap = train_indices & test_indices
if len(overlap) > 0:
    print(f"  ⚠️  WARNING: {len(overlap)} papers appear in both train and test!")
else:
    print(f"  ✓ No overlap between train and test sets")

print("\n" + "="*80)
print("QUALITY CHECK COMPLETE")
print("="*80)
