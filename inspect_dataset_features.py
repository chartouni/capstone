"""
Inspect Features in Cleaned Dataset
Shows all columns and features used for model training
"""

import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("CLEANED DATASET FEATURE INSPECTION")
print("="*80)

# 1. CLEANED DATA
print("\n1. CLEANED DATA (data/processed/cleaned_data.pkl)")
print("-"*80)

try:
    df = pd.read_pickle('data/processed/cleaned_data.pkl')
    print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nAll columns in cleaned_data.pkl:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
except Exception as e:
    print(f"✗ Error: {e}")

# 2. ENGINEERED FEATURES
print("\n\n2. ENGINEERED FEATURES (data/features/X_all.pkl)")
print("-"*80)

try:
    X = pd.read_pickle('data/features/X_all.pkl')
    print(f"✓ Loaded: {X.shape[0]} rows × {X.shape[1]} features")

    # Categorize features
    venue_cols = [col for col in X.columns if any(x in str(col).lower() for x in ['snip', 'citescore', 'sjr', 'venue', 'journal', 'percentile', 'composite', 'top_journal'])]

    author_cols = [col for col in X.columns if any(x in str(col).lower() for x in ['author', 'institution', 'team', 'collab', 'countries', 'country', 'single_author', 'international'])]

    text_cols = [col for col in X.columns if str(col).startswith('tfidf_')]

    other_cols = [col for col in X.columns if col not in venue_cols and col not in author_cols and col not in text_cols]

    # Display breakdown
    print(f"\n📊 FEATURE BREAKDOWN:")
    print(f"  • Venue features: {len(venue_cols)}")
    print(f"  • Author features: {len(author_cols)}")
    print(f"  • Text features (TF-IDF): {len(text_cols)}")
    print(f"  • Other features: {len(other_cols)}")
    print(f"  • TOTAL: {len(X.columns)}")

    # Show venue features
    print(f"\n🏛️ VENUE FEATURES ({len(venue_cols)}):")
    for col in venue_cols:
        print(f"    {col}")

    # Show author features
    print(f"\n👥 AUTHOR FEATURES ({len(author_cols)}):")
    for col in author_cols:
        print(f"    {col}")

    # Show sample text features
    print(f"\n📝 TEXT FEATURES (TF-IDF): {len(text_cols)} total")
    print(f"    Sample (first 30):")
    for col in text_cols[:30]:
        print(f"      {col}")
    if len(text_cols) > 30:
        print(f"      ... and {len(text_cols) - 30} more TF-IDF features")

    # Show other features if any
    if other_cols:
        print(f"\n❓ OTHER FEATURES ({len(other_cols)}):")
        for col in other_cols:
            print(f"    {col}")

except Exception as e:
    print(f"✗ Error: {e}")

# 3. INDIVIDUAL FEATURE FILES
print("\n\n3. INDIVIDUAL FEATURE FILES")
print("-"*80)

feature_files = [
    ('data/features/venue_features.pkl', 'Venue Features'),
    ('data/features/author_features.pkl', 'Author Features'),
    ('data/features/text_features.pkl', 'Text Features (TF-IDF)'),
]

for filepath, name in feature_files:
    print(f"\n{name} ({filepath}):")
    try:
        features = pd.read_pickle(filepath)
        print(f"  ✓ Shape: {features.shape[0]} rows × {features.shape[1]} features")
        print(f"  Columns:")
        for col in features.columns:
            print(f"    • {col}")
    except FileNotFoundError:
        print(f"  ✗ File not found")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# 4. TEMPORAL SPLITS
print("\n\n4. TEMPORAL TRAIN/TEST SPLITS")
print("-"*80)

try:
    X_train = pd.read_pickle('data/features/X_train_temporal.pkl')
    X_test = pd.read_pickle('data/features/X_test_temporal.pkl')

    print(f"✓ Training set: {X_train.shape[0]} rows × {X_train.shape[1]} features")
    print(f"✓ Test set: {X_test.shape[0]} rows × {X_test.shape[1]} features")
    print(f"\nFeature columns match: {list(X_train.columns) == list(X_test.columns)}")

except Exception as e:
    print(f"✗ Error: {e}")

# 5. TARGET VARIABLES
print("\n\n5. TARGET VARIABLES")
print("-"*80)

try:
    y_class = pd.read_pickle('data/features/y_classification.pkl')
    y_reg = pd.read_pickle('data/features/y_regression.pkl')

    print(f"✓ Classification target: {len(y_class)} samples")
    print(f"  • High-impact papers: {y_class.sum()} ({y_class.mean()*100:.1f}%)")
    print(f"  • Low-impact papers: {(~y_class).sum()} ({(1-y_class.mean())*100:.1f}%)")

    print(f"\n✓ Regression target: {len(y_reg)} samples")
    print(f"  • Min: {y_reg.min():.2f}")
    print(f"  • Max: {y_reg.max():.2f}")
    print(f"  • Mean: {y_reg.mean():.2f}")
    print(f"  • Median: {y_reg.median():.2f}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
