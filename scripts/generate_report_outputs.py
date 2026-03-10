"""
Generate outputs needed for progress report.
Run this script and send the output to fill the progress report.
"""

import sys
from pathlib import Path
import pandas as pd
import pickle

print("=" * 70)
print("CAPSTONE PROJECT STATUS REPORT - DATA OUTPUTS")
print("=" * 70)

# 1. Data Statistics
print("\n1. DATA STATISTICS")
print("-" * 70)
try:
    data_path = Path('data/processed/cleaned_data.pkl')
    if data_path.exists():
        df = pd.read_pickle(data_path)
        print(f"✓ Total papers after cleaning: {len(df)}")
        if 'Year' in df.columns:
            print(f"✓ Publication years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
        if 'Citations' in df.columns:
            print(f"✓ Citation range: {df['Citations'].min():.0f} - {df['Citations'].max():.0f}")
            print(f"✓ Median citations: {df['Citations'].median():.0f}")
    else:
        print("✗ Cleaned data file not found")
except Exception as e:
    print(f"✗ Error loading data: {e}")

# 2. Feature Engineering
print("\n2. FEATURE ENGINEERING")
print("-" * 70)

feature_dir = Path('data/features')

# Text features
try:
    text_feat = pd.read_pickle(feature_dir / 'text_features.pkl')
    print(f"✓ Text features (TF-IDF): {text_feat.shape[1]} features")
except:
    print("✗ Text features not found")

# Venue features
try:
    venue_feat = pd.read_pickle(feature_dir / 'venue_features.pkl')
    print(f"✓ Venue features: {venue_feat.shape[1]} features")
    print(f"  Feature names: {list(venue_feat.columns)}")
except:
    print("✗ Venue features not found")

# Author features
try:
    author_feat = pd.read_pickle(feature_dir / 'author_features.pkl')
    print(f"✓ Author features: {author_feat.shape[1]} features")
    print(f"  Feature names: {list(author_feat.columns)}")
except:
    print("✗ Author features not found")

# Combined features
try:
    X_all = pd.read_pickle(feature_dir / 'X_all.pkl')
    print(f"✓ Total combined features: {X_all.shape[1]} features")
    print(f"✓ Total samples: {X_all.shape[0]} papers")
except:
    print("✗ Combined features not found")

# 3. Train/Test Split
print("\n3. TRAIN/TEST SPLIT (Temporal Validation)")
print("-" * 70)
try:
    X_train = pd.read_pickle(feature_dir / 'X_train_temporal.pkl')
    X_test = pd.read_pickle(feature_dir / 'X_test_temporal.pkl')
    y_train_cls = pd.read_pickle(feature_dir / 'y_train_cls_temporal.pkl')
    y_test_cls = pd.read_pickle(feature_dir / 'y_test_cls_temporal.pkl')

    print(f"✓ Training set: {X_train.shape[0]} papers (2015-2017)")
    print(f"  High-impact: {y_train_cls.sum()} papers ({y_train_cls.mean()*100:.1f}%)")
    print(f"✓ Test set: {X_test.shape[0]} papers (2018-2020)")
    print(f"  High-impact: {y_test_cls.sum()} papers ({y_test_cls.mean()*100:.1f}%)")
except:
    print("✗ Train/test splits not found")

# 4. Model Performance
print("\n4. MODEL PERFORMANCE")
print("-" * 70)

# Check if models exist
models_dir = Path('models')
if not models_dir.exists():
    print("✗ Models directory not found - models not yet trained")
    print("  STATUS: Model training pending")
else:
    # Classification models
    clf_dir = models_dir / 'classification'
    if clf_dir.exists():
        clf_models = list(clf_dir.glob('*.pkl'))
        print(f"✓ Classification models trained: {len(clf_models)}")
        for model_file in clf_models:
            print(f"  - {model_file.stem}")
    else:
        print("✗ Classification models not found")

    # Regression models
    reg_dir = models_dir / 'regression'
    if reg_dir.exists():
        reg_models = list(reg_dir.glob('*.pkl'))
        print(f"✓ Regression models trained: {len(reg_models)}")
        for model_file in reg_models:
            print(f"  - {model_file.stem}")
    else:
        print("✗ Regression models not found")

# Check for performance metrics
reports_dir = Path('reports/metrics')
if reports_dir.exists():
    if (reports_dir / 'classification_results.csv').exists():
        print("\n✓ Classification results available:")
        results = pd.read_csv(reports_dir / 'classification_results.csv', index_col=0)
        print(results.to_string())
    if (reports_dir / 'regression_results.csv').exists():
        print("\n✓ Regression results available:")
        results = pd.read_csv(reports_dir / 'regression_results.csv', index_col=0)
        print(results.to_string())
else:
    print("  No performance metrics saved yet")

# 5. Deployment Status
print("\n5. DEPLOYMENT STATUS")
print("-" * 70)

# Check Streamlit app
app_dir = Path('app')
if app_dir.exists():
    app_files = list(app_dir.glob('*.py')) + list(app_dir.glob('pages/*.py'))
    print(f"✓ Streamlit app files: {len(app_files)} files")
    for f in sorted(app_files)[:5]:  # Show first 5
        print(f"  - {f.relative_to(app_dir.parent)}")
else:
    print("✗ Streamlit app not found")

# 6. Project Completion Checklist
print("\n6. PROJECT COMPLETION CHECKLIST")
print("-" * 70)

checklist = {
    "Data collection": data_path.exists() if 'data_path' in locals() else False,
    "Data cleaning": data_path.exists() if 'data_path' in locals() else False,
    "Feature engineering": (feature_dir / 'X_all.pkl').exists(),
    "Model training": models_dir.exists() if 'models_dir' in locals() else False,
    "Model evaluation": reports_dir.exists() if 'reports_dir' in locals() else False,
    "Streamlit app": app_dir.exists() if 'app_dir' in locals() else False,
}

for task, completed in checklist.items():
    status = "✓ DONE" if completed else "⧖ IN PROGRESS / TODO"
    print(f"{status:20s} {task}")

print("\n" + "=" * 70)
print("END OF REPORT")
print("=" * 70)
