#!/usr/bin/env python3
"""
Check if all required files for the Streamlit app are available.
"""

from pathlib import Path
from typing import Dict, List

def check_requirements() -> Dict[str, Dict[str, bool]]:
    """Check for required files and return status."""

    project_root = Path(__file__).parent

    # Required classification models
    classification_models = [
        'logistic_regression.pkl',
        'random_forest.pkl',
        'xgboost.pkl',
        'lightgbm.pkl'
    ]

    # Required regression models
    regression_models = [
        'random_forest.pkl',
        'xgboost.pkl',
        'lightgbm.pkl'
    ]

    # Feature artifacts
    feature_artifacts = {
        'tfidf_vectorizer.pkl': True,  # Required
        'venue_statistics.pkl': False   # Optional
    }

    # Performance metrics
    metrics_files = {
        'classification_results.csv': False,  # Optional
        'regression_results.csv': False        # Optional
    }

    results = {
        'classification_models': {},
        'regression_models': {},
        'feature_artifacts': {},
        'metrics': {}
    }

    # Check classification models
    cls_dir = project_root / 'models' / 'classification'
    for model in classification_models:
        results['classification_models'][model] = (cls_dir / model).exists()

    # Check regression models
    reg_dir = project_root / 'models' / 'regression'
    for model in regression_models:
        results['regression_models'][model] = (reg_dir / model).exists()

    # Check feature artifacts
    features_dir = project_root / 'data' / 'features'
    for artifact, required in feature_artifacts.items():
        results['feature_artifacts'][artifact] = {
            'exists': (features_dir / artifact).exists(),
            'required': required
        }

    # Check metrics
    metrics_dir = project_root / 'reports' / 'metrics'
    for metric_file, required in metrics_files.items():
        results['metrics'][metric_file] = {
            'exists': (metrics_dir / metric_file).exists(),
            'required': required
        }

    return results


def print_status(results: Dict):
    """Print a formatted status report."""

    print("=" * 70)
    print("STREAMLIT APP REQUIREMENTS CHECK")
    print("=" * 70)

    # Classification models
    print("\n📊 Classification Models (models/classification/):")
    all_cls_present = all(results['classification_models'].values())
    for model, exists in results['classification_models'].items():
        status = "✅" if exists else "❌"
        print(f"  {status} {model}")

    # Regression models
    print("\n📈 Regression Models (models/regression/):")
    all_reg_present = all(results['regression_models'].values())
    for model, exists in results['regression_models'].items():
        status = "✅" if exists else "❌"
        print(f"  {status} {model}")

    # Feature artifacts
    print("\n🔧 Feature Artifacts (data/features/):")
    required_artifacts_present = True
    for artifact, info in results['feature_artifacts'].items():
        status = "✅" if info['exists'] else "❌"
        req_label = "(REQUIRED)" if info['required'] else "(optional)"
        print(f"  {status} {artifact} {req_label}")
        if info['required'] and not info['exists']:
            required_artifacts_present = False

    # Metrics
    print("\n📋 Performance Metrics (reports/metrics/):")
    for metric_file, info in results['metrics'].items():
        status = "✅" if info['exists'] else "❌"
        req_label = "(optional)"
        print(f"  {status} {metric_file} {req_label}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    can_run = all_cls_present and all_reg_present and required_artifacts_present

    if can_run:
        print("✅ All required files are present!")
        print("   You can run the app with: streamlit run app/main.py")
    else:
        print("❌ Missing required files. The app will not work yet.")
        print("\nTo generate these files:")
        print("1. Ensure you have the raw data files:")
        print("   - data/raw/scopus.csv")
        print("   - data/raw/scival.csv")
        print("\n2. Run the notebooks in order:")
        print("   - notebooks/20_feature_engineering_text.ipynb")
        print("   - notebooks/21_feature_engineering_venue.ipynb")
        print("   - notebooks/22_feature_engineering_author.ipynb")
        print("   - notebooks/23_feature_engineering_final.ipynb")
        print("   - notebooks/30_classification_models.ipynb")
        print("   - notebooks/31_regression_models.ipynb")

    print("=" * 70)


if __name__ == "__main__":
    results = check_requirements()
    print_status(results)
