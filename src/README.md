# Source Code Directory

This directory contains all reusable Python modules for the citation prediction project.

## Structure

### `data/`
Data loading, cleaning, and preprocessing modules
- `load_data.py`: Load Scopus and SciVal files
- `merge_data.py`: Merge datasets using EID column
- `clean_data.py`: Data cleaning and validation
- `preprocess.py`: Data preprocessing pipelines

### `features/`
Feature engineering and transformation modules
- `text_features.py`: TF-IDF and text feature extraction
- `author_features.py`: Author reputation metrics (h-index, citations)
- `venue_features.py`: Venue prestige and historical statistics
- `feature_engineering.py`: Main feature engineering pipeline
- `feature_selection.py`: Feature selection methods

### `models/`
Model training, evaluation, and prediction modules
- `base_model.py`: Base model class with common functionality
- `classification.py`: Classification models (Logistic Regression, RF, XGBoost, LightGBM)
- `regression.py`: Regression models for citation count prediction
- `train.py`: Training pipeline
- `evaluate.py`: Model evaluation metrics
- `predict.py`: Prediction interface
- `hyperparameter_tuning.py`: Hyperparameter optimization

### `visualization/`
Visualization and plotting modules
- `plots.py`: Common plotting functions
- `roc_curves.py`: ROC curve visualization
- `confusion_matrix.py`: Confusion matrix plots
- `feature_importance.py`: Feature importance visualizations
- `distribution_plots.py`: Citation distribution plots

### `utils/`
Utility functions and helpers
- `config.py`: Configuration loading and management
- `logger.py`: Logging setup
- `metrics.py`: Custom metric calculations
- `io.py`: Input/output utilities
- `validation.py`: Data validation functions

### `deployment/`
Deployment-related modules for Streamlit app
- `app_utils.py`: Utility functions for Streamlit app
- `model_loader.py`: Load trained models for deployment
- `prediction_service.py`: Prediction service for API

## Usage

### Import modules in notebooks:
```python
import sys
sys.path.append('../')

from src.data.load_data import load_scopus_data, load_scival_data
from src.features.text_features import extract_tfidf_features
from src.models.classification import train_classification_model
from src.visualization.plots import plot_feature_importance
```

### Import in other modules:
```python
from src.data.merge_data import merge_datasets
from src.features.feature_engineering import FeatureEngineer
from src.models.train import train_all_models
```

## Development Guidelines

1. **Modularity**: Keep functions focused and reusable
2. **Documentation**: Use docstrings for all functions and classes
3. **Type hints**: Add type hints for function parameters and returns
4. **Testing**: Write unit tests in `tests/` directory
5. **Configuration**: Use config files instead of hardcoding values
6. **Logging**: Use the logger module for debugging and tracking

## Code Quality

Maintain code quality with:
```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Run tests
pytest tests/
```
