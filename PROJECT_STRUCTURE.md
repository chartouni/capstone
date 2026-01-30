# Citation Predictor - Project Structure

This document describes the complete architecture and organization of the citation prediction project.

## Directory Structure

```
capstone/
├── README.md                          # Project overview and objectives
├── PROJECT_STRUCTURE.md               # This file - architecture documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── config/                           # Configuration files
│   └── config.yaml                   # Main configuration (paths, hyperparameters, etc.)
│
├── data/                             # Data directory (files not tracked in git)
│   ├── README.md                     # Data directory documentation
│   ├── raw/                          # Original, immutable data
│   │   ├── scopus_data.*            # Scopus publications (with abstracts)
│   │   └── scival_data.*            # SciVal publications (with citations)
│   ├── interim/                      # Intermediate data during processing
│   ├── processed/                    # Final cleaned and merged datasets
│   │   └── merged_data.csv          # Merged Scopus + SciVal
│   └── features/                     # Feature matrices for modeling
│       └── features.pkl              # Engineered features
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── README.md                     # Notebook organization guide
│   ├── 01_data_exploration.ipynb    # Initial data exploration
│   ├── 02_data_merging.ipynb        # Merge Scopus and SciVal
│   ├── 03_data_cleaning.ipynb       # Data cleaning and validation
│   ├── 10_eda_*.ipynb               # Exploratory data analysis
│   ├── 20_feature_engineering_*.ipynb # Feature engineering
│   ├── 30_*_models.ipynb            # Model development
│   ├── 40_*_evaluation.ipynb        # Model evaluation
│   └── 50_visualizations.ipynb      # Final visualizations
│
├── src/                              # Source code (reusable modules)
│   ├── README.md                     # Source code documentation
│   ├── __init__.py
│   │
│   ├── data/                         # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── load_data.py             # Load Scopus/SciVal files
│   │   ├── merge_data.py            # Merge datasets using EID
│   │   ├── clean_data.py            # Data cleaning functions
│   │   └── preprocess.py            # Preprocessing pipelines
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── text_features.py         # TF-IDF and text features
│   │   ├── author_features.py       # Author metrics (h-index, etc.)
│   │   ├── venue_features.py        # Venue prestige features
│   │   ├── feature_engineering.py   # Main feature pipeline
│   │   └── feature_selection.py     # Feature selection methods
│   │
│   ├── models/                       # Machine learning models
│   │   ├── __init__.py
│   │   ├── base_model.py            # Base model class
│   │   ├── classification.py        # Classification models
│   │   ├── regression.py            # Regression models
│   │   ├── train.py                 # Training pipeline
│   │   ├── evaluate.py              # Evaluation metrics
│   │   ├── predict.py               # Prediction interface
│   │   └── hyperparameter_tuning.py # Hyperparameter optimization
│   │
│   ├── visualization/                # Visualization and plotting
│   │   ├── __init__.py
│   │   ├── plots.py                 # Common plotting functions
│   │   ├── roc_curves.py            # ROC curve visualization
│   │   ├── confusion_matrix.py      # Confusion matrix plots
│   │   ├── feature_importance.py    # Feature importance plots
│   │   └── distribution_plots.py    # Distribution visualizations
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration management
│   │   ├── logger.py                # Logging utilities
│   │   ├── metrics.py               # Custom metrics
│   │   ├── io.py                    # Input/output utilities
│   │   └── validation.py            # Data validation
│   │
│   └── deployment/                   # Deployment utilities
│       ├── __init__.py
│       ├── app_utils.py             # Streamlit app utilities
│       ├── model_loader.py          # Model loading for deployment
│       └── prediction_service.py    # Prediction API service
│
├── models/                           # Trained models (not tracked in git)
│   ├── classification/              # Classification models
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   └── lightgbm.pkl
│   └── regression/                  # Regression models
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       └── lightgbm.pkl
│
├── reports/                          # Analysis reports and results
│   ├── figures/                     # Generated figures and plots
│   │   ├── roc_curves/
│   │   ├── confusion_matrices/
│   │   ├── feature_importance/
│   │   └── distributions/
│   └── metrics/                     # Evaluation metrics
│       ├── classification_metrics.csv
│       └── regression_metrics.csv
│
├── app/                              # Streamlit deployment application
│   ├── README.md                     # App documentation
│   ├── main.py                       # Main Streamlit app
│   ├── pages/                        # Multi-page app structure
│   │   ├── 01_predict.py            # Single prediction page
│   │   └── 03_model_info.py         # Model performance page
│   ├── components/                   # Reusable UI components
│   └── styles/                       # Custom CSS styling
│
└── tests/                            # Unit tests
    ├── __init__.py
    ├── test_data/                    # Data processing tests
    ├── test_features/                # Feature engineering tests
    ├── test_models/                  # Model tests
    └── test_utils/                   # Utility tests
```

## Key Components

### 1. Data Pipeline

**Input**: Two large files from AUB
- **Scopus file**: Contains abstracts and publication metadata
- **SciVal file**: Contains citation counts and metrics
- **Key**: EID column for matching

**Process**:
1. Load both files (`src/data/load_data.py`)
2. Merge using EID column (`src/data/merge_data.py`)
3. Add abstracts from Scopus to SciVal entries
4. Clean and validate data (`src/data/clean_data.py`)
5. Save to `data/processed/`

### 2. Feature Engineering

**Text Features** (`src/features/text_features.py`):
- TF-IDF vectorization of abstracts
- N-gram features (unigrams and bigrams)
- Text statistics (length, keyword presence)

**Author Features** (`src/features/author_features.py`):
- H-index statistics (max, mean, sum)
- Citation count aggregations
- Number of authors
- Author reputation metrics

**Venue Features** (`src/features/venue_features.py`):
- Venue prestige scores
- Historical citation statistics per venue
- Top venue indicators

### 3. Model Development

**Classification** (Binary: Top 25% vs Rest):
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

**Regression** (Citation Count Prediction):
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

**Evaluation**:
- 5-fold cross-validation
- Stratified splits for classification
- Temporal validation: Train on 2015-2017, test on 2018-2020

### 4. Model Interpretation

**Feature Importance**:
- Tree-based importance scores
- SHAP values for model interpretability
- LIME for local explanations

**Visualizations**:
- ROC curves and confusion matrices
- Feature importance bar charts
- Prediction distribution plots
- Error analysis

### 5. Deployment

**Streamlit Application** (`app/`):
- Interactive web interface
- Single prediction interface
- Model performance dashboard
- Real-time prediction capability

## Workflow

### Phase 1: Data Preparation
1. Place data files in `data/raw/`
2. Run notebooks 01-03 for exploration, merging, and cleaning
3. Output: `data/processed/merged_data.csv`

### Phase 2: Feature Engineering
1. Run notebooks 20-23 for feature extraction
2. Generate text, author, and venue features
3. Output: `data/features/features.pkl`

### Phase 3: Model Training
1. Run notebooks 30-33 for model development
2. Train classification and regression models
3. Hyperparameter tuning
4. Output: Trained models in `models/`

### Phase 4: Evaluation
1. Run notebooks 40-43 for evaluation
2. Generate metrics and visualizations
3. Temporal validation
4. Output: Reports in `reports/`

### Phase 5: Deployment
1. Test app locally: `streamlit run app/main.py`
2. Deploy to production (Streamlit Cloud, AWS, etc.)
3. Enable continuous updates

## Configuration

All configuration is centralized in `config/config.yaml`:
- Data paths
- Feature engineering parameters
- Model hyperparameters
- Evaluation metrics
- Visualization settings
- Deployment configuration

## Development Guidelines

### Code Quality
- Use type hints for all functions
- Write docstrings (Google style)
- Follow PEP 8 style guide
- Keep functions focused and modular

### Testing
- Write unit tests for core functionality
- Run tests before committing: `pytest tests/`
- Maintain test coverage >80%

### Version Control
- Commit frequently with clear messages
- Don't commit large data files or trained models
- Use `.gitignore` for data and models
- Keep notebooks clean (clear outputs before commit)

### Documentation
- Update README files when adding features
- Document complex algorithms and decisions
- Keep configuration files well-commented
- Maintain this PROJECT_STRUCTURE.md

## Dependencies

Install all requirements:
```bash
pip install -r requirements.txt
```

Key libraries:
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost, lightgbm, torch
- **NLP**: nltk, spacy, transformers
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretation**: shap, lime
- **Deployment**: streamlit, fastapi
- **Development**: jupyter, pytest, black

## Getting Started

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd capstone
   pip install -r requirements.txt
   ```

2. **Add data files**:
   - Place Scopus and SciVal files in `data/raw/`

3. **Explore data**:
   ```bash
   jupyter lab
   # Open notebooks/01_data_exploration.ipynb
   ```

4. **Run pipeline**:
   - Follow notebooks sequentially (01 → 02 → ... → 50)
   - Use src/ modules for reusable code

5. **Deploy app**:
   ```bash
   streamlit run app/main.py
   ```

## Notes

- Large files (data, models) are excluded from git
- Use configuration files instead of hardcoding
- Keep notebooks focused on exploration
- Move reusable code to src/ modules
- Document decisions and findings

## Contact

For questions about this project structure or implementation, refer to the main README.md or contact the project maintainer.
