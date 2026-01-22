# Notebooks Directory

This directory contains Jupyter notebooks for exploration, analysis, and experimentation.

## Notebook Organization

Notebooks should follow a numbered naming convention for sequential execution:

### 1. Data Exploration & Preprocessing
- `01_data_exploration.ipynb`: Initial data exploration of Scopus and SciVal files
- `02_data_merging.ipynb`: Merge datasets using EID column
- `03_data_cleaning.ipynb`: Clean and validate merged data

### 2. Exploratory Data Analysis
- `10_eda_citation_distribution.ipynb`: Analyze citation count distributions
- `11_eda_author_features.ipynb`: Explore author metrics (h-index, citations)
- `12_eda_venue_analysis.ipynb`: Analyze venue prestige and impact
- `13_eda_temporal_patterns.ipynb`: Explore temporal trends (2015-2020)

### 3. Feature Engineering
- `20_feature_engineering_text.ipynb`: TF-IDF and text feature extraction
- `21_feature_engineering_authors.ipynb`: Author reputation features
- `22_feature_engineering_venues.ipynb`: Venue prestige features
- `23_feature_engineering_final.ipynb`: Combine all features

### 4. Model Development
- `30_classification_models.ipynb`: Binary classification (top 25%)
- `31_regression_models.ipynb`: Citation count regression
- `32_model_comparison.ipynb`: Compare all models
- `33_hyperparameter_tuning.ipynb`: Optimize best models

### 5. Evaluation & Interpretation
- `40_model_evaluation.ipynb`: Comprehensive evaluation metrics
- `41_feature_importance.ipynb`: Analyze feature contributions
- `42_error_analysis.ipynb`: Investigate prediction errors
- `43_temporal_validation.ipynb`: Train on 2015-2017, test on 2018-2020

### 6. Visualization & Reporting
- `50_visualizations.ipynb`: Generate all charts and figures
- `51_final_results.ipynb`: Summary of findings

## Best Practices

1. **Clear naming**: Use descriptive names with number prefixes
2. **Documentation**: Add markdown cells explaining each section
3. **Reproducibility**: Set random seeds, use configuration files
4. **Save outputs**: Export figures to `reports/figures/`
5. **Version control**: Clear outputs before committing notebooks

## Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

## Notes

- Keep notebooks focused on single tasks
- Use `src/` modules for reusable code
- Save important plots and metrics
- Document findings and insights
