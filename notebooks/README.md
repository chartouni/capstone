# Notebooks

Numbered by pipeline stage. Run in order within each group; groups are independent.

## 00–06 · Data Pipeline
| Notebook | Purpose |
|----------|---------|
| `00_paper_lookup.ipynb` | Initial paper lookup |
| `01_data_exploration.ipynb` | Explore raw Scopus/SciVal files |
| `02_data_merging.ipynb` | Merge on EID column |
| `03_data_cleaning.ipynb` | Clean and validate merged data |
| `04_peer_uni_data_check.ipynb` | Peer university data check |
| `05_peer_uni_merge.ipynb` | Merge peer university data |
| `06_all_unis_merge_clean.ipynb` | Final multi-university merge and clean |

## 10 · EDA
| Notebook | Purpose |
|----------|---------|
| `10_eda_citation_distribution.ipynb` | Citation count distributions and target definition |

## 20–24 · Feature Engineering
| Notebook | Purpose |
|----------|---------|
| `20_feature_engineering_text.ipynb` | Abstract TF-IDF features |
| `20b_feature_engineering_title.ipynb` | Title TF-IDF features |
| `21_feature_engineering_venue.ipynb` | Venue prestige features |
| `21b_venue_features_clean.ipynb` | Revised/clean venue feature pipeline |
| `22_feature_engineering_author.ipynb` | Author reputation features |
| `22b_feature_engineering_additional.ipynb` | Additional author/metadata features |
| `23_feature_engineering_final.ipynb` | Combine all features into final matrix |
| `24_rebuild_temporal_split.ipynb` | Rebuild train/test split with temporal cutoff |

## 30–36 · Model Development
| Notebook | Purpose |
|----------|---------|
| `30_classification_models.ipynb` | Binary classification (top-25% highly cited) |
| `31_regression_models.ipynb` | Citation count regression |
| `35_hyperparameter_tuning.ipynb` | RandomizedSearchCV tuning |
| `36_threshold_optimization.ipynb` | Decision threshold search → 0.54 |

## 37–42 · Experiments
| Notebook | Purpose |
|----------|---------|
| `37_smote_experiment.ipynb` | Exp 1: SMOTE oversampling |
| `38_f1_improvement_experiments.ipynb` | Exp 2–6: Feature selection, title, ensembles |
| `39_advanced_f1_experiments.ipynb` | Exp 7a–7f: Enhanced features, stacking, MLP |
| `40_extract_unused_features.ipynb` | Exp 8: Additional dataset features |
| `40b_feature_importance.ipynb` | Feature importance analysis |
| `41_error_analysis.ipynb` | Error analysis on misclassified papers |
| `41b_year_normalized_target.ipynb` | Exp 9: Year-normalised citation target |
| `42_domain_segmentation_experiment.ipynb` | Exp 10: Domain-specific models → **63.33% F1** |

## 50–53 · Unsupervised Learning
| Notebook | Purpose |
|----------|---------|
| `50_unsupervised_learning.ipynb` | Clustering and dimensionality reduction |
| `51_ul_feature_integration.ipynb` | Integrate UL features into pipeline |
| `52_add_ul_to_baseline.ipynb` | Test UL features against baseline |
| `53_eda_feature_opportunities.ipynb` | EDA for further feature opportunities |

## 98–99 · Data Quality
| Notebook | Purpose |
|----------|---------|
| `98_investigate_missing_papers.ipynb` | Investigate missing paper records |
| `99_data_quality_check.ipynb` | Final data quality validation |
