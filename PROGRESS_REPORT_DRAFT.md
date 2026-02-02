# MSBA Capstone Project - Bi-Weekly Progress Report

**Student Name:** Mario Chartouni
**Project Title:** CitaPred: Machine Learning-Based Citation Impact Predictor for AUB Research

---

## Elevator Pitch
(50 words maximum)

CitaPred predicts citation impact of academic papers using publication-time features (abstracts, venue prestige, author metrics) through machine learning. The system classifies papers as high-impact (top 25%) and predicts citation counts, enabling AUB to identify promising research early. A Streamlit web application provides interactive predictions with interpretable results.

---

## Tasks Done During the Previous Two Weeks
(200 words maximum - use bullets)

**Data Processing & Feature Engineering:**
- Merged Scopus and SciVal datasets using EID matching, yielding 14,832 clean AUB publications (2010-2025)
- Engineered 5,019 features: TF-IDF text features (5,000), venue prestige metrics (9), and author collaboration features (10)
- Implemented temporal train/test split: 2,545 papers for training (2015-2017), 3,573 papers for testing (2018-2020)

**Model Development & Training:**
- Trained classification models (Logistic Regression, Random Forest, XGBoost, LightGBM) achieving 79.01% ROC-AUC and 60.22% F1-score with Logistic Regression on test set
- Developed regression models achieving R²=34.74% (Random Forest) and Spearman correlation=58.10% (LightGBM) for citation count prediction
- Implemented 5-fold cross-validation showing consistent performance across models
- Completed feature importance analysis: Text features (50.5%) and venue prestige (43.7%) drive predictions, while author collaboration contributes 5.9%
- Conducted error analysis identifying model strengths (best for 11-25 citation papers) and limitations (struggles with highly-cited outliers >100 citations)

**Deployment Infrastructure:**
- Built complete Streamlit web application with multi-page interface (Home, Single Prediction, Model Performance, Feature Importance)
- Created model loader and prediction service for inference
- Developed paper lookup tool for testing with real AUB publications
- Added requirements checker script to validate deployment setup

**Code Quality & Documentation:**
- Fixed critical data leakage issues by removing post-publication metrics (views, field-weighted citation impact)
- Simplified project scope by removing batch prediction feature and Docker deployment
- Created comprehensive project documentation (README, PROJECT_STRUCTURE.md, data documentation)

---

## Difficulties and Challenges Encountered
(100 words maximum)

**Data Leakage Prevention:**
Identified and eliminated multiple sources of data leakage, including post-publication metrics (views, field-weighted citation impact) that accumulate after paper release. Implemented temporal validation to ensure models only use information available at publication time.

**Feature Engineering Complexity:**
Handling missing abstracts and author metrics required robust preprocessing. TF-IDF vectorization of 14,832 abstracts created a high-dimensional sparse matrix requiring careful memory management.

**Deployment Simplification:**
Initial scope included Docker containerization and batch prediction features. Reduced complexity to focus on core prediction functionality, prioritizing single-paper predictions through Streamlit interface for clearer demonstration value.

---

## Significant Changes to Proposal (if any)
(50 words maximum)

Removed batch prediction feature and Docker deployment to simplify capstone scope and focus on core prediction functionality. Eliminated post-publication metrics (views, citation-derived features) after identifying data leakage concerns, ensuring models use only publication-time information for ethical and practical validity.

---

## Tasks to be Completed
(100 words maximum)

**Next Two Weeks:**
- **Model deployment verification**: Ensure trained models are properly saved and integrated with Streamlit app for end-to-end predictions
- **Save visualization artifacts**: Export ROC curves, confusion matrices, and feature importance plots generated in notebooks 40-41 for final report
- **Ex ante feature validation**: Verify venue metrics are temporal (not current snapshots) to confirm no data leakage
- **Final capstone report**: Write comprehensive report including introduction, methodology, results analysis (model performance, feature importance, error patterns), business recommendations for AUB research strategy, and deployment guide
- **Presentation preparation**: Create defense slides summarizing problem, approach, results, and business impact

---

## Notes for Company Representative Sign-Off

**Current Status:** Phase 5 (Model Development) completed. Phase 6 (Deployment & Documentation) in progress.

**Deliverables Completed:**
- Data pipeline (loading, merging, cleaning) ✓
- Feature engineering system (5,019 features) ✓
- Model training (4 classification + 3 regression models) ✓
- Feature importance analysis ✓
- Error analysis and model diagnostics ✓
- Streamlit deployment application framework ✓

**Model Performance Achieved:**
- Classification: 79.01% ROC-AUC, 60.22% F1-score (Logistic Regression)
- Regression: R²=34.74%, Spearman=58.10% (Random Forest/LightGBM)
- High-impact threshold: 26 citations (top 25% of papers)

**Dataset Summary:**
- Total papers: 14,832 AUB publications (2010-2025)
- Training set: 2,545 papers (2015-2017)
- Test set: 3,573 papers (2018-2020)
- Median citations: 10 | Mean citations: 35.62

**Next Phase:**
- Deployment verification and testing
- Ex ante feature validation (venue metrics)
- Final capstone report writing
- Presentation/defense preparation

**Key Insights from Analysis:**
- Venue prestige (SJR: 20.7%) and abstract content (50.5%) are strongest predictors
- Model performs best on papers with 11-25 citations (MAE=0.41)
- Main limitation: Struggles with highly-cited outliers (>100 citations)
- Classification captures 69.6% of papers correctly; misses 305 high-impact papers (false negatives)

---

