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
- Merged Scopus and SciVal datasets using EID matching for clean publication records
- Engineered multi-modal features: TF-IDF text features from abstracts, venue prestige metrics (SNIP, SJR, CiteScore with percentiles), and author collaboration features (team size, institutions, international collaboration)
- Implemented temporal train/test split (2015-2017 training, 2018-2020 testing) to prevent data leakage

**Model Development:**
- Built classification models (Logistic Regression, Random Forest, XGBoost, LightGBM) for high-impact paper identification
- Developed regression models for citation count prediction with log-transformation
- Implemented 5-fold cross-validation for robust performance evaluation

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
- **Complete model training**: Execute classification and regression model notebooks (30_classification_models.ipynb, 31_regression_models.ipynb)
- **Save trained models**: Export trained models to models/ directory in .pkl format for deployment
- **Performance evaluation**: Generate ROC curves, confusion matrices, feature importance visualizations using notebooks 40 and 41
- **Error analysis**: Identify systematic prediction errors, analyze which paper types are misclassified
- **Finalize Streamlit app**: Connect trained models to deployment interface, test end-to-end prediction workflow with real AUB papers
- **Documentation**: Prepare final report sections on methodology, results, feature importance insights, and business recommendations for AUB

---

## Notes for Company Representative Sign-Off

**Current Status:** Phase 5 (Model Development) nearing completion. Infrastructure for deployment (Phase 6) is ready.

**Deliverables Ready:**
- Data pipeline (loading, merging, cleaning) ✓
- Feature engineering system ✓
- Streamlit deployment application framework ✓

**In Progress:**
- Model training and evaluation
- Feature importance analysis
- Final performance metrics

**Dataset:** [RUN generate_report_outputs.py TO GET ACTUAL NUMBERS]

---

