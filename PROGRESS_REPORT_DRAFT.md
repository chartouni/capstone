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
- Merged Scopus and SciVal datasets using EID matching, yielding 14,832 clean AUB publications (2010-2025), exceeding the proposed 10,000+ paper target
- Engineered 5,019 features: TF-IDF text features (5,000), Scopus-provided venue prestige metrics (9: SJR, CiteScore, SNIP with percentiles), and collaboration-based author features (10: team size, institutional diversity, international collaboration)
- Implemented temporal train/test split: 2,545 papers for training (2015-2017), 3,573 papers for testing (2018-2020)
- Pivoted from proposed H-index features to collaboration metrics due to temporal data availability constraints, avoiding potential data leakage

**Model Development & Training:**
- Trained classification models (Logistic Regression, Random Forest, XGBoost, LightGBM) achieving 79.01% ROC-AUC and 60.22% F1-score with Logistic Regression on test set
- Developed regression models achieving R²=34.74% (Random Forest) and Spearman correlation=58.10% (LightGBM) for citation count prediction
- Implemented 5-fold cross-validation showing consistent performance across models
- Completed feature importance analysis: Text features (50.5%) and venue prestige (43.7%) drive predictions, while author collaboration contributes 5.9%
- Conducted error analysis identifying model strengths (best for 11-25 citation papers) and limitations (struggles with highly-cited outliers >100 citations)
- Validated all 5,019 features as ex ante (observable at publication time): temporal venue metrics confirmed, no data leakage detected

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

**Extreme Outlier Challenge:**
Test set contains papers with up to 26,135 citations—4x higher than training set maximum (6,165 citations). This explains higher prediction errors for highly-cited outliers, as the model was not exposed to such extreme values during training. Temporal validation creates this natural challenge where breakthrough papers in later years exceed historical patterns.

**Model Selection and Scope:**
While deep learning was mentioned as a possibility in the proposal, gradient boosting methods (LightGBM, XGBoost) proved sufficient, achieving 79% ROC-AUC without the complexity and computational overhead of neural networks. This pragmatic choice allowed focus on feature engineering and ex ante validation rather than model architecture tuning.

---

## Significant Changes to Proposal (if any)
(50 words maximum)

Pivoted from H-index to collaboration features due to temporal data unavailability; used Scopus-provided venue metrics instead of manual curation (more scalable). Focused on gradient boosting methods (LightGBM, XGBoost) over deep learning, as traditional methods achieved strong performance (79% ROC-AUC). Eliminated post-publication metrics (views, citation-derived features) after identifying data leakage concerns, ensuring ex ante compliance.

---

## Tasks to be Completed
(100 words maximum)

**Next Two Weeks:**
- **Generate visualization artifacts**: Re-run notebooks 10, 40, 41 to export figures (citation distribution, feature importance, error analysis) to reports/figures/ directory for final report
- **Model deployment testing**: Test Streamlit app end-to-end with real AUB papers, verify predictions are working correctly
- **Final capstone report**: Write comprehensive report (15-20 pages) including introduction, literature review, methodology, results analysis (model performance, feature importance, error patterns), discussion of limitations, business recommendations for AUB research strategy, and deployment guide
- **Presentation preparation**: Create defense slides (12-15 slides) summarizing problem statement, approach, results, key findings, business impact, and future work
- **Report polish**: Proofread final report, ensure all figures are properly referenced, add citations, format references

---

## Notes for Company Representative Sign-Off

**Current Status:** Phase 5 (Model Development) completed. Phase 6 (Deployment & Documentation) in progress.

**Deliverables Completed:**
- Data pipeline (loading, merging, cleaning) ✓
- Feature engineering system (5,019 features) ✓
- Ex ante feature validation (zero data leakage confirmed) ✓
- Model training (4 classification + 3 regression models) ✓
- Feature importance analysis ✓
- Error analysis and model diagnostics ✓
- Visualization export code added to notebooks ✓
- Streamlit deployment application framework ✓

**Model Performance Achieved:**
- Classification: 79.01% ROC-AUC, 60.22% F1-score (Logistic Regression)
- Regression: R²=34.74%, Spearman=58.10% (Random Forest/LightGBM)
- High-impact threshold: 26 citations (top 25% of papers)

**Dataset Summary:**
- Total papers: 14,832 AUB publications (2010-2025)
- Training set: 2,545 papers (2015-2017, mean citations: 43.6)
- Test set: 3,573 papers (2018-2020, mean citations: 40.1)
- Overall median: 10 citations | Mean: 35.6 citations
- Citation range: 0-66,291 (highly right-skewed distribution)
- Note: Test set has lower citations due to less accumulation time, creating a more realistic and challenging prediction scenario

**Next Phase:**
- Generate and save visualization artifacts (8 figures for report)
- Test Streamlit deployment end-to-end
- Write final capstone report (15-20 pages)
- Create presentation slides (12-15 slides)
- Prepare for defense

**Key Insights from Analysis:**
- Venue prestige (SJR: 20.7% individual importance) and abstract content (50.5% aggregate) are strongest predictors
- Collaboration features (team size, institutional diversity) contribute 5.9% to predictions, validating their use as H-index alternative
- Model performs best on papers with 11-25 citations (MAE=0.41); struggles with highly-cited outliers (MAE=3.13 for 100+ citations)
- Extreme outlier challenge: Test set maximum (26,135 citations) is 4x training set maximum (6,165 citations)
- Classification: 69.6% accuracy, 305 false negatives (high-impact papers missed), 781 false positives
- Temporal validation validated: Test set has lower mean citations (40.1 vs 43.6) due to less accumulation time, proving model works on harder "newer paper" scenarios
- Ex ante compliance: All 5,019 features validated as observable at publication time—venue metrics are temporal, collaboration-based author features used instead of H-index to avoid data leakage

---

