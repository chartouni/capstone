# Citation Prediction Model - Feature Summary

**Document Version:** 1.0
**Date:** February 2026
**Total Features:** 5,027

---

## Executive Summary

This document provides a comprehensive overview of the features used in the citation prediction model. The model uses 5,027 features across four main categories to predict whether a scientific paper will achieve high citation impact (top 25% of papers). All features are observable at the time of publication, ensuring the model makes realistic predictions without data leakage.

**Key Findings:**
- Topic Prominence Percentile is the single most important feature
- Venue prestige metrics contribute 16-42% of predictive power
- Text features from abstracts account for 47-67% of model importance
- Model achieves 81.28% ROC-AUC for classification

---

## Table of Contents

1. Feature Categories Overview
2. Venue Prestige Features (9 features)
3. Author Collaboration Features (10 features)
4. Publication Metadata Features (8 features)
5. Text Features (5,000 features)
6. Feature Importance Rankings
7. Model Performance
8. Data Validation & Quality Assurance

---

## 1. Feature Categories Overview

| Category | Count | % of Total | Classification Impact | Regression Impact |
|----------|-------|------------|---------------------|-------------------|
| Text Features (TF-IDF) | 5,000 | 99.46% | 67.0% | 46.9% |
| Venue Prestige | 9 | 0.18% | 16.5% | 41.6% |
| Author Collaboration | 10 | 0.20% | 6.0% | 5.5% |
| Publication Metadata | 8 | 0.16% | 10.4% | 6.1% |
| **TOTAL** | **5,027** | **100%** | **100%** | **100%** |

**Key Insight:** Despite comprising only 0.34% of total features, venue and metadata features contribute 26.9% of classification power and 47.7% of regression power.

---

## 2. Venue Prestige Features (9 features)

These features measure the quality and impact of the publication venue (journal or conference) at the time of publication.

### 2.1 Core Metrics

| Feature Name | Description | Data Source |
|-------------|-------------|-------------|
| `snip` | Source Normalized Impact per Paper | Scopus (publication year) |
| `citescore` | CiteScore metric | Scopus (publication year) |
| `sjr` | SCImago Journal Rank | Scopus (publication year) |

### 2.2 Percentile Rankings

| Feature Name | Description | Range |
|-------------|-------------|-------|
| `snip_percentile` | SNIP percentile ranking | 0-100 |
| `citescore_percentile` | CiteScore percentile ranking | 0-100 |
| `sjr_percentile` | SJR percentile ranking | 0-100 |

### 2.3 Composite Metrics

| Feature Name | Formula | Purpose |
|-------------|---------|---------|
| `avg_venue_percentile` | (SNIP_percentile + CiteScore_percentile + SJR_percentile) / 3 | Average prestige |
| `venue_score_composite` | (SNIP × 0.33) + (CiteScore × 0.33) + (SJR × 0.34) | Weighted prestige |
| `is_top_journal` | 1 if any percentile ≥ 90, else 0 | Elite venue flag |

### 2.4 Data Quality
- **Source:** Scopus-provided metrics
- **Temporal Validity:** Metrics reflect journal performance AT publication year, not current
- **Missing Values:** Imputed with median values
- **Coverage:** 14,832 papers with complete venue data

---

## 3. Author Collaboration Features (10 features)

These features capture team composition and collaboration patterns visible from the paper byline.

### 3.1 Count Features

| Feature Name | Description | Typical Range |
|-------------|-------------|---------------|
| `num_authors` | Total number of authors | 1-50+ |
| `num_institutions` | Number of distinct institutions | 1-30+ |
| `num_countries` | Number of distinct countries/regions | 1-20+ |

### 3.2 Binary Collaboration Flags

| Feature Name | Condition | Purpose |
|-------------|-----------|---------|
| `is_single_author` | 1 if num_authors = 1 | Identify solo research |
| `is_international_collab` | 1 if num_countries > 1 | Cross-border collaboration |
| `is_multi_institution` | 1 if num_institutions > 1 | Inter-institutional work |

### 3.3 Team Size Categories

| Feature Name | Condition | Category |
|-------------|-----------|----------|
| `team_size_small` | 1 if ≤ 3 authors | Small team |
| `team_size_medium` | 1 if 4-10 authors | Medium team |
| `team_size_large` | 1 if > 10 authors | Large collaboration |

### 3.4 Derived Metrics

| Feature Name | Formula | Interpretation |
|-------------|---------|----------------|
| `authors_per_institution` | num_authors / num_institutions | Collaboration density |

### 3.5 Design Decisions
- **H-index NOT included:** Would cause data leakage (includes future work)
- **Affiliation-based only:** Uses information visible on paper byline
- **Missing values:** Imputed with median

---

## 4. Publication Metadata Features (8 features)

These features capture publication characteristics and Scopus-provided metadata.

### 4.1 Access and Impact

| Feature Name | Type | Description | Values |
|-------------|------|-------------|--------|
| `is_open_access` | Binary | Open access status | 0 or 1 |
| `topic_prominence` | Continuous | Topic Prominence Percentile | 0-100 |

**Note:** Topic Prominence is the #1 most important classification feature (importance: 92.0)

### 4.2 Publication Type

| Feature Name | Description | Typical % |
|-------------|-------------|-----------|
| `pubtype_Article` | Regular research article | ~85% |
| `pubtype_Review` | Review article | ~15% |

### 4.3 Source Type

| Feature Name | Description | Typical % |
|-------------|-------------|-----------|
| `sourcetype_Journal` | Published in journal | ~70% |
| `sourcetype_Conference Proceeding` | Published in conference | ~25% |
| `sourcetype_Book` | Published as book chapter | ~3% |
| `sourcetype_Book Series` | Published in book series | ~2% |

### 4.4 Impact of Metadata Features

Adding these 8 features provided substantial performance gains:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| F1 Score | 60.22% | 62.07% | +1.85 points |
| ROC-AUC | 79.01% | 81.28% | +2.27 points |

---

## 5. Text Features (5,000 features)

Text features are extracted from paper abstracts using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

### 5.1 TF-IDF Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max features | 5,000 | Balance coverage vs. dimensionality |
| N-gram range | (1, 2) | Capture unigrams and bigrams |
| Min document frequency | 5 | Exclude rare terms |
| Max document frequency | 0.8 | Exclude overly common terms |
| Vectorizer | scikit-learn TfidfVectorizer | Industry standard |

### 5.2 Feature Naming Convention

Features are named as `tfidf_<term>`, where `<term>` is the word or phrase:
- `tfidf_methods` → papers mentioning "methods"
- `tfidf_review` → papers mentioning "review"
- `tfidf_machine learning` → papers mentioning "machine learning" (bigram)

### 5.3 Top Important Text Features

Based on feature importance analysis, these text features are most predictive:

| Feature | Interpretation | Importance Score |
|---------|---------------|------------------|
| `tfidf_methods` | Mentions methodological approach | 21.0 |
| `tfidf_review` | Review-type papers | 19.0 |
| `tfidf_results` | Emphasizes results | 19.0 |
| `tfidf_health` | Healthcare/medical domain | 18.0 |
| `tfidf_overall` | Summary/overview language | 17.0 |
| `tfidf_study` | Research study | 16.0 |
| `tfidf_factors` | Factor analysis | 15.0 |
| `tfidf_2015` | Year mention (temporal signal) | 21.0 |
| `tfidf_2016` | Year mention (temporal signal) | 20.0 |
| `tfidf_2017` | Year mention (temporal signal) | 19.0 |

### 5.4 Important Clarification

**These are NOT post-publication metrics:**
- `tfidf_citations` → Papers that MENTION "citations" in their abstract (e.g., citation analysis papers)
- `tfidf_review` → Papers that MENTION "review" in their abstract (e.g., literature reviews)

All text features are derived from abstracts written BEFORE publication.

### 5.5 Data Processing Pipeline

1. **Extract abstracts** from Scopus export
2. **Clean text:** Remove special characters, lowercase
3. **Vectorize:** Apply TF-IDF with configured parameters
4. **Save vectorizer:** Store for deployment inference
5. **Result:** 14,832 papers × 5,000 features

---

## 6. Feature Importance Rankings

### 6.1 Classification Model (Logistic Regression)

**Top 15 Features:**

| Rank | Feature | Category | Importance Score |
|------|---------|----------|------------------|
| 1 | **topic_prominence** | Metadata | **92.0** |
| 2 | venue_score_composite | Venue | 40.0 |
| 3 | avg_venue_percentile | Venue | 29.0 |
| 4 | snip | Venue | 25.0 |
| 5 | citescore | Venue | 23.0 |
| 6 | sjr | Venue | 22.0 |
| 7 | num_institutions | Author | 22.0 |
| 8 | tfidf_2015 | Text | 21.0 |
| 9 | tfidf_results | Text | 19.0 |
| 10 | num_authors | Author | 19.0 |
| 11 | tfidf_2016 | Text | 19.0 |
| 12 | tfidf_review | Text | 19.0 |
| 13 | is_open_access | Metadata | 18.0 |
| 14 | tfidf_health | Text | 18.0 |
| 15 | tfidf_methods | Text | 17.0 |

**Key Finding:** Topic Prominence is 2.3× more important than the next feature.

### 6.2 Regression Model (Random Forest)

**Top 15 Features:**

| Rank | Feature | Category | Importance Score |
|------|---------|----------|------------------|
| 1 | **sjr** | Venue | **0.1974** |
| 2 | avg_venue_percentile | Venue | 0.0579 |
| 3 | topic_prominence | Metadata | 0.0579 |
| 4 | citescore | Venue | 0.0564 |
| 5 | venue_score_composite | Venue | 0.0520 |
| 6 | snip | Venue | 0.0256 |
| 7 | num_institutions | Author | 0.0208 |
| 8 | num_authors | Author | 0.0158 |
| 9 | snip_percentile | Venue | 0.0119 |
| 10 | num_countries | Author | 0.0095 |
| 11 | tfidf_methods | Text | 0.0089 |
| 12 | is_international_collab | Author | 0.0078 |
| 13 | sjr_percentile | Venue | 0.0075 |
| 14 | is_open_access | Metadata | 0.0071 |
| 15 | citescore_percentile | Venue | 0.0068 |

**Key Finding:** Venue prestige (SJR) dominates regression importance.

### 6.3 Aggregate Category Importance

| Category | Classification (%) | Regression (%) |
|----------|-------------------|----------------|
| Text features | 67.0% | 46.9% |
| Venue features | 16.5% | 41.6% |
| Author features | 6.0% | 5.5% |
| Metadata features | 10.4% | 6.1% |

**Insight:** Classification relies more on abstract content, while regression relies more on venue prestige.

---

## 7. Model Performance

### 7.1 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total papers | 14,832 |
| Training set (2015-2017) | 2,545 papers |
| Test set (2018-2020) | 3,573 papers |
| Total features | 5,027 |
| High-impact threshold | 26 citations (top 25%) |
| High-impact papers in test set | 1,068 (29.9%) |
| Low-impact papers in test set | 2,505 (70.1%) |

### 7.2 Classification Performance - Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **70.61%** | **50.53%** | **80.43%** | **62.23%** | **81.28%** |
| Random Forest | 72.49% | 53.02% | 69.85% | 60.29% | 78.50% |
| XGBoost | 66.00% | 46.11% | 81.55% | 58.87% | 79.80% |
| LightGBM | 70.11% | 50.00% | 76.03% | 60.32% | 80.10% |

**Best Overall Model:** Logistic Regression (highest ROC-AUC and F1 score)

### 7.3 Classification Performance - Detailed Confusion Matrices

#### Logistic Regression (Best Model)
```
                 Predicted
               Low    High
Actual  Low   1664    841
       High    209    859
```
- **True Negatives (TN):** 1,664 - Correctly identified low-impact papers
- **False Positives (FP):** 841 - Low-impact papers incorrectly predicted as high-impact
- **False Negatives (FN):** 209 - High-impact papers missed (19.6% miss rate)
- **True Positives (TP):** 859 - Correctly identified high-impact papers (80.4% catch rate)

#### Random Forest
```
                 Predicted
               Low    High
Actual  Low   1844    661
       High    322    746
```
- Higher precision (53.02%) but lower recall (69.85%) - misses more high-impact papers

#### XGBoost
```
                 Predicted
               Low    High
Actual  Low   1487   1018
       High    197    871
```
- Highest recall (81.55%) but lowest precision (46.11%) - many false positives

#### LightGBM
```
                 Predicted
               Low    High
Actual  Low   1693    812
       High    256    812
```
- Balanced performance with 50% precision and 76% recall

### 7.4 Regression Performance - Citation Count Prediction

| Model | R² Score | Spearman Correlation | Best For |
|-------|----------|---------------------|----------|
| Linear Regression | -180.85% | 28.18% | ❌ Poor fit (negative R²) |
| **Random Forest** | **37.42%** | 61.52% | General predictions |
| XGBoost | 36.81% | 61.83% | Balanced performance |
| **LightGBM** | **37.42%** | **61.94%** | ✅ Best ranking (Spearman) |

**Key Insight:** Tree-based models (Random Forest, XGBoost, LightGBM) significantly outperform linear regression. LightGBM achieves the best ranking correlation (61.94% Spearman), meaning it's best at predicting the relative order of citation counts.

### 7.5 Model Interpretation

**Classification Trade-offs:**
- **Logistic Regression** balances precision and recall well (F1=62.23%)
- **XGBoost** catches the most high-impact papers (81.55% recall) but with more false alarms
- **Random Forest** has highest accuracy (72.49%) but misses 30% of high-impact papers

**Regression Patterns (from scatter plots):**
- All models show strong correlation for low-to-medium citation papers (0-100 citations)
- Models struggle with extreme outliers (>1000 citations) due to log transformation
- Vertical bands at integer log values indicate discrete citation counts
- LightGBM shows tightest clustering around the ideal prediction line

### 7.6 Performance by Category

**Impact of removing feature categories (Classification ROC-AUC):**

| Features Used | ROC-AUC | Change |
|--------------|---------|--------|
| All features | 81.28% | baseline |
| Without metadata | 79.01% | -2.27% |
| Without venue | 76.5% | -4.78% |
| Without author | 80.8% | -0.48% |
| Text only | 75.3% | -5.98% |

**Key Finding:** Removing venue features causes the largest performance drop (-4.78%), showing that journal prestige is critical for citation prediction despite being only 9 features.

---

## 8. Data Validation & Quality Assurance

### 8.1 Ex Ante Compliance

All features are **observable at publication time:**

✅ **Venue metrics:** Temporal (journal's metrics IN publication year, based on prior years)
✅ **Author features:** Observable from paper byline at publication
✅ **Text features:** Derived from abstracts written before publication
✅ **Metadata:** Scopus-provided at publication time

❌ **Explicitly excluded to prevent data leakage:**
- Post-publication views/downloads
- Citation-derived metrics (Field Weighted Citation Impact, etc.)
- Current author H-index (includes future work)
- Any metrics that accumulate after publication

### 8.2 Temporal Validation

**Train/Test Split Strategy:**
- Training: Papers from 2015-2017
- Testing: Papers from 2018-2020
- No data from test years used in training
- Simulates real-world prediction scenario

### 8.3 Missing Value Handling

| Feature Type | Missing % | Imputation Method |
|-------------|-----------|-------------------|
| Venue features | ~5% | Median imputation |
| Author features | <1% | Median imputation |
| Text features | 0% | N/A (all papers have abstracts) |
| Metadata | <2% | Median/mode imputation |

### 8.4 Data Sources

- **Scopus Export:** Paper metadata, abstracts, author information
- **SciVal Export:** Citation counts, venue metrics (SJR, CiteScore, SNIP)
- **Merged Dataset:** 14,832 papers with complete feature coverage
- **Date Range:** 2015-2020 publications, citations counted through 2022

---

## 9. Feature Engineering Pipeline

### 9.1 Text Processing (`notebooks/20_feature_engineering_text.ipynb`)

```
Input: Abstract text
↓
Clean text (lowercase, remove special chars)
↓
TF-IDF vectorization (max_features=5000, ngram_range=(1,2))
↓
Output: 5,000 text features
↓
Save: data/features/text_features.pkl
Save: data/models/tfidf_vectorizer.pkl (for deployment)
```

### 9.2 Venue Processing (`notebooks/21_feature_engineering_venue.ipynb`)

```
Input: Scopus venue metrics (SJR, CiteScore, SNIP)
↓
Parse temporal metrics (publication year)
↓
Compute percentiles and composite scores
↓
Median imputation for missing values
↓
Output: 9 venue features
↓
Save: data/features/venue_features.pkl
```

### 9.3 Author Processing (`notebooks/22_feature_engineering_author.ipynb`)

```
Input: Author, institution, country counts
↓
Count authors, institutions, countries from metadata
↓
Compute collaboration flags and team size categories
↓
Median imputation for missing values
↓
Output: 10 author features
↓
Save: data/features/author_features.pkl
```

### 9.4 Final Combination (`notebooks/23_feature_engineering_final.ipynb`)

```
Load: text_features (5,000) + venue_features (9) + author_features (10)
↓
Concatenate horizontally
↓
Output: 14,832 × 5,019 feature matrix
↓
Save: data/features/X_all.pkl
Save: data/features/y_classification.pkl (target)
Save: data/features/y_regression_log.pkl (log-transformed target)
```

---

## 10. Deployment & Inference

### 10.1 For a New Paper at Publication Time

**Required Information:**
1. Abstract text
2. Journal name (to lookup SJR, CiteScore, SNIP for publication year)
3. Number of authors, institutions, countries
4. Open access status
5. Publication type (Article/Review)
6. Source type (Journal/Conference/Book)
7. Topic prominence percentile (from Scopus)

**Inference Pipeline:**
```
1. Extract abstract → TF-IDF transform using saved vectorizer → 5,000 features
2. Lookup journal metrics for publication year → 9 features
3. Count authors, institutions, countries from byline → 10 features
4. Extract metadata (OA, pub type, topic, source) → 8 features
5. Concatenate → 5,027-dimensional feature vector
6. Feed to trained model → Predict citation impact
```

### 10.2 Saved Model Artifacts

| File | Purpose |
|------|---------|
| `tfidf_vectorizer.pkl` | Transform new abstracts |
| `logistic_regression_model.pkl` | Classification predictions |
| `random_forest_regression_model.pkl` | Citation count predictions |
| `venue_lookup.csv` | Historical venue metrics by year |

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

1. **Venue data availability:** Requires Scopus/SciVal access for journal metrics
2. **Abstract quality:** Papers with missing/poor abstracts may predict poorly
3. **Temporal decay:** Model trained on 2015-2020 data may degrade over time
4. **Domain coverage:** Model performance may vary by academic field

### 11.2 Potential Enhancements

1. **Add title TF-IDF features:** Capture additional semantic signals
2. **Reference count:** Number of references cited (observable at publication)
3. **Author network features:** Co-authorship network centrality (if available)
4. **Field-specific models:** Train separate models for different research domains
5. **Model updates:** Retrain annually with new data

---

## 12. Contact & Documentation

**Project Repository:** `/home/user/capstone`

**Key Documentation Files:**
- `COMPLETE_FEATURE_LIST.md` - Full feature specifications
- `ACTUAL_FEATURES_IN_DATASET.md` - Implementation details
- `EX_POST_FEATURE_AUDIT.md` - Data leakage validation
- `PROJECT_STRUCTURE.md` - Repository organization

**Feature Engineering Notebooks:**
- `notebooks/20_feature_engineering_text.ipynb`
- `notebooks/21_feature_engineering_venue.ipynb`
- `notebooks/22_feature_engineering_author.ipynb`
- `notebooks/23_feature_engineering_final.ipynb`

**Model Training Notebooks:**
- `notebooks/30_model_classification.ipynb`
- `notebooks/31_model_regression.ipynb`
- `notebooks/32_threshold_optimization.ipynb`

---

## Appendix A: Feature Matrix Dimensions

| Component | Rows (Papers) | Columns (Features) | File Size |
|-----------|---------------|-------------------|-----------|
| Text features | 14,832 | 5,000 | ~450 MB |
| Venue features | 14,832 | 9 | ~1 MB |
| Author features | 14,832 | 10 | ~1 MB |
| Metadata features | 14,832 | 8 | ~1 MB |
| **Combined (X_all)** | **14,832** | **5,027** | **~452 MB** |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **TF-IDF** | Term Frequency-Inverse Document Frequency; weights words by importance |
| **SNIP** | Source Normalized Impact per Paper; citation metric normalized by field |
| **CiteScore** | Average citations per paper published in journal (3-year window) |
| **SJR** | SCImago Journal Rank; prestige metric using weighted citations |
| **Topic Prominence** | Scopus-calculated percentile of topic visibility/impact |
| **Ex Ante** | Information available before the event (at publication time) |
| **Ex Post** | Information available after the event (would cause data leakage) |
| **ROC-AUC** | Area Under Receiver Operating Characteristic curve; classification quality |
| **F1 Score** | Harmonic mean of precision and recall |
| **Spearman Correlation** | Rank-based correlation coefficient |

---

**END OF DOCUMENT**
