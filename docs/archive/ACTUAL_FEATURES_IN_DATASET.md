# Actual Features in Cleaned Dataset Used for Training

Based on the feature engineering notebooks (21, 22, 23), here are all features in `X_all.pkl`:

---

## Summary
- **Total Features:** 5,019
- **Venue Features:** 9
- **Author Features:** 10
- **Text Features (TF-IDF):** 5,000

---

## 1. VENUE FEATURES (9 features)
*Source: `notebooks/21_feature_engineering_venue.ipynb`*

From `data/features/venue_features.pkl`:

1. **snip** - SNIP (publication year) metric
2. **snip_percentile** - SNIP percentile (publication year)
3. **citescore** - CiteScore (publication year) metric
4. **citescore_percentile** - CiteScore percentile (publication year)
5. **sjr** - SJR (publication year) metric
6. **sjr_percentile** - SJR percentile (publication year)
7. **avg_venue_percentile** - Average of SNIP, CiteScore, SJR percentiles
8. **is_top_journal** - Binary (1 if any percentile ≥ 90, else 0)
9. **venue_score_composite** - Weighted composite: (SNIP×0.33 + CiteScore×0.33 + SJR×0.34)

**Missing values:** Imputed with median

---

## 2. AUTHOR FEATURES (10 features)
*Source: `notebooks/22_feature_engineering_author.ipynb`*

From `data/features/author_features.pkl`:

1. **num_authors** - Number of authors on the paper
2. **num_institutions** - Number of distinct institutions
3. **num_countries** - Number of distinct countries/regions
4. **is_single_author** - Binary (1 if single author, else 0)
5. **is_international_collab** - Binary (1 if >1 country, else 0)
6. **is_multi_institution** - Binary (1 if >1 institution, else 0)
7. **authors_per_institution** - Ratio: num_authors / num_institutions
8. **team_size_small** - Binary (1 if ≤3 authors, else 0)
9. **team_size_medium** - Binary (1 if 4-10 authors, else 0)
10. **team_size_large** - Binary (1 if >10 authors, else 0)

**Missing values:** Imputed with median

**Note:** No H-index features included (due to temporal data unavailability)

---

## 3. TEXT FEATURES (5,000 features)
*Source: `notebooks/20_feature_engineering_text.ipynb`*

From `data/features/text_features.pkl`:

**Method:** TF-IDF vectorization of abstracts

**Configuration:**
- Max features: 5,000
- N-gram range: (1, 2) - unigrams and bigrams
- Min document frequency: 5 documents
- Max document frequency: 0.8 (80% of documents)

**Feature names:** Auto-generated as `tfidf_<term>`

**Example features:**
- tfidf_methods
- tfidf_review
- tfidf_results
- tfidf_health
- tfidf_overall
- tfidf_study
- tfidf_factors
- tfidf_2015, tfidf_2016, tfidf_2017 (year mentions in abstracts)
- ... and 4,992 more TF-IDF features

**Note:** These are words/phrases from abstracts, NOT post-publication metrics

---

## Feature Combination
*Source: `notebooks/23_feature_engineering_final.ipynb`*

Final feature matrix `X_all.pkl` is created by:
```python
X = pd.concat([
    text_features,    # 5,000 features
    venue_features,   # 9 features
    author_features   # 10 features
], axis=1)
```

**Result:** 14,832 papers × 5,019 features

---

## Target Variables

### Classification Target
- **Variable:** `y_classification.pkl`
- **Definition:** Binary (1 = high-impact, 0 = low-impact)
- **Threshold:** Top 25% of citations (26 citations)
- **Distribution:**
  - High-impact: ~25% of papers
  - Low-impact: ~75% of papers

### Regression Target
- **Variable:** `y_regression_log.pkl`
- **Definition:** Log-transformed citation counts: log1p(Citations)
- **Original citations range:** 0 to 66,291
- **Mean citations:** 35.6
- **Median citations:** 10

---

## Temporal Split

### Training Set (`X_train_temporal.pkl`)
- **Years:** 2015-2017
- **Papers:** 2,545
- **Features:** 5,019
- **Mean citations:** 43.6

### Test Set (`X_test_temporal.pkl`)
- **Years:** 2018-2020
- **Papers:** 3,573
- **Features:** 5,019
- **Mean citations:** 40.1

**Note:** Test set has lower mean citations due to less accumulation time (more recent papers)

---

## Columns in `cleaned_data.pkl`

The original cleaned dataset contains these key columns:
- EID (unique identifier)
- Title
- Year
- Scopus Source title
- Abstract
- Number of Authors
- Number of Institutions
- Number of Countries/Regions
- Citations
- SNIP (publication year)
- SNIP percentile (publication year) *
- CiteScore (publication year)
- CiteScore percentile (publication year) *
- SJR (publication year)
- SJR percentile (publication year) *
- ... and other metadata columns

**Features extracted from cleaned_data:**
- Text features → from Abstract column
- Venue features → from SNIP, CiteScore, SJR columns
- Author features → from Number of Authors/Institutions/Countries columns
- Target → from Citations column

---

## Ex Ante Validation ✅

**All 5,019 features are observable at publication time:**
- ✅ Venue metrics are temporal (year-specific)
- ✅ Author features based on paper byline
- ✅ Text features from abstracts (written before publication)
- ✅ No H-index (would include future work)
- ✅ No post-publication views/downloads
- ✅ No citation-derived metrics

**Validation method:** Temporal train/test split (2015-2017 → 2018-2020)

---

## Files Created in Feature Engineering

All saved to `data/features/`:
- `text_features.pkl` (14,832 × 5,000)
- `venue_features.pkl` (14,832 × 9)
- `author_features.pkl` (14,832 × 10)
- `X_all.pkl` (14,832 × 5,019) - Combined feature matrix
- `y_classification.pkl` (14,832) - Binary target
- `y_regression.pkl` (14,832) - Citation counts
- `y_regression_log.pkl` (14,832) - Log-transformed citations
- `metadata.pkl` (14,832 × 4) - EID, Title, Year, Source
- `X_train_temporal.pkl` (2,545 × 5,019)
- `X_test_temporal.pkl` (3,573 × 5,019)
- `y_train_cls_temporal.pkl` (2,545)
- `y_test_cls_temporal.pkl` (3,573)
- `y_train_reg_temporal.pkl` (2,545)
- `y_test_reg_temporal.pkl` (3,573)
- `metadata_train.pkl` (2,545 × 4)
- `metadata_test.pkl` (3,573 × 4)

---

## Model Training

These features were used to train:

**Classification Models:**
- Logistic Regression → 79.01% ROC-AUC
- Random Forest
- XGBoost
- LightGBM

**Regression Models:**
- Random Forest → R²=34.74%
- XGBoost
- LightGBM → Spearman=58.10%

All models trained on `X_train_temporal` and evaluated on `X_test_temporal`.
