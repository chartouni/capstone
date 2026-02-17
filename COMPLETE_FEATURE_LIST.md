# Complete Feature List for Citation Prediction Model

## Feature Summary
- **Total Features:** 5,027
- **Text Features:** 5,000 (TF-IDF)
- **Venue Features:** 9
- **Author Features:** 10
- **Additional Metadata Features:** 8

---

## 1. Venue Prestige Features (9 features)

1. **snip** - Source Normalized Impact per Paper (publication year)
2. **snip_percentile** - SNIP percentile ranking (publication year)
3. **citescore** - CiteScore metric (publication year)
4. **citescore_percentile** - CiteScore percentile ranking (publication year)
5. **sjr** - SCImago Journal Rank (publication year)
6. **sjr_percentile** - SJR percentile ranking (publication year)
7. **avg_venue_percentile** - Average of SNIP, CiteScore, and SJR percentiles
8. **is_top_journal** - Binary flag (1 if any percentile ≥ 90th, else 0)
9. **venue_score_composite** - Weighted composite: (SNIP × 0.33) + (CiteScore × 0.33) + (SJR × 0.34)

**Source:** Scopus-provided metrics at publication year (temporal, not current)

---

## 2. Author Collaboration Features (10 features)

1. **num_authors** - Total number of authors on the paper
2. **num_institutions** - Number of distinct institutions represented
3. **num_countries** - Number of distinct countries/regions represented
4. **is_single_author** - Binary flag (1 if single author, else 0)
5. **is_international_collab** - Binary flag (1 if >1 country, else 0)
6. **is_multi_institution** - Binary flag (1 if >1 institution, else 0)
7. **authors_per_institution** - Ratio: num_authors / num_institutions
8. **team_size_small** - Binary flag (1 if ≤3 authors, else 0)
9. **team_size_medium** - Binary flag (1 if 4-10 authors, else 0)
10. **team_size_large** - Binary flag (1 if >10 authors, else 0)

**Note:** H-index features were initially proposed but replaced with collaboration metrics due to temporal data unavailability and to avoid data leakage.

---

## 3. Additional Metadata Features (8 features)

1. **is_open_access** - Binary flag (1 if open access, 0 otherwise)
2. **topic_prominence** - Topic Prominence Percentile from Scopus
3. **pubtype_Article** - Binary flag for Article publication type
4. **pubtype_Review** - Binary flag for Review publication type
5. **sourcetype_Journal** - Binary flag for Journal source type
6. **sourcetype_Conference Proceeding** - Binary flag for Conference source type
7. **sourcetype_Book** - Binary flag for Book source type
8. **sourcetype_Book Series** - Binary flag for Book Series source type

**Source:** Scopus-provided metadata at publication time

**Impact:** Adding these 8 features improved F1 score from 60.22% to 62.07% (+1.85 points) and ROC-AUC from 79.01% to 81.28% (+2.27 points)

---

## 4. Text Features (5,000 features)

**Method:** TF-IDF (Term Frequency-Inverse Document Frequency) vectorization of paper abstracts

**Configuration:**
- **Vectorizer:** scikit-learn TfidfVectorizer
- **Max features:** 5,000
- **N-gram range:** (1, 2) - unigrams and bigrams
- **Min document frequency:** 5 (term must appear in at least 5 papers)
- **Max document frequency:** 0.8 (exclude terms appearing in >80% of papers)
- **Feature names:** Auto-generated (e.g., tfidf_00, tfidf_000, tfidf_0001, etc.)

**Top Important Text Features (from feature importance analysis):**
- tfidf_methods
- tfidf_review
- tfidf_results
- tfidf_health
- tfidf_overall
- tfidf_study
- tfidf_factors
- tfidf_2015, tfidf_2016, tfidf_2017 (year mentions in abstracts)

**Note:** These are WORDS from abstracts (written before publication), not post-publication metrics. Features like "tfidf_citations" and "tfidf_review" represent papers that mention "citations" or "review" in their abstracts, not actual citation counts or review metrics.

---

## Feature Validation: Ex Ante Compliance

All 5,019 features are **observable at publication time**:

**Venue metrics:** Temporal (journal's metrics IN publication year, based on prior years)
**Author features:** Observable from paper byline at publication
**Text features:** Derived from abstracts written before publication

**Excluded to prevent data leakage:**
- Post-publication views/downloads
- Citation-derived metrics (Field Weighted Citation Impact, etc.)
- Current author H-index (includes future work)
- Any metrics that accumulate after publication

---

## Feature Importance Rankings

### Classification (LightGBM):
**Top 10 Features by Importance:**
1. avg_venue_percentile (35.0)
2. venue_score_composite (35.0)
3. sjr (33.0)
4. citescore (30.0)
5. snip (26.0)
6. num_authors (24.0)
7. authors_per_institution (21.0)
8. num_institutions (20.0)
9. tfidf_methods (18.0)
10. tfidf_review (17.0)

### Regression (Random Forest):
**Top 10 Features by Importance:**
1. sjr (0.2070)
2. avg_venue_percentile (0.0623)
3. citescore (0.0594)
4. venue_score_composite (0.0561)
5. snip (0.0252)
6. num_institutions (0.0212)
7. num_authors (0.0181)
8. snip_percentile (0.0131)
9. num_countries (0.0098)
10. sjr_percentile (0.0088)

### Aggregate Importance by Category:
- **Text features (5,000):** 50.5% (regression), 72.9% (classification)
- **Venue features (9):** 43.7% (regression), 18.9% (classification)
- **Author features (10):** 5.9% (regression), 8.1% (classification)
- **Additional metadata features (8):** [Run notebook 40 to update with latest importance scores]

---

## Data Sources

- **Scopus Export:** Paper metadata, abstracts, author information
- **SciVal Export:** Citation counts, venue metrics (SJR, CiteScore, SNIP)
- **Merged Dataset:** 14,832 papers with complete feature coverage

---

## Feature Engineering Pipeline

1. **Text Processing:**
   - Clean abstracts (remove special characters, lowercase)
   - TF-IDF vectorization with specified parameters
   - Save vectorizer for deployment inference

2. **Venue Processing:**
   - Parse Scopus-provided metrics (SJR, CiteScore, SNIP)
   - Compute percentiles and composite scores
   - Handle missing values with median imputation

3. **Author Processing:**
   - Count authors, institutions, countries from metadata
   - Compute collaboration flags and team size categories
   - Handle missing values with median imputation

4. **Feature Combination:**
   - Concatenate all feature types (text + venue + author)
   - Result: 14,832 × 5,019 feature matrix
   - Saved as pickle for model training

---

## Usage in Prediction

For a **new paper at publication time:**
1. Extract abstract → TF-IDF transform using saved vectorizer → 5,000 features
2. Lookup journal metrics (SJR, CiteScore, SNIP) for publication year → 9 features
3. Count authors, institutions, countries from byline → 10 features
4. Concatenate → 5,019-dimensional feature vector
5. Feed to trained model → Predict citation impact

All information required is available at the moment of publication.
