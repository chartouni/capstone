# Ex Post Feature Audit Checklist
# How to verify you're NOT using post-publication data

## Understanding Ex Ante vs Ex Post Features

**Ex Ante (Valid ✅):** Information available AT publication time
**Ex Post (Invalid ❌):** Information that accumulates AFTER publication

---

## FEATURE AUDIT CHECKLIST

### 1. TEXT FEATURES (TF-IDF from abstracts) ✅
**Status:** VALID - Abstracts are written before publication

**Check:**
- [ ] Using abstract text? → ✅ Valid
- [ ] Using full text/PDF content? → ⚠️ Check if available at publication
- [ ] Using citations within the text? → ❌ Invalid (references are OK, citations TO the paper are not)

**Your status:** ✅ Using abstracts only

---

### 2. VENUE FEATURES (Journal metrics) ⚠️ NEEDS VERIFICATION

**The Critical Question:** When were these metrics calculated?

**Valid scenarios:**
- ✅ Journal's 2018 SJR/CiteScore for a 2018 paper (based on 2015-2017 data)
- ✅ Historical journal metrics from BEFORE or AT publication year

**Invalid scenarios:**
- ❌ Journal's 2024 SJR/CiteScore for a 2018 paper (includes future data)
- ❌ Journal metrics calculated AFTER the paper's publication year

**How to check:**
Look at your SciVal/Scopus data columns:
- "CiteScore (publication year)" → Likely VALID ✅
- "CiteScore (current)" → INVALID ❌
- "SJR (publication year)" → Likely VALID ✅

**Your features:**
- snip (SNIP publication year)
- snip_percentile
- citescore (CiteScore publication year)
- citescore_percentile
- sjr (SJR publication year)
- sjr_percentile
- avg_venue_percentile
- is_top_journal
- venue_score_composite

**Verification needed:**
Check if "(publication year)" means:
1. Journal's metrics IN that year (based on prior years) → ✅ Valid
2. Metrics OF papers from that year (based on future citations) → ❌ Invalid

**How to verify:** See Section 4 below

---

### 3. AUTHOR FEATURES (Collaboration metrics) ✅

**Your features:**
- num_authors → ✅ Known at publication
- num_institutions → ✅ Known at publication
- num_countries → ✅ Known at publication
- is_single_author → ✅ Derived from above
- is_international_collab → ✅ Derived from above
- is_multi_institution → ✅ Derived from above
- authors_per_institution → ✅ Derived from above
- team_size_small/medium/large → ✅ Derived from above

**Status:** ✅ ALL VALID - These are all observable at publication time

**Note:** You correctly do NOT use H-index, which would be ex post unless historical

---

### 4. CITATION-DERIVED FEATURES ❌

**These are ALWAYS ex post - check you removed them:**
- [ ] Citations (target variable only - not a feature) → Should only be in y, not X
- [ ] Field Weighted Citation Impact → ❌ Remove if present
- [ ] Citation percentiles → ❌ Remove if present
- [ ] Views count → ❌ Remove if present
- [ ] Downloads → ❌ Remove if present
- [ ] Altmetric scores → ❌ Remove if present (accumulate over time)
- [ ] "Highly Cited Paper" status → ❌ Remove if present

**Your status:** ✅ Already removed views and citation-derived features

---

## HOW TO VERIFY YOUR FEATURES

### Step 1: List all feature columns
Run this to see EXACTLY what's in your feature matrix:

```python
import pandas as pd
import pickle

# Load your combined features
X = pd.read_pickle('data/features/X_all.pkl')

print("All features in your model:")
print("="*60)
for i, col in enumerate(X.columns, 1):
    print(f"{i:4d}. {col}")

print(f"\nTotal features: {len(X.columns)}")
```

### Step 2: Check venue feature values
Verify venue metrics are temporal (not current):

```python
import pandas as pd

# Load cleaned data with original columns
df = pd.read_pickle('data/processed/cleaned_data.pkl')

# Sample papers from different years
print("Checking if venue metrics are temporal or current:")
print("="*70)

for year in [2015, 2018, 2020]:
    sample = df[df['Year'] == year].iloc[0:2]
    print(f"\nYear {year} papers:")

    if 'SNIP (publication year)' in df.columns:
        print(f"  SNIP values: {sample['SNIP (publication year)'].values}")
    if 'CiteScore (publication year)' in df.columns:
        print(f"  CiteScore values: {sample['CiteScore (publication year)'].values}")
    if 'SJR (publication year)' in df.columns:
        print(f"  SJR values: {sample['SJR (publication year)'].values}")

# If the same journal has DIFFERENT values for different years → ✅ Valid (temporal)
# If the same journal has IDENTICAL values across all years → ❌ Invalid (current snapshot)
```

### Step 3: Verify no citation leakage
Make sure Citations only appears as target (y), not in features (X):

```python
import pandas as pd

X = pd.read_pickle('data/features/X_all.pkl')

# Check for suspicious column names
suspicious_keywords = ['citation', 'cited', 'impact', 'view', 'download', 'altmetric', 'h-index', 'hindex']

print("Checking for suspicious feature names:")
print("="*60)

found_issues = []
for col in X.columns:
    col_lower = str(col).lower()
    for keyword in suspicious_keywords:
        if keyword in col_lower:
            found_issues.append((col, keyword))

if found_issues:
    print("⚠️  POTENTIAL ISSUES FOUND:")
    for col, keyword in found_issues:
        print(f"  - Feature '{col}' contains '{keyword}'")
else:
    print("✅ No suspicious feature names found")

# Verify Citations is NOT in features
if 'Citations' in X.columns or 'citations' in X.columns:
    print("\n❌ ERROR: Citations column found in features!")
else:
    print("\n✅ Citations correctly excluded from features")
```

---

## FINAL VERIFICATION: The Time Travel Test

**Ask yourself:**
If I had a NEW paper published TODAY (2026), could I calculate ALL these features using ONLY information available today?

**For each feature, check:**

1. **Abstract text features** → Can I get the abstract? ✅ Yes
2. **Journal SJR/CiteScore** → Can I look up the journal's 2025 metrics? ✅ Yes
3. **Number of authors** → Is it on the paper? ✅ Yes
4. **Citations** → Do I know how many citations it will have in 5 years? ❌ NO

If you answer "NO" or "I need to wait X years" → That's an ex post feature!

---

## RECOMMENDATION

Run all 3 verification scripts above and check:
1. ✅ No citation-derived features in X
2. ✅ Venue metrics are temporal (different values for different years)
3. ✅ All features pass the "Time Travel Test"

If everything checks out → You're good! No ex post features.
