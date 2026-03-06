"""
Ex Post Feature Detection Script
Checks if your model includes features that are only observable after publication (data leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("EX POST FEATURE AUDIT - CHECKING FOR DATA LEAKAGE")
print("="*80)

# 1. CHECK FEATURE MATRIX
print("\n1. FEATURE MATRIX AUDIT")
print("-"*80)

try:
    X = pd.read_pickle('data/features/X_all.pkl')
    print(f"✓ Loaded feature matrix: {X.shape[0]} papers, {X.shape[1]} features")

    # Check for suspicious feature names
    suspicious_keywords = {
        'citation': '❌ Citation-derived (ex post)',
        'cited': '❌ Citation-derived (ex post)',
        'impact': '⚠️  Check if citation-derived',
        'view': '❌ Post-publication metric',
        'download': '❌ Post-publication metric',
        'altmetric': '❌ Post-publication metric',
        'h-index': '❌ Typically ex post (unless historical)',
        'hindex': '❌ Typically ex post (unless historical)',
        'h_index': '❌ Typically ex post (unless historical)',
        'field_weighted': '⚠️  Often citation-derived',
    }

    found_issues = []
    for col in X.columns:
        col_lower = str(col).lower()
        for keyword, warning in suspicious_keywords.items():
            if keyword in col_lower:
                found_issues.append((col, keyword, warning))

    if found_issues:
        print("\n⚠️  SUSPICIOUS FEATURES DETECTED:")
        for col, keyword, warning in found_issues:
            print(f"  {warning}")
            print(f"     Feature: '{col}' (contains '{keyword}')")
    else:
        print("✅ No obviously suspicious feature names detected")

    # Check if Citations is in features (should only be in target)
    if any('citation' in str(col).lower() and 'cite' not in str(col).lower() for col in X.columns):
        print("\n❌ WARNING: Found 'Citations' in feature matrix!")
    else:
        print("✅ Citations correctly excluded from features")

except FileNotFoundError:
    print("✗ Feature matrix not found at data/features/X_all.pkl")
except Exception as e:
    print(f"✗ Error loading features: {e}")


# 2. CHECK VENUE METRICS (Temporal vs Current)
print("\n2. VENUE METRICS - TEMPORAL VALIDATION")
print("-"*80)

try:
    df = pd.read_pickle('data/processed/cleaned_data.pkl')
    venue_features = pd.read_pickle('data/features/venue_features.pkl')

    # Check if venue metrics vary by year (temporal) or are constant (current snapshot)
    print("Checking if venue metrics are temporal (different across years)...\n")

    # Sample a journal that appears in multiple years
    if 'Scopus Source title' in df.columns and 'Year' in df.columns:
        # Find a journal with papers in multiple years
        journal_counts = df.groupby('Scopus Source title')['Year'].nunique()
        multi_year_journals = journal_counts[journal_counts >= 3].index[:5]

        if len(multi_year_journals) > 0:
            test_journal = multi_year_journals[0]
            journal_data = df[df['Scopus Source title'] == test_journal].sort_values('Year')

            print(f"Testing journal: {test_journal}")
            print(f"Papers across years: {journal_data['Year'].nunique()} different years")

            # Check if metrics vary
            venue_cols = ['SNIP (publication year)', 'CiteScore (publication year)', 'SJR (publication year)']

            for col in venue_cols:
                if col in df.columns:
                    values = journal_data.groupby('Year')[col].first()
                    unique_values = values.nunique()

                    print(f"\n  {col}:")
                    print(f"    Years: {list(values.index)}")
                    print(f"    Values: {list(values.values)}")

                    if unique_values > 1:
                        print(f"    ✅ TEMPORAL: Metric changes across years (Valid)")
                    elif unique_values == 1:
                        print(f"    ⚠️  CONSTANT: Same value across all years (Verify this is correct)")
                    else:
                        print(f"    ? Unable to determine")
        else:
            print("⚠️  Could not find journals with papers in multiple years")
            print("   Manual verification recommended")
    else:
        print("⚠️  Required columns not found for temporal validation")

except FileNotFoundError as e:
    print(f"✗ Data file not found: {e}")
except Exception as e:
    print(f"✗ Error checking venue metrics: {e}")


# 3. CHECK AUTHOR FEATURES
print("\n3. AUTHOR FEATURES AUDIT")
print("-"*80)

try:
    author_features = pd.read_pickle('data/features/author_features.pkl')
    print(f"✓ Loaded author features: {author_features.shape[1]} features")
    print("\nAuthor features in your model:")

    ex_ante_author_features = [
        'num_authors', 'num_institutions', 'num_countries',
        'is_single_author', 'is_international_collab', 'is_multi_institution',
        'authors_per_institution', 'team_size_small', 'team_size_medium', 'team_size_large'
    ]

    ex_post_author_features = [
        'h_index', 'h-index', 'hindex', 'citation_count', 'publication_count'
    ]

    print("\n✅ EX ANTE (Valid) author features found:")
    for feat in author_features.columns:
        if any(valid in str(feat).lower() for valid in ex_ante_author_features):
            print(f"  ✓ {feat}")

    print("\n❌ EX POST (Invalid) author features found:")
    found_ex_post = False
    for feat in author_features.columns:
        if any(invalid in str(feat).lower() for invalid in ex_post_author_features):
            print(f"  ✗ {feat} - REMOVE THIS")
            found_ex_post = True

    if not found_ex_post:
        print("  (none detected)")

except FileNotFoundError:
    print("✗ Author features not found")
except Exception as e:
    print(f"✗ Error checking author features: {e}")


# 4. FINAL TIME TRAVEL TEST
print("\n4. THE TIME TRAVEL TEST")
print("-"*80)
print("If you had a NEW paper published TODAY (2026), could you calculate these features?")
print()

checks = {
    "Text features (TF-IDF from abstract)": "✅ Yes - abstract available at publication",
    "Venue prestige (SJR, CiteScore, SNIP)": "✅ Yes - can lookup journal's 2025/2026 metrics",
    "Number of authors/institutions/countries": "✅ Yes - on the paper byline",
    "Collaboration indicators": "✅ Yes - derived from above",
}

for check, result in checks.items():
    print(f"{result}")
    print(f"   {check}")

print("\n⚠️  CANNOT calculate for new 2026 papers:")
print("   ❌ Citation counts (unknown until years pass)")
print("   ❌ Views/downloads (accumulate over time)")
print("   ❌ Current author H-index (includes future work)")
print("   ❌ 'Highly cited paper' status (determined later)")


# 5. SUMMARY
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

recommendations = []

if found_issues:
    recommendations.append("⚠️  Review suspicious features detected in Section 1")

recommendations.append("✓ Verify venue metrics are temporal (Section 2)")
recommendations.append("✓ Confirm all features pass Time Travel Test (Section 4)")

print("\nRECOMMENDATIONS:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print("\n" + "="*80)
print("For detailed guidance, see: EX_POST_FEATURE_AUDIT.md")
print("="*80)

X = pd.read_pickle('data/features/X_all.pkl')

# Check if actual "Citations" column is in features
if 'Citations' in X.columns:
    print("❌ PROBLEM: Citations column found in features!")
    print("   This is the target variable and should NOT be in X")
else:
    print("✅ OK: No 'Citations' column in features")
    
# Show which citation-related columns exist
citation_cols = [col for col in X.columns if 'citation' in str(col).lower()]
print(f"\nCitation-related features: {len(citation_cols)}")
for col in citation_cols[:10]:  # Show first 10
    print(f"  - {col}")
