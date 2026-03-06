# How to Revert Code Review Changes

All code review improvements were made on branch **`code-review-improvements`**.

## To discard all changes and go back to your previous branch

```powershell
git checkout claude/citation-predictor-setup-CccfT
git branch -D code-review-improvements   # optional: delete the branch
```

## To revert specific commits (while staying on code-review-improvements)

```powershell
git log --oneline   # view commit history
git revert <commit-hash>   # revert a specific commit
```

## What was changed

1. **src/features/venue_features.py** – Training mode now uses real venue statistics when `df` + `citation_col` are provided
2. **src/features/author_features.py** – Improved `parse_author_count` for " and ", " & ", semicolon, and comma formats
3. **app/main.py** – Metrics loaded from `reports/metrics/*.csv` instead of hardcoded values
4. **src/deployment/prediction_service.py** – Added feature validation before prediction
5. **src/deployment/model_loader.py** – Added `get_expected_feature_names()` for validation
6. **tests/** – New pytest suite (22 tests) for data loading, author features, venue features
