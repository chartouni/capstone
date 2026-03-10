# Citation Impact Prediction — AUB Capstone

Predicting whether a paper will become highly cited using only metadata available at publication time (author reputation, venue prestige, abstract text, publication year).

**Best result**: **63.33% F1** — selective domain-segmented LogisticRegression (Experiment 10c)

---

## Project Structure

```
capstone/
├── notebooks/        # Jupyter notebooks (numbered by pipeline stage)
├── src/              # Reusable Python modules
├── app/              # Streamlit deployment app
├── scripts/          # Utility and diagnostic scripts
├── docs/             # Project documentation
│   └── archive/      # Older drafts and reports
├── config/           # Configuration files
├── data/             # Raw and processed data (not committed)
└── requirements.txt
```

## Notebooks

| Range | Stage |
|-------|-------|
| 00–06 | Data pipeline (lookup, merging, cleaning) |
| 10    | EDA — citation distribution |
| 20–24 | Feature engineering (text, venue, author) |
| 30–36 | Model development (classification, regression, tuning) |
| 37–42 | Experiments (SMOTE, domain segmentation, error analysis) |
| 50–53 | Unsupervised learning + EDA feature opportunities |
| 98–99 | Data quality checks |

## Key Results

See [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md) for the full experiment history and [`docs/FINAL_RESULTS_SUMMARY.md`](docs/FINAL_RESULTS_SUMMARY.md) for the final summary.

| Method | F1 |
|--------|----|
| Baseline (LogisticRegression, threshold=0.54) | 62.55% |
| **Selective domain segmentation (Exp 10c)** | **63.33%** |

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Data files are not committed. Place raw Scopus/SciVal exports in `data/` and run notebooks 00–06 to build the merged dataset.
