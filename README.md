# Connections Analysis

**One entry point, one notebook, one database, one neural-ML module.**

The SQLite file `data/raw/connections.db` is the canonical source. Everything else (CSV tables, notebook analysis and ML results) is derived from it. A populated database is included for offline work.

## Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4  # optional synonym signal
```

## Everyday usage

```bash
python project.py scrape                              # append missing historical puzzles through yesterday
python project.py scrape --start 2026-10-01 --end 2026-10-10  # only a date range
python project.py build                               # SQLite -> basic analysis CSV
python project.py build --semantic                    # additionally rebuild semantic CSVs (slower)
python project.py train --epochs 30                   # attention + pooling + synonym features
python project.py train --epochs 30 --no-synonyms     # synonym ablation
python project.py difficulty                          # PyTorch difficulty regressor
jupyter lab Connections_BERN02.ipynb                # plots and all comparisons
```

Run commands from the extracted project directory. The scraping command is deliberately separate: ordinary training never silently requests new data. **After scraping, run `build --semantic`** to refresh all notebook source tables. First run of the embedding model may download its weights. Scraping requires network access and can fail if the third-party source changes; failures are logged in `data/raw/scrape_errors.csv` and existing rows are not erased.

## Structure

```text
Connections_analysis/
├── project.py                    # CLI: scrape, build, train, difficulty
├── Connections_BERN02.ipynb     # analysis and model comparison
├── requirements.txt             # one dependency list
├── README.md
├── DATA_NOTICE.md
├── connections/
│   ├── scraper.py               # web pages -> SQLite
│   ├── database.py              # SQLite -> CSV tables
│   ├── data.py                  # read and prepare puzzles
│   ├── analysis.py              # descriptive statistics
│   ├── ml.py                    # ALL PyTorch ML, pooling, synonyms, optimization, metrics
│   ├── clustering.py            # balanced K-means comparison baseline
│   └── paths.py                 # shared data locations
├── data/
│   ├── raw/                     # SQLite source plus exported CSV
│   ├── semantic_source/         # derived embedding similarities
│   └── derived/                 # model scores and analytics
└── tests/
```

**Model comparison:** the contextual PyTorch grouping solver evaluates pairwise AUC and exact four-group puzzle accuracy, with and without WordNet synonym features. A separate PyTorch difficulty regressor is compared with a training-mean baseline using MAE. Balanced K-means is retained as a **non-neural clustering baseline**, not as a sklearn MLP. sklearn is used only for clustering, PCA and evaluation utilities—not for neural training.

**Splits:** chronological 80% training / 20% validation; **no held-out test split**. Validation is used for early stopping and comparisons, so results are development estimates, not independent test estimates. Actual model performance requires executing training; no improvement is asserted in advance.

See `DATA_NOTICE.md` for data provenance and scraping limitations.
