# Connections | Reproducible puzzle analysis

A readable, course-level statistical learning project built around historical
**NYT Connections** puzzles. The main notebook tells the story; the Python
modules handle data preparation, analysis and solution prediction.

## Quick start and data collection

Install dependencies:

```bash
python -m pip install -r requirements.txt
# Needed when creating semantic CSVs or using the embedding-based solver:
python -m pip install -r requirements-semantic.txt
jupyter lab Connections_BERN02.ipynb
```

The **first code cell** selects the data workflow:

| `DATA_MODE` | Action | Network access |
| --- | --- | --- |
| `"existing"` (default) | Read existing local CSVs | No |
| `"database"` | Rebuild word CSV from local SQLite | No |
| `"scrape"` | Scrape missing dates, update SQLite, rebuild word CSV | Yes |

For a fresh clone with no local data, set `DATA_MODE = "scrape"` and
`REBUILD_SEMANTIC = True`. This requires Sentence Transformers and downloads
its pretrained model. You may optionally specify `START_DATE` and `END_DATE`.
The default scraping range starts at the first puzzle and ends yesterday.
Scraping is opt-in, and the scraper skips dates already in the database.

The scraper relies on the source website's current HTML and may require
maintenance if the site changes. Difficulty ratings can remain missing.
Generated semantic features may differ from previously saved notebook outputs.

## Project layout

```text
Connections_BERN02.ipynb  # Notebook with the data-source switch and analysis
connections/
  scraper.py             # Data collection into SQLite
  pipeline.py            # Notebook data preparation entry point
  data.py                # Data loading and feature engineering
  analysis.py            # Statistics, regression, PCA and puzzle clustering
  supervised.py          # Pairwise classification and grouping
  clustering.py          # Balanced K-means word solver
  paths.py               # Shared paths
build_csvs.py            # SQLite to CSV, optional semantic statistics
DATA_NOTICE.md           # Data source and rights information
.gitignore               # Excludes local databases and CSVs
requirements*.txt        # Dependencies
notebooks/               # Archived analysis notebooks (local project)
legacy/                  # Historical experiments (local project)
data/                    # Local only, ignored by Git
```

## GitHub publication

This project **does not publish** the puzzle database, raw puzzle records,
CSV datasets, or cached embeddings. The notebook's aggregate statistical
analysis, plots and evaluation outputs may remain in the public repository,
as described in `DATA_NOTICE.md`. Review notebook outputs for raw puzzle
records before committing. `.gitignore` cannot untrack files already committed.

A clean GitHub-ready code-only package is distributed separately from the
full local research package. Do not upload the full local package to GitHub.

## What the notebook does

1. Audits the dataset and missing difficulty ratings.
2. Studies difficulty trends, group similarities and word recurrence.
3. Creates puzzle-level features, Spearman correlations and linear regression.
4. Compares regression with a mean baseline using a chronological test split
   and time-series cross-validation.
5. Uses PCA and K-means to **cluster entire puzzles**.
6. Uses logistic regression and random forest to **predict word groups**.
7. Optionally adds Sentence Transformers semantic similarities.
8. Uses **balanced K-means on the 16 words of one puzzle** to predict four
   non-overlapping groups of exactly four words.
9. Optionally trains small **MLPs with lexical and semantic features** to
   predict puzzle solutions and published difficulty (compared with a baseline).

Model selection is based on validation data, with later puzzles held out for
final evaluation. Full puzzle recovery is substantially harder than pairwise
classification. Semantic embeddings are a pretrained *feature representation*;
the statistical methods remain course-level.

## Rebuild the source CSVs

```bash
python build_csvs.py
python build_csvs.py --semantic
```

The second command needs `requirements-semantic.txt`. The SQLite database
does **not** contain semantic similarity scores. Rebuilding them can change
numerical results relative to the bundled snapshots. All CSVs stay in `data/`.

## Known limitations

- The April 1, 2024 puzzle uses emojis and is excluded from word-based analyses.
- Difficulty ratings are missing for some puzzles; these rows are excluded
  only when the relevant statistical method requires a rating.
- The supervised lexical baseline cannot understand many semantic categories.
- Balanced K-means is constrained to four words per group but does not
  guarantee the correct Connections solution.
- Some older scripts and results are preserved under `legacy/` and
  `notebooks/` for provenance, not presented as the recommended workflow.

## Data source and attribution

Historical Connections puzzle data was collected from
[FiveLetterWords](https://www.fiveletterwords.io/), an independent project
not affiliated with or endorsed by The New York Times. The scraper supports
reproducible research; including data does not grant redistribution rights.

The source data and puzzle content remain subject to their respective rights.
The **MIT License applies only to the original source code**, not the
underlying puzzle content. See [DATA_NOTICE.md](DATA_NOTICE.md) and the
source's [licenses](https://www.fiveletterwords.io/licenses) and
[press information](https://www.fiveletterwords.io/press).

The source website can change over time. For publication, identify the source,
collection date if known, and analysis-code version. The original collection
date is not documented in the supplied snapshot.

## Optional MLP experiments

Install `requirements-semantic.txt`, then set `RUN_MLP = True` in Section 10
of the notebook. The MLP uses Sentence Transformers as a fixed semantic feature
extractor, plus interpretable lexical features. It trains one classifier for
word-pair grouping and one regressor for published difficulty. It does not
train or fine-tune the transformer. Validation and test puzzles are later
than training puzzles; missing difficulty ratings are excluded only from
the regression targets. Metrics and predictions are saved in `data/derived/`.
The MLP is an optional extension rather than a syllabus-required method.
