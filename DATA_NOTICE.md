# Data Source Notice

## Source

The Connections puzzle data in this project was collected from:

https://www.fiveletterwords.io/

FiveLetterWords is an independent project and is not affiliated with or
endorsed by The New York Times.

## Purpose

This dataset was collected for research and educational purposes, including
the statistical analysis of Connections puzzle structure and difficulty.

The dataset is used as an input to analysis performed by this project.

## Rights and licensing

This project does not claim ownership of the underlying puzzle content and
does not grant a new license to that content.

The license for the source code of this repository is separate from the
rights applicable to the collected puzzle data.

Please consult the source website's current licensing and attribution
information before redistributing the raw dataset:

https://www.fiveletterwords.io/licenses

For research-use guidance:

https://www.fiveletterwords.io/press

## Collection information

Data source: FiveLetterWords  
Collection method: Automated HTTP retrieval  
Collection date: Not documented in the provided repository snapshot  
Date range: Determined from the bundled data (see notebook quality section)

The dataset may contain information reproduced from the source website.
Changes to the source website may result in differences between this dataset
and subsequently collected versions.

## Attribution

If you publish research, analysis, visualizations, or other material based
on this dataset, please acknowledge FiveLetterWords as the data source and
provide a link to the relevant source page or pages.

Suggested attribution:

> Connections puzzle data collected from FiveLetterWords
> (https://www.fiveletterwords.io/), accessed on the date of the original collection (not recorded in this snapshot).

## Local data pipeline

`data/raw/connections.db` is the populated SQLite source used by
`build_csvs.py` to regenerate `data/raw/connections2.csv`. The semantic
statistics in `data/semantic_source/` are generated separately with a
pretrained Sentence Transformers model (`python build_csvs.py --semantic`).
Those regenerated values can differ from the bundled snapshots if the model,
its version or the preprocessing changes. The notebook writes analytical
outputs to `data/derived/`; the source data remains unchanged.

The April 1, 2024 emoji puzzle is preserved in the raw data but excluded
from analyses that assume ordinary written words.

## Public repository and notebook results

The public repository distributes the scraper, analysis code, documentation,
and notebook **analysis outputs** (aggregate statistics, charts, model results).
It does **not** distribute the SQLite database, puzzle-level raw CSV files,
semantic CSV datasets, or embedding caches. These are stored locally under
`data/`, which is ignored by Git. Notebook cells that display individual raw
puzzle records should be reviewed before publication.

To recreate the analysis without distributing the data, a reader can choose
`DATA_MODE = "scrape"` and `REBUILD_SEMANTIC = True` in the first notebook
cell. Scraping is optional, subject to source-site availability and applicable
terms. The source content remains subject to its own rights and conditions.
The newly collected dataset may differ from the one used to produce the
published notebook outputs.
