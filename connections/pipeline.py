"""Small, explicit entry points for notebook data preparation."""
from pathlib import Path

from .paths import RAW, SEMANTIC


def prepare_data(mode="existing", semantic=False, start_date=None, end_date=None):
    """Use existing CSVs, rebuild from SQLite, or collect then rebuild.

    Semantic CSVs must exist for the current notebook analysis. Set
    ``semantic=True`` to regenerate them with Sentence Transformers.
    """
    from build_csvs import read_words, save_csv, build_semantic_tables, MODEL_NAME

    database = RAW / "connections.db"
    if mode not in {"existing", "database", "scrape"}:
        raise ValueError("mode must be 'existing', 'database', or 'scrape'")
    if mode == "scrape":
        from .scraper import update_database
        update_database(start_date=start_date, end_date=end_date, database=database)
    if mode in {"database", "scrape"}:
        words = read_words(database)
        save_csv(words, RAW / "connections2.csv")
    if semantic:
        words = read_words(database)
        groups, pairs = build_semantic_tables(words, MODEL_NAME)
        save_csv(groups, SEMANTIC / "semantic_group_statistics.csv")
        save_csv(pairs, SEMANTIC / "semantic_pair_statistics.csv")
    required = [RAW / "connections2.csv", SEMANTIC / "semantic_group_statistics.csv",
                SEMANTIC / "semantic_pair_statistics.csv"]
    missing = [str(path.relative_to(RAW.parents[1])) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing local data files: " + ", ".join(missing) +
            ". Set DATA_MODE='scrape' and REBUILD_SEMANTIC=True to collect "
            "and build everything, or supply your local database and choose DATA_MODE='database'."
        )
    print(f"Data ready ({mode} mode).")
