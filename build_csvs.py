"""Rebuild the BERN02 source CSVs from the original SQLite database.

Examples (run from the project directory):
    python build_csvs.py
    python build_csvs.py --semantic
    python build_csvs.py --database data/raw/connections.db --semantic

The database contains puzzle solutions, but not semantic similarity scores.
The --semantic option computes those scores with a pretrained model.
"""

import argparse
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
COLORS = ("yellow", "green", "blue", "purple")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_words(database):
    """Read the populated SQLite words table"""
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "words" not in tables:
            raise ValueError(f"Database {database} has no 'words' table")
        words = pd.read_sql_query(
            "SELECT puzzle_date, puzzle_number, difficulty, color, "
            "category, position, word FROM words", connection
        )
    if words.empty:
        raise ValueError(
            f"The words table in {database} is empty. "
            "Use the populated connections.db, not connections2.db."
        )
    if words[["puzzle_date", "color", "position", "word"]].isna().any().any():
        raise ValueError("Required puzzle fields contain missing values")
    words = words.sort_values(
        ["puzzle_date", "puzzle_number", "color", "position"],
        kind="stable",
    ).reset_index(drop=True)
    return words


def save_csv(frame, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame):,} rows: {destination.relative_to(PROJECT)}")


def build_semantic_tables(words, model_name):
    """Compute within-group and all-pair cosine similarity statistics.

    Embeddings are normalized, so their dot product equals cosine similarity.
    The same word can appear in multiple puzzles; it is encoded only once.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Semantic generation needs sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from error

    words = words.copy()
    words["word"] = words["word"].astype(str).str.strip().str.upper()
    unique_words = words["word"].drop_duplicates().tolist()
    print(f"Encoding {len(unique_words):,} unique words with {model_name}...")
    encoder = SentenceTransformer(model_name)
    vectors = encoder.encode(
        unique_words, normalize_embeddings=True,
        show_progress_bar=True, batch_size=128,
    )
    embeddings = dict(zip(unique_words, vectors))

    group_rows = []
    pair_rows = []
    for (date, number), puzzle in words.groupby(
        ["puzzle_date", "puzzle_number"], sort=True
    ):
        puzzle_id = f"{date}_{int(number)}"
        puzzle = puzzle.sort_values(["color", "position"])
        items = puzzle.to_dict("records")

        # All 120 pairs of 16 words, labeled using the known solution.
        for left, right in combinations(items, 2):
            similarity = float(np.dot(
                embeddings[left["word"]], embeddings[right["word"]]
            ))
            pair_rows.append({
                "puzzle_id": puzzle_id,
                "pair_type": (
                    "Same group" if left["color"] == right["color"]
                    else "Different groups"
                ),
                "similarity": similarity,
            })

        # Six within-group pairs for each group of four words.
        for (category, color), group in puzzle.groupby(
            ["category", "color"], sort=True
        ):
            group_words = group.sort_values("position")["word"].tolist()
            scores = [
                float(np.dot(embeddings[a], embeddings[b]))
                for a, b in combinations(group_words, 2)
            ]
            if not scores:
                raise ValueError(f"No word pairs in {puzzle_id}: {category}")
            group_rows.append({
                "puzzle_id": puzzle_id,
                "category": category,
                "color": color,
                "mean_similarity": float(np.mean(scores)),
                "median_similarity": float(np.median(scores)),
                "min_similarity": float(np.min(scores)),
                "max_similarity": float(np.max(scores)),
                "similarity_std": float(np.std(scores, ddof=0)),
            })

    groups = pd.DataFrame(group_rows)
    pairs = pd.DataFrame(pair_rows)
    return groups, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=PROJECT / "data/raw/connections.db",
        help="Path to the SQLite database",
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="Also rebuild semantic group and pair CSVs (downloads a model on first use)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="Sentence Transformers model used with --semantic",
    )
    args = parser.parse_args()
    database = args.database.resolve()
    words = read_words(database)
    save_csv(words, PROJECT / "data/raw/connections2.csv")
    print(f"Found {words['puzzle_date'].nunique():,} puzzle dates")
    if args.semantic:
        groups, pairs = build_semantic_tables(words, args.model)
        save_csv(groups, PROJECT / "data/semantic_source/semantic_group_statistics.csv")
        save_csv(pairs, PROJECT / "data/semantic_source/semantic_pair_statistics.csv")
    else:
        print("Semantic CSVs unchanged; pass --semantic to regenerate them.")


if __name__ == "__main__":
    main()
