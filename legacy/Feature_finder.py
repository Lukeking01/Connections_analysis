import sqlite3
import pandas as pd
import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from config import *


SCRABBLE_VALUES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2,
    'H': 4, 'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1,
    'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
}

DIFFICULTY_MAP = {
    "yellow": 1,
    "green": 2,
    "blue": 3,
    "purple": 4,
}

# ================================
# Define grouping keys
# ================================

group_keys = ["puzzle_date", "category"]
puzzle_keys = ["puzzle_date"]


# ================================
# Read the database
# ================================

def load_database(path):
    conn = sqlite3.connect(path)
    words = pd.read_sql("SELECT * FROM words", conn)
    conn.close()
    return words


# ================================
# Clean words and encode difficulty
# ================================

def clean_encode(words):
    words = words.copy()
    words["word"] = words["word"].astype("string").str.strip().str.upper()
    words["word_length"] = words["word"].str.len()
    words["color"] = words["color"].str.lower().map(DIFFICULTY_MAP)
    return words


def _scrabble_score(word):
    if pd.isna(word):
        return pd.NA
    return sum(SCRABBLE_VALUES.get(letter, 0) for letter in word)


def add_scrabble_score(words):
    words = words.copy()
    words["scrabble_score"] = words["word"].apply(_scrabble_score)
    words["scrabble_per_letter"] = (
        words["scrabble_score"] / words["word_length"].replace(0, pd.NA)
    )
    return words


# ================================
# Semantic embeddings
# ================================

def load_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a pretrained sentence transformer (downloads on first run)."""
    return SentenceTransformer(model_name)


def add_embeddings(words, encoder=None):
    """
    Add a column of normalized embedding vectors.
    Returns (words_with_embeddings, embeddings_dict).
    """
    words = words.copy()
    if encoder is None:
        encoder = load_encoder()

    unique_words = words["word"].dropna().unique().tolist()
    vectors = encoder.encode(
        unique_words,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb_dict = dict(zip(unique_words, vectors))

    # Store the vector itself (useful later) – optional, can be dropped if memory is tight
    words["embedding"] = words["word"].map(emb_dict)
    return words, emb_dict


def _pairwise_cosine_stats(vectors: np.ndarray) -> dict:
    """
    Given a (4, dim) matrix of normalized embeddings,
    return mean / min / max / std of the 6 pairwise cosines.
    """
    if len(vectors) != 4:
        return {
            "mean_semantic_sim": np.nan,
            "min_semantic_sim": np.nan,
            "max_semantic_sim": np.nan,
            "std_semantic_sim": np.nan,
        }
    cosine = vectors @ vectors.T
    pair_sims = cosine[np.triu_indices(4, k=1)]
    return {
        "mean_semantic_sim": float(pair_sims.mean()),
        "min_semantic_sim": float(pair_sims.min()),
        "max_semantic_sim": float(pair_sims.max()),
        "std_semantic_sim": float(pair_sims.std()),
    }


# ================================
# Group-level features
# One row per category
# ================================

def get_group_features(words, emb_dict=None):
    """
    Compute lexical + semantic features for every (puzzle_date, category) group.
    If emb_dict is provided, semantic similarity features are added.
    """
    # ----- lexical features (your original code) -----
    group_features = (
        words.groupby(group_keys, dropna=False)
        .agg(
            difficulty=("color", "first"),          # you mapped color → difficulty
            mean_scrabble=("scrabble_score", "mean"),
            max_scrabble=("scrabble_score", "max"),
            min_scrabble=("scrabble_score", "min"),
            sd_scrabble=("scrabble_score", "std"),
            mean_scrabble_per_letter=("scrabble_per_letter", "mean"),
            mean_word_length=("word_length", "mean"),
            max_word_length=("word_length", "max"),
            sd_word_length=("word_length", "std"),
        )
        .reset_index()
    )

    # ----- semantic features -----
    if emb_dict is not None:
        semantic_rows = []
        for keys, group in words.groupby(group_keys, dropna=False):
            word_list = group["word"].tolist()
            vectors = np.stack([emb_dict[w] for w in word_list if w in emb_dict])
            stats = _pairwise_cosine_stats(vectors)
            row = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
            row.update(stats)
            semantic_rows.append(row)

        semantic_df = pd.DataFrame(semantic_rows)
        group_features = group_features.merge(semantic_df, on=group_keys, how="left")

    return group_features


# ================================
# Puzzle-level features
# One row per puzzle
# ================================

def get_puzzle_features(words, group_features=None):
    """
    Puzzle-level aggregates.
    If group_features (with semantic columns) is passed, those are also aggregated.
    """
    puzzle_features = (
        words.groupby(puzzle_keys, dropna=False)
        .agg(
            mean_scrabble=("scrabble_score", "mean"),
            max_scrabble=("scrabble_score", "max"),
            mean_word_length=("word_length", "mean"),
            sd_word_length=("word_length", "std"),
            mean_group_difficulty=("color", "mean"),
            max_group_difficulty=("color", "max"),
            min_group_difficulty=("color", "min"),
        )
        .reset_index()
    )

    puzzle_features["group_difficulty_range"] = (
        puzzle_features["max_group_difficulty"]
        - puzzle_features["min_group_difficulty"]
    )

    # Optional: aggregate the semantic group features up to puzzle level
    if group_features is not None and "mean_semantic_sim" in group_features.columns:
        sem_agg = (
            group_features.groupby(puzzle_keys, dropna=False)
            .agg(
                mean_group_semantic_sim=("mean_semantic_sim", "mean"),
                max_group_semantic_sim=("mean_semantic_sim", "max"),
                min_group_semantic_sim=("mean_semantic_sim", "min"),
                std_group_semantic_sim=("mean_semantic_sim", "std"),
            )
            .reset_index()
        )
        puzzle_features = puzzle_features.merge(sem_agg, on=puzzle_keys, how="left")

    return puzzle_features