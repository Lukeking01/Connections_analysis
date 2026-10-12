
import sqlite3
import pandas as pd
import seaborn as sns
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
    words["word"] = words["word"].astype("string").str.strip().str.upper()
    words["word_length"] = words["word"].str.len()
    words["color"] = words["color"].str.lower().map(DIFFICULTY_MAP)
    return words

# Standard English Scrabble letter values


def _scrabble_score(word):
    if pd.isna(word):
        return pd.NA
    return sum(SCRABBLE_VALUES.get(letter, 0) for letter in word)

def add_scrabble_score(words):
    words["scrabble_score"] = words["word"].apply(_scrabble_score)

    words["scrabble_per_letter"] = (
        words["scrabble_score"] / words["word_length"].replace(0, pd.NA)
    )
    return words







# ================================
# Group-level features
# One row per category
# ================================
def get_group_features(words):
    group_features = (
        words.groupby(group_keys, dropna=False)
        .agg(
            difficulty=("difficulty", "first"),
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
    return group_features

# print("Group-level features:")
# print(group_features.head())


# ================================
# Puzzle-level features
# One row per puzzle
# ================================

def get_puzzle_features(words):
    puzzle_features = (
        words.groupby(puzzle_keys, dropna=False)
        .agg(
            mean_scrabble=("scrabble_score", "mean"),
            max_scrabble=("scrabble_score", "max"),
            mean_word_length=("word_length", "mean"),
            sd_word_length=("word_length", "std"),
            mean_group_difficulty=("difficulty", "mean"),
            max_group_difficulty=("difficulty", "max"),
            min_group_difficulty=("difficulty", "min"),
        )
        .reset_index()
    )

    puzzle_features["group_difficulty_range"] = (
        puzzle_features["max_group_difficulty"]
        - puzzle_features["min_group_difficulty"]
    )

    return puzzle_features

# print("\nPuzzle-level features:")
# print(puzzle_features.head())


