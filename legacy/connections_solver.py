"""
NYT Connections solver starter
==============================

Run this from the repository root, where Feature_finder.py and
data/connections.db are available.

Install dependencies:
    pip install pandas numpy scikit-learn sentence-transformers

The script trains a group-scoring model from the known categories, evaluates on
held-out puzzles, and then solves a puzzle by selecting four non-overlapping
groups with the highest total score.

Important:
- The solver receives only the 16 words at prediction time.
- Puzzle-level train/test splitting prevents groups from the same puzzle leaking
  into both sets.
- The first version is a research baseline, not a guarantee of solving every
  puzzle. Wordplay and multiple plausible meanings remain difficult.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Uses your project's existing loader, as in your notebook.
import Feature_finder as ff


RANDOM_STATE = 42
NEGATIVES_PER_GROUP = 20


def load_words() -> pd.DataFrame:
    """Load and standardize the data using the project's existing database."""
    df = ff.load_database("data/connections.db").copy()
    required = {"puzzle_date", "puzzle_number", "category", "word"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["word"] = df["word"].astype(str).str.strip().str.upper()
    df["category"] = df["category"].astype(str).str.strip()
    df["puzzle_id"] = (
        df["puzzle_date"].astype(str) + "_" + df["puzzle_number"].astype(str)
    )
    df["word_length"] = df["word"].str.len()

    # Scrabble scores are simple lexical features, not a measure of puzzle difficulty.
    letter_values = {
        **dict.fromkeys("AEILNORSTU", 1),
        **dict.fromkeys("DG", 2),
        **dict.fromkeys("BCMP", 3),
        **dict.fromkeys("FHVWY", 4),
        "K": 5, "J": 8, "X": 8, "Q": 10, "Z": 10,
    }
    df["scrabble_score"] = df["word"].apply(
        lambda word: sum(letter_values.get(ch, 0) for ch in word)
    )
    df["scrabble_per_letter"] = (
        df["scrabble_score"] / df["word_length"].replace(0, np.nan)
    )
    return df


def make_puzzle_records(df: pd.DataFrame) -> dict:
    """Return puzzle_id -> {'words': [...], 'groups': [[word,...], ...]}."""
    puzzles = {}
    for puzzle_id, part in df.groupby("puzzle_id", sort=False):
        # Category is the known answer during training only.
        groups = []
        for _, group in part.groupby("category", sort=False):
            group_words = group["word"].tolist()
            if len(group_words) == 4:
                groups.append(group_words)

        all_words = part["word"].tolist()
        if len(all_words) == 16 and len(groups) == 4 and len(set(all_words)) == 16:
            puzzles[puzzle_id] = {"words": all_words, "groups": groups}
    if not puzzles:
        raise ValueError(
            "No valid 16-word puzzles with four categories of four words were found. "
            "Check the database schema and duplicate/missing rows."
        )
    return puzzles


def load_encoder():
    """Load a pretrained semantic encoder. First run may download model weights."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def group_features(group_words, embeddings, word_stats):
    """
    Create features for a candidate group of four.
    embeddings: dict word -> normalized vector
    word_stats: dict word -> (length, Scrabble score, score per letter)
    """
    words = list(group_words)
    vectors = np.stack([embeddings[w] for w in words])
    cosine = vectors @ vectors.T
    pair_sims = cosine[np.triu_indices(4, k=1)]  # six pairwise similarities

    stats = np.array([word_stats[w] for w in words], dtype=float)
    lengths = stats[:, 0]
    scrabble = stats[:, 1]
    per_letter = stats[:, 2]

    return np.array([
        pair_sims.mean(),
        pair_sims.min(),
        pair_sims.max(),
        pair_sims.std(),
        lengths.mean(),
        lengths.std(),
        lengths.min(),
        lengths.max(),
        scrabble.mean(),
        scrabble.std(),
        scrabble.min(),
        scrabble.max(),
        per_letter.mean(),
        per_letter.std(),
    ], dtype=float)


def build_training_examples(puzzles, train_ids, embeddings, word_stats):
    """Positive examples are true categories; negatives are mixed-word groups."""
    X, y, puzzle_groups = [], [], []
    rng = random.Random(RANDOM_STATE)

    for puzzle_id in train_ids:
        record = puzzles[puzzle_id]
        words = record["words"]
        true_groups = [tuple(sorted(g)) for g in record["groups"]]
        true_set = set(true_groups)

        for group in true_groups:
            X.append(group_features(group, embeddings, word_stats))
            y.append(1)
            puzzle_groups.append(puzzle_id)

        # Draw negative four-word combinations from this puzzle.
        # Hard negatives may still be plausible, which is desirable.
        candidates = list(combinations(words, 4))
        rng.shuffle(candidates)
        negatives_added = 0
        for candidate in candidates:
            key = tuple(sorted(candidate))
            if key in true_set:
                continue
            X.append(group_features(candidate, embeddings, word_stats))
            y.append(0)
            puzzle_groups.append(puzzle_id)
            negatives_added += 1
            if negatives_added >= NEGATIVES_PER_GROUP * 4:
                break

    return np.vstack(X), np.asarray(y), np.asarray(puzzle_groups)


def make_candidate_groups(words):
    """Enumerate all 1,820 possible groups of four."""
    return list(combinations(range(len(words)), 4))


def solve_puzzle(words, model, embeddings, word_stats):
    """
    Score every candidate group, then solve an exact-cover optimization:
    choose four disjoint groups covering all 16 words, maximizing summed scores.
    """
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError("Please provide exactly 16 distinct words.")

    candidates = make_candidate_groups(words)
    X = np.vstack([
        group_features([words[i] for i in idxs], embeddings, word_stats)
        for idxs in candidates
    ])
    # Convert probability to log-odds so confident positives are rewarded.
    probs = model.predict_proba(X)[:, 1]
    probs = np.clip(probs, 1e-5, 1 - 1e-5)
    scores = np.log(probs / (1 - probs))

    by_word = defaultdict(list)
    masks = []
    for candidate_index, idxs in enumerate(candidates):
        mask = sum(1 << i for i in idxs)
        masks.append(mask)
        for i in idxs:
            by_word[i].append(candidate_index)

    full_mask = (1 << 16) - 1

    @lru_cache(None)
    def best(remaining_mask):
        if remaining_mask == 0:
            return 0.0, ()
        # Choose the lowest-index remaining word; every solution must include it
        # in exactly one of the candidate groups considered here.
        first_bit = remaining_mask & -remaining_mask
        first_index = first_bit.bit_length() - 1
        best_score = -float("inf")
        best_solution = ()
        for candidate_index in by_word[first_index]:
            mask = masks[candidate_index]
            if mask & remaining_mask != mask:
                continue
            rest_score, rest_solution = best(remaining_mask ^ mask)
            total = scores[candidate_index] + rest_score
            if total > best_score:
                best_score = total
                best_solution = (candidate_index,) + rest_solution
        return best_score, best_solution

    total_score, solution_indices = best(full_mask)
    groups = [
        {
            "words": [words[i] for i in candidates[candidate_index]],
            "model_probability": float(probs[candidate_index]),
        }
        for candidate_index in solution_indices
    ]
    return groups, float(total_score)



def solve_semantic_baseline(words, embeddings):
    """
    Untrained baseline: rank groups by their average pairwise embedding similarity,
    then choose a complete non-overlapping partition. This is useful for checking
    whether supervised training improves on semantic similarity alone.
    """
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError("Please provide exactly 16 distinct words.")

    candidates = make_candidate_groups(words)
    scores = []
    for idxs in candidates:
        vectors = np.stack([embeddings[words[i]] for i in idxs])
        cosine = vectors @ vectors.T
        pair_sims = cosine[np.triu_indices(4, k=1)]
        scores.append(float(pair_sims.mean()))

    by_word = defaultdict(list)
    masks = []
    for candidate_index, idxs in enumerate(candidates):
        mask = sum(1 << i for i in idxs)
        masks.append(mask)
        for i in idxs:
            by_word[i].append(candidate_index)

    full_mask = (1 << 16) - 1

    @lru_cache(None)
    def best(remaining_mask):
        if remaining_mask == 0:
            return 0.0, ()
        first_bit = remaining_mask & -remaining_mask
        first_index = first_bit.bit_length() - 1
        best_score = -float("inf")
        best_solution = ()
        for candidate_index in by_word[first_index]:
            mask = masks[candidate_index]
            if mask & remaining_mask != mask:
                continue
            rest_score, rest_solution = best(remaining_mask ^ mask)
            total = scores[candidate_index] + rest_score
            if total > best_score:
                best_score = total
                best_solution = (candidate_index,) + rest_solution
        return best_score, best_solution

    total_score, solution_indices = best(full_mask)
    groups = [
        {
            "words": [words[i] for i in candidates[candidate_index]],
            "model_probability": None,
            "semantic_similarity": scores[candidate_index],
        }
        for candidate_index in solution_indices
    ]
    return groups, float(total_score)


def evaluate_semantic_baseline(puzzles, puzzle_ids, embeddings):
    exact = 0
    group_hits = 0
    total_groups = 0
    for puzzle_id in puzzle_ids:
        record = puzzles[puzzle_id]
        predicted, _ = solve_semantic_baseline(record["words"], embeddings)
        pred_set = {tuple(sorted(item["words"])) for item in predicted}
        true_set = {tuple(sorted(group)) for group in record["groups"]}
        exact += int(pred_set == true_set)
        group_hits += len(pred_set & true_set)
        total_groups += 4
    n = len(puzzle_ids)
    return {
        "puzzles_tested": n,
        "exact_puzzle_accuracy": exact / n if n else float("nan"),
        "group_accuracy": group_hits / total_groups if total_groups else float("nan"),
    }


def exact_puzzle_accuracy(puzzles, puzzle_ids, model, embeddings, word_stats):
    """Exact accuracy means all four predicted groups match the four true groups."""
    exact = 0
    group_hits = 0
    total_groups = 0

    for puzzle_id in puzzle_ids:
        record = puzzles[puzzle_id]
        predicted, _ = solve_puzzle(record["words"], model, embeddings, word_stats)
        pred_set = {tuple(sorted(item["words"])) for item in predicted}
        true_set = {tuple(sorted(group)) for group in record["groups"]}
        exact += int(pred_set == true_set)
        group_hits += len(pred_set & true_set)
        total_groups += 4

    n = len(puzzle_ids)
    return {
        "puzzles_tested": n,
        "exact_puzzle_accuracy": exact / n if n else float("nan"),
        "group_accuracy": group_hits / total_groups if total_groups else float("nan"),
    }


def main():
    df = load_words()
    puzzles = make_puzzle_records(df)
    puzzle_ids = np.array(list(puzzles.keys()))

    print(f"Valid puzzles: {len(puzzle_ids)}")
    print("Example puzzle:", puzzles[puzzle_ids[0]]["words"])

    # Split by puzzle before creating training examples.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    dummy_X = np.zeros((len(puzzle_ids), 1))
    train_idx, test_idx = next(splitter.split(dummy_X, groups=puzzle_ids))
    train_ids = puzzle_ids[train_idx].tolist()
    test_ids = puzzle_ids[test_idx].tolist()

    all_words = sorted({word for rec in puzzles.values() for word in rec["words"]})
    print("Loading pretrained semantic encoder...")
    encoder = load_encoder()
    vectors = encoder.encode(all_words, normalize_embeddings=True, show_progress_bar=True)
    embeddings = dict(zip(all_words, vectors))

    # Lexical stats from the dataframe; all words are uppercased in load_words().
    stats_df = df.drop_duplicates("word").set_index("word")
    word_stats = {
        word: (
            float(stats_df.loc[word, "word_length"]),
            float(stats_df.loc[word, "scrabble_score"]),
            float(stats_df.loc[word, "scrabble_per_letter"]),
        )
        for word in all_words
    }

    X_train, y_train, _ = build_training_examples(
        puzzles, train_ids, embeddings, word_stats
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, y_train)

    # Candidate-group classification metrics on held-out puzzles.
    X_test, y_test, _ = build_training_examples(
        puzzles, test_ids, embeddings, word_stats
    )
    pred = model.predict(X_test)
    print("\nHeld-out candidate-group classification")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 3))
    print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 3))
    print(classification_report(y_test, pred, target_names=["not a group", "true group"]))

    print("\nHeld-out full-puzzle solving")
    print("Supervised group model:", exact_puzzle_accuracy(
        puzzles, test_ids, model, embeddings, word_stats
    ))
    print("Untrained semantic baseline:", evaluate_semantic_baseline(
        puzzles, test_ids, embeddings
    ))

    # Example: solve a puzzle from the dataset. For a genuinely new puzzle,
    # replace this list with the 16 words and ensure the encoder has their vectors.
    example_id = test_ids[0]
    example_words = puzzles[example_id]["words"]
    answer, _ = solve_puzzle(example_words, model, embeddings, word_stats)
    print(f"\nExample held-out puzzle: {example_id}")
    for i, group in enumerate(answer, start=1):
        print(f"Group {i}: {group['words']} (score={group['model_probability']:.3f})")


if __name__ == "__main__":
    main()
