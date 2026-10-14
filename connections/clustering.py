"""Unsupervised Connections solver: Sentence Transformers + balanced K-means.

Each puzzle is solved independently; correct answers are used only for evaluation.
Run: python clustering_solver.py --evaluate 30
Or:  python clustering_solver.py --words APPLE BANANA ... (exactly 16 words)
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "connections2.csv"
OUTPUT = ROOT / "data" / "derived"


def load_encoder():
    """Load the fixed pretrained model; it downloads once on first use."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install with: pip install sentence-transformers") from exc
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def balanced_kmeans(vectors, random_state=42, n_init=10, max_iter=30):
    """Assign 16 embeddings to four clusters, each containing exactly four words.

    K-means supplies initial centers. The Hungarian algorithm assigns four slots
    to each center; updated centers are normalized means of assigned words.
    Several starts are compared using the total within-group cosine similarity.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != 16:
        raise ValueError("Expected a matrix containing exactly 16 word embeddings.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite.")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError("Zero-length embedding encountered.")
    vectors = vectors / lengths

    best_groups, best_score = None, -np.inf
    for seed in range(random_state, random_state + n_init):
        initial = KMeans(n_clusters=4, random_state=seed, n_init=1).fit(vectors)
        centers = initial.cluster_centers_
        labels = None
        for _ in range(max_iter):
            center_norms = np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
            unit_centers = centers / center_norms
            similarities = vectors @ unit_centers.T
            # Each center has four identical assignment slots.
            costs = -np.repeat(similarities, 4, axis=1)
            row_ids, slot_ids = linear_sum_assignment(costs)
            new_labels = np.empty(16, dtype=int)
            new_labels[row_ids] = slot_ids // 4
            if labels is not None and np.array_equal(labels, new_labels):
                break
            labels = new_labels
            centers = np.array([vectors[labels == group].mean(axis=0) for group in range(4)])

        groups = [np.flatnonzero(labels == group).tolist() for group in range(4)]
        # Use a comparable objective for all random starts.
        score = sum(
            float((vectors[group] @ vectors[group].T)[np.triu_indices(4, 1)].sum())
            for group in groups
        )
        if score > best_score:
            best_groups, best_score = groups, score
    return best_groups


def solve_puzzle(puzzle_words, encoder, random_state=42):
    """Return four lists of four words; no correct categories are required."""
    if len(puzzle_words) != 16 or len(set(puzzle_words)) != 16:
        raise ValueError("Provide exactly 16 distinct words.")
    vectors = encoder.encode(list(puzzle_words), normalize_embeddings=True)
    groups = balanced_kmeans(vectors, random_state=random_state)
    return [[puzzle_words[index] for index in group] for group in groups]


def evaluate_puzzles(encoder, n_test=30):
    """Evaluate on the latest n_test dated puzzles; answers are never used to fit."""
    words = pd.read_csv(DATA, keep_default_na=False)
    dates = sorted(words["puzzle_date"].unique())
    rows = []
    for date in dates[-n_test:]:
        puzzle = words.loc[words["puzzle_date"] == date].sort_values(["color", "position"])
        puzzle_words = puzzle["word"].tolist()
        # The emoji-only April Fools puzzle is not comparable to written words.
        if len(puzzle_words) != 16 or len(set(puzzle_words)) != 16 or date == "2024-04-01":
            continue
        predicted = solve_puzzle(puzzle_words, encoder)
        actual = [frozenset(group["word"]) for _, group in puzzle.groupby("color")]
        recovered = sum(frozenset(group) in actual for group in predicted)
        rows.append({"puzzle_date": date, "correct_groups": recovered,
                     "fully_solved": int(recovered == 4)})
    results = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "clustering_solver_test_results.csv", index=False)
    if len(results):
        print(f"Puzzles evaluated: {len(results)}")
        print(f"Mean correct groups: {results['correct_groups'].mean():.3f} / 4")
        print(f"Fully solved: {results['fully_solved'].mean():.1%}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=int, metavar="N", help="Evaluate on the last N puzzles")
    parser.add_argument("--words", nargs=16, metavar="WORD", help="Sixteen puzzle words")
    args = parser.parse_args()
    if args.evaluate is None and args.words is None:
        parser.error("Choose --evaluate N or --words followed by 16 words")
    model = load_encoder()
    if args.evaluate is not None:
        evaluate_puzzles(model, n_test=args.evaluate)
    if args.words is not None:
        for number, group in enumerate(solve_puzzle(args.words, model), 1):
            print(f"Group {number}: {', '.join(group)}")
