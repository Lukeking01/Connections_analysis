"""
Evaluate the pure semantic NYT Connections solver on the entire dataset.
=======================================================================

Run from the repository root (same place as Feature_finder.py and data/connections.db):

    python evaluate_semantic.py
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

import Feature_finder as ff


# ---------------------------------------------------------------------------
# Data loading (simplified for your schema)
# ---------------------------------------------------------------------------

def load_words() -> pd.DataFrame:
    df = ff.load_database("data/connections.db").copy()

    required = {"puzzle_number", "category", "word"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["word"] = df["word"].astype(str).str.strip().str.upper()
    df["category"] = df["category"].astype(str).str.strip()
    # puzzle_number is already unique
    df["puzzle_id"] = df["puzzle_number"].astype(str)
    return df


def make_puzzle_records(df: pd.DataFrame) -> dict:
    """
    Every puzzle has:
      - 16 unique words
      - exactly 4 groups of 4 (one per color/category)
    """
    puzzles = {}
    for puzzle_id, part in df.groupby("puzzle_id", sort=False):
        groups = [
            group["word"].tolist()
            for _, group in part.groupby("category", sort=False)
        ]
        words = part["word"].tolist()

        # light sanity checks (should never fail given your description)
        if len(words) != 16 or len(set(words)) != 16:
            print(f"Skipping {puzzle_id}: expected 16 unique words, got {len(words)}")
            continue
        if len(groups) != 4 or any(len(g) != 4 for g in groups):
            print(f"Skipping {puzzle_id}: expected 4 groups of 4")
            continue

        puzzles[puzzle_id] = {"words": words, "groups": groups}

    if not puzzles:
        raise ValueError("No valid puzzles found.")
    return puzzles


# ---------------------------------------------------------------------------
# Pure semantic solver
# ---------------------------------------------------------------------------

def load_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def make_candidate_groups(n: int = 16):
    return list(combinations(range(n), 4))


def solve_semantic(words: list[str], embeddings: dict):
    words = [w.strip().upper() for w in words]
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError("Please provide exactly 16 distinct words.")

    candidates = make_candidate_groups(16)
    scores = []
    for idxs in candidates:
        vectors = np.stack([embeddings[words[i]] for i in idxs])
        cosine = vectors @ vectors.T
        pair_sims = cosine[np.triu_indices(4, k=1)]
        scores.append(float(pair_sims.mean()))

    by_word = defaultdict(list)
    masks = []
    for cand_idx, idxs in enumerate(candidates):
        mask = sum(1 << i for i in idxs)
        masks.append(mask)
        for i in idxs:
            by_word[i].append(cand_idx)

    full_mask = (1 << 16) - 1

    @lru_cache(None)
    def best(remaining_mask: int):
        if remaining_mask == 0:
            return 0.0, ()
        first_bit = remaining_mask & -remaining_mask
        first_index = first_bit.bit_length() - 1
        best_score = -float("inf")
        best_solution = ()
        for cand_idx in by_word[first_index]:
            mask = masks[cand_idx]
            if (mask & remaining_mask) != mask:
                continue
            rest_score, rest_sol = best(remaining_mask ^ mask)
            total = scores[cand_idx] + rest_score
            if total > best_score:
                best_score = total
                best_solution = (cand_idx,) + rest_sol
        return best_score, best_solution

    total_score, solution_indices = best(full_mask)

    groups = [
        {
            "words": [words[i] for i in candidates[idx]],
            "semantic_similarity": scores[idx],
        }
        for idx in solution_indices
    ]
    return groups, float(total_score)


# ---------------------------------------------------------------------------
# Evaluation on the full dataset
# ---------------------------------------------------------------------------

def evaluate(puzzles: dict, puzzle_ids: list[str], embeddings: dict):
    exact = 0
    group_hits = 0
    total_groups = 0
    failures = []

    for i, pid in enumerate(puzzle_ids, 1):
        if i % 50 == 0 or i == len(puzzle_ids):
            print(f"  processed {i}/{len(puzzle_ids)} puzzles …", flush=True)

        record = puzzles[pid]
        predicted, _ = solve_semantic(record["words"], embeddings)
        pred_set = {tuple(sorted(g["words"])) for g in predicted}
        true_set = {tuple(sorted(g)) for g in record["groups"]}

        if pred_set == true_set:
            exact += 1
        else:
            failures.append(pid)

        group_hits += len(pred_set & true_set)
        total_groups += 4

    n = len(puzzle_ids)
    return {
        "puzzles_tested": n,
        "exact_puzzle_accuracy": exact / n if n else float("nan"),
        "group_accuracy": group_hits / total_groups if total_groups else float("nan"),
        "exact_count": exact,
        "group_hits": group_hits,
        "failures": failures,
    }


def main(nr=100):
    print("Loading dataset …")
    df = load_words()
    puzzles = make_puzzle_records(df[:16*nr])
    puzzle_ids = list(puzzles.keys())
    print(f"Valid puzzles: {len(puzzle_ids)}")

    all_words = sorted({w for rec in puzzles.values() for w in rec["words"]})
    print(f"Unique words: {len(all_words)}")
    print("Loading encoder and computing embeddings …")
    encoder = load_encoder()
    vectors = encoder.encode(all_words, normalize_embeddings=True, show_progress_bar=True)
    embeddings = dict(zip(all_words, vectors))

    print("\nEvaluating pure semantic solver on the entire dataset …")
    results = evaluate(puzzles, puzzle_ids, embeddings)

    print("\n=== Results (all puzzles) ===")
    print(f"Puzzles tested       : {results['puzzles_tested']}")
    print(f"Exact puzzle accuracy: {results['exact_puzzle_accuracy']:.3f} "
          f"({results['exact_count']}/{results['puzzles_tested']})")
    print(f"Group accuracy       : {results['group_accuracy']:.3f} "
          f"({results['group_hits']}/{results['puzzles_tested'] * 4})")

    if results["failures"]:
        print(f"\nExample failures (first 5 of {len(results['failures'])}):")
        for pid in results["failures"][:5]:
            rec = puzzles[pid]
            pred, _ = solve_semantic(rec["words"], embeddings)
            print(f"\nPuzzle {pid}")
            print("  True groups:")
            for g in rec["groups"]:
                print(f"    {sorted(g)}")
            print("  Predicted groups:")
            for g in pred:
                print(f"    {sorted(g['words'])}  (sim={g['semantic_similarity']:.3f})")


if __name__ == "__main__":
    main()