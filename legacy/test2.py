"""
NYT Connections – Semantic Solver + Evaluation
==============================================
Uses Feature_finder + SQLite.
"""

from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import numpy as np
from sentence_transformers import SentenceTransformer
import Feature_finder as ff


# ------------------------------------------------------------------
# 1. Load data & embeddings
# ------------------------------------------------------------------

def load_everything(db_path: str = "data/connections.db"):
    df = ff.load_database(db_path).copy()
    df["word"] = df["word"].astype(str).str.strip().str.upper()
    df["color"] = df["color"].astype(str).str.lower().str.strip()

    key = "puzzle_number" if "puzzle_number" in df.columns else "puzzle_date"
    complete = df.groupby(key).filter(
        lambda x: len(x) == 16 and x["word"].nunique() == 16
    )

    puzzles = []
    for pid, g in complete.groupby(key):
        puzzles.append({
            "id": str(pid),
            "words": g["word"].tolist(),
            "colors": g["color"].tolist(),
        })

    print(f"Loaded {len(puzzles)} complete puzzles")

    print("Loading encoder and computing embeddings …")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    unique_words = complete["word"].unique().tolist()
    vectors = encoder.encode(
        unique_words, normalize_embeddings=True, show_progress_bar=True
    )
    embeddings = dict(zip(unique_words, vectors))

    return encoder, embeddings, puzzles


encoder, embeddings, puzzles = load_everything()


# ------------------------------------------------------------------
# 2. Core solver
# ------------------------------------------------------------------

def solve(words: list[str], encoder=None, embeddings=None):
    if encoder is None:
        encoder = globals()["encoder"]
    if embeddings is None:
        embeddings = globals()["embeddings"]

    words = [w.strip().upper() for w in words]
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError("Please provide exactly 16 distinct words.")

    # Encode any new words on the fly
    missing = [w for w in words if w not in embeddings]
    if missing:
        new_vecs = encoder.encode(
            missing, normalize_embeddings=True, show_progress_bar=False
        )
        for w, v in zip(missing, new_vecs):
            embeddings[w] = v

    candidates = list(combinations(range(16), 4))
    scores = []
    for idxs in candidates:
        V = np.stack([embeddings[words[i]] for i in idxs])
        cos = V @ V.T
        pair = cos[np.triu_indices(4, k=1)]
        scores.append(float(pair.mean()))

    by_word = defaultdict(list)
    masks = []
    for ci, idxs in enumerate(candidates):
        mask = sum(1 << i for i in idxs)
        masks.append(mask)
        for i in idxs:
            by_word[i].append(ci)

    full = (1 << 16) - 1

    @lru_cache(None)
    def best(remaining: int):
        if remaining == 0:
            return 0.0, ()
        first_bit = remaining & -remaining
        first = first_bit.bit_length() - 1
        best_score, best_sol = -1e18, ()
        for ci in by_word[first]:
            if (masks[ci] & remaining) != masks[ci]:
                continue
            rest_score, rest_sol = best(remaining ^ masks[ci])
            total = scores[ci] + rest_score
            if total > best_score:
                best_score = total
                best_sol = (ci,) + rest_sol
        return best_score, best_sol

    total_score, chosen = best(full)
    groups = [[words[i] for i in candidates[c]] for c in chosen]
    return groups, float(total_score)


def predict(words: list[str]):
    """Solve and pretty-print."""
    groups, score = solve(words)
    print(f"Total semantic score: {score:.4f}\n")
    for i, g in enumerate(groups, 1):
        print(f"Group {i}: {sorted(g)}")
    return groups


# ------------------------------------------------------------------
# 3. Evaluation on the full dataset
# ------------------------------------------------------------------

def evaluate(puzzle_list):
    group_hits = 0
    puzzle_hits = 0
    failures = []

    for i, p in enumerate(puzzle_list, 1):
        if i % 100 == 0 or i == len(puzzle_list):
            print(f"  evaluated {i}/{len(puzzle_list)} …", flush=True)

        predicted, _ = solve(p["words"])
        actual = [
            {w for w, c in zip(p["words"], p["colors"]) if c == col}
            for col in ["yellow", "green", "blue", "purple"]
        ]

        hits = sum(set(g) in actual for g in predicted)
        group_hits += hits
        if hits == 4:
            puzzle_hits += 1
        else:
            failures.append(p["id"])

    n = len(puzzle_list)
    print("\n=== Evaluation results (all puzzles) ===")
    print(f"Puzzles tested       : {n}")
    print(f"Group accuracy       : {group_hits / (4 * n):.3f}  ({group_hits}/{4 * n})")
    print(f"Exact puzzle accuracy: {puzzle_hits / n:.3f}  ({puzzle_hits}/{n})")

    return {
        "group_accuracy": group_hits / (4 * n),
        "puzzle_accuracy": puzzle_hits / n,
        "failures": failures,
    }


# ------------------------------------------------------------------
# 4. Run evaluation + example prediction
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ----- Accuracy on the whole dataset -----
    results = evaluate(puzzles)

    # ----- Example prediction -----
    print("\n" + "=" * 50)
    print("Example prediction:")
    example = [
        "EYE", "LASH", "LIPS", "JACKET",
        "TIE", "SENSE", "CHANGE", "BELT",
        "RING", "TASTE", "CUP", "HITCH",
        "CANNON", "TETHER", "FLAIR", "ENDS"
    ]
    predict(example)