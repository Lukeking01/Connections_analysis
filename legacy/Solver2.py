
"""
BERN02-style Connections solver
===============================
Uses Feature_finder.py for data loading, cleaning, Scrabble,
embeddings and group/puzzle features.
Adds logistic regression + GroupKFold CV + exact-cover solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from functools import lru_cache

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, balanced_accuracy_score

import Feature_finder as ff   # your module


# ------------------------------------------------------------------
# 1. Load & prepare data with Feature_finder
# ------------------------------------------------------------------

print("Loading database …")
words = ff.load_database("data/connections.db")
words = ff.clean_encode(words)
words = ff.add_scrabble_score(words)

print("Adding embeddings …")
words, emb_dict = ff.add_embeddings(words)

# True group-level features (one row per real category)
print("Computing true group features …")
true_group_feats = ff.get_group_features(words, emb_dict=emb_dict)

# Keep only complete 16-word puzzles
key = "puzzle_date"          # matches your group_keys / puzzle_keys
complete = (
    words.groupby(key)
    .filter(lambda x: len(x) == 16 and x["word"].nunique() == 16)
)

puzzles = []
for pid, g in complete.groupby(key):
    puzzles.append({
        "id": str(pid),
        "words": g["word"].tolist(),
        "colors": g["color"].tolist(),   # already mapped to 1-4 by clean_encode
    })
print(f"Complete puzzles: {len(puzzles)}")


# ------------------------------------------------------------------
# 2. Feature vector for an arbitrary candidate group of 4 words
#    (mirrors the columns produced by get_group_features)
# ------------------------------------------------------------------

FEATURE_COLS = [
    "mean_semantic_sim", "min_semantic_sim", "max_semantic_sim", "std_semantic_sim",
    "mean_scrabble", "max_scrabble", "min_scrabble", "sd_scrabble",
    "mean_scrabble_per_letter",
    "mean_word_length", "max_word_length", "sd_word_length",
]

def candidate_group_features(word_list: list[str], emb_dict: dict) -> np.ndarray:
    """
    Compute the same features that ff.get_group_features would produce
    for a single group of four words.
    """
    # semantic
    vectors = np.stack([emb_dict[w] for w in word_list])
    sem = ff._pairwise_cosine_stats(vectors)

    # lexical (re-use the same logic as Feature_finder)
    lengths = np.array([len(w) for w in word_list], dtype=float)
    scrabble = np.array([
        sum(ff.SCRABBLE_VALUES.get(ch, 0) for ch in w) for w in word_list
    ], dtype=float)
    per_letter = scrabble / np.maximum(lengths, 1)

    feats = {
        **sem,
        "mean_scrabble": scrabble.mean(),
        "max_scrabble": scrabble.max(),
        "min_scrabble": scrabble.min(),
        "sd_scrabble": scrabble.std(ddof=1) if len(scrabble) > 1 else 0.0,
        "mean_scrabble_per_letter": per_letter.mean(),
        "mean_word_length": lengths.mean(),
        "max_word_length": lengths.max(),
        "sd_word_length": lengths.std(ddof=1) if len(lengths) > 1 else 0.0,
    }
    return np.array([feats[c] for c in FEATURE_COLS], dtype=np.float32)


# ------------------------------------------------------------------
# 3. Build supervised dataset (positives = true groups, negatives = random)
# ------------------------------------------------------------------

def make_dataset(puzzle_list, emb_dict, negatives_per_puzzle=4):
    X, y, groups = [], [], []
    rng = np.random.default_rng(42)

    for p in puzzle_list:
        words_list = p["words"]
        colors = p["colors"]          # 1=yellow … 4=purple

        # positives
        true_idxs = []
        for c in [1, 2, 3, 4]:
            idxs = [i for i, col in enumerate(colors) if col == c]
            if len(idxs) == 4:
                true_idxs.append(tuple(sorted(idxs)))
                X.append(candidate_group_features(
                    [words_list[i] for i in idxs], emb_dict
                ))
                y.append(1)
                groups.append(p["id"])

        true_set = set(true_idxs)

        # negatives from same puzzle
        combos = list(combinations(range(16), 4))
        rng.shuffle(combos)
        added = 0
        for combo in combos:
            if tuple(sorted(combo)) in true_set:
                continue
            X.append(candidate_group_features(
                [words_list[i] for i in combo], emb_dict
            ))
            y.append(0)
            groups.append(p["id"])
            added += 1
            if added >= negatives_per_puzzle:
                break

    return np.array(X), np.array(y), np.array(groups)


print("Building training examples …")
X, y, group_ids = make_dataset(puzzles, emb_dict)
print(f"Examples: {len(X)}  |  Positive rate: {y.mean():.3f}")


# ------------------------------------------------------------------
# 4. Logistic regression + GroupKFold (BERN02 core methods)
# ------------------------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
    ),
)

cv = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    model, X, y, groups=group_ids,
    cv=cv, scoring="balanced_accuracy", n_jobs=-1
)
print(f"\n5-fold GroupKFold balanced accuracy: "
      f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

y_pred = cross_val_predict(model, X, y, groups=group_ids, cv=cv)
print("\nClassification report (CV):")
print(classification_report(y, y_pred, target_names=["not group", "true group"]))

# Final model on all data
model.fit(X, y)


# ------------------------------------------------------------------
# 5. Exact-cover solver
# ------------------------------------------------------------------

def solve(words: list[str]):
    words = [w.strip().upper() for w in words]
    assert len(words) == 16 and len(set(words)) == 16

    # encode any unseen words
    missing = [w for w in words if w not in emb_dict]
    if missing:
        encoder = ff.load_encoder()
        new_vecs = encoder.encode(missing, normalize_embeddings=True)
        for w, v in zip(missing, new_vecs):
            emb_dict[w] = v

    candidates = list(combinations(range(16), 4))
    feats = np.stack([
        candidate_group_features([words[i] for i in idxs], emb_dict)
        for idxs in candidates
    ])
    probs = model.predict_proba(feats)[:, 1]

    by_word = defaultdict(list)
    masks = []
    for ci, idxs in enumerate(candidates):
        mask = sum(1 << i for i in idxs)
        masks.append(mask)
        for i in idxs:
            by_word[i].append(ci)

    full = (1 << 16) - 1

    @lru_cache(None)
    def best(remaining):
        if remaining == 0:
            return 0.0, ()
        first = (remaining & -remaining).bit_length() - 1
        best_s, best_sol = -1e18, ()
        for ci in by_word[first]:
            if (masks[ci] & remaining) != masks[ci]:
                continue
            rs, rsol = best(remaining ^ masks[ci])
            tot = probs[ci] + rs
            if tot > best_s:
                best_s, best_sol = tot, (ci,) + rsol
        return best_s, best_sol

    _, chosen = best(full)
    groups = [[words[i] for i in candidates[c]] for c in chosen]
    return groups, float(sum(probs[c] for c in chosen))


def predict(words):
    groups, score = solve(words)
    print(f"Total score: {score:.3f}")
    for i, g in enumerate(groups, 1):
        print(f"Group {i}: {sorted(g)}")
    return groups


# ------------------------------------------------------------------
# 6. Full evaluation
# ------------------------------------------------------------------

def evaluate(puzzle_list):
    group_hits = puzzle_hits = 0
    for i, p in enumerate(puzzle_list, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(puzzle_list)} …", flush=True)
        pred, _ = solve(p["words"])
        actual = [
            {w for w, c in zip(p["words"], p["colors"]) if c == col}
            for col in [1, 2, 3, 4]
        ]
        hits = sum(set(g) in actual for g in pred)
        group_hits += hits
        puzzle_hits += (hits == 4)

    n = len(puzzle_list)
    print("\n=== Evaluation ===")
    print(f"Group accuracy : {group_hits/(4*n):.3f}")
    print(f"Puzzle accuracy: {puzzle_hits/n:.3f}")


if __name__ == "__main__":
    evaluate(puzzles)

    print("\nExample prediction:")
    predict([
        "EYE","LASH","LIPS","JACKET","TIE","SENSE","CHANGE","BELT",
        "RING","TASTE","CUP","HITCH","CANNON","TETHER","FLAIR","ENDS"
    ])