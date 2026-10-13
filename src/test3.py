
"""
Improved NYT Connections solver
===============================
- Scores groups of 4 directly (instead of only pairs)
- Stronger features + neural net
- Exact-cover search at inference time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from collections import defaultdict
from functools import lru_cache
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------------
# 1. Data loading (same as your code)
# ------------------------------------------------------------------

data = pd.read_csv(r"data/connections.csv")
data = data.dropna(subset=["word"])
complete = data.groupby("puzzle_date").filter(lambda x: len(x) == 16)

puzzles = []
for date, group in complete.groupby("puzzle_date"):
    puzzles.append({
        "date": date,
        "words": group["word"].str.upper().tolist(),
        "colors": group["color"].str.lower().tolist(),
    })

print(f"Number of puzzles: {len(puzzles)}")


# ------------------------------------------------------------------
# 2. Embeddings
# ------------------------------------------------------------------

# You can swap to a stronger model if you want (e.g. "all-mpnet-base-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

unique_words = complete["word"].str.upper().unique().tolist()
vectors = embedding_model.encode(
    unique_words,
    normalize_embeddings=True,
    show_progress_bar=True,
)
word_embeddings = dict(zip(unique_words, vectors))
DIM = vectors.shape[1]          # 384 for MiniLM


# ------------------------------------------------------------------
# 3. Group feature construction
# ------------------------------------------------------------------

def group_features(words: list[str]) -> np.ndarray:
    """
    Features for a candidate group of 4 words.
    Returns a fixed-size vector that the network will score.
    """
    vecs = np.stack([word_embeddings[w] for w in words])          # (4, DIM)

    # Pairwise cosines
    cos = vecs @ vecs.T
    pair = cos[np.triu_indices(4, k=1)]                           # 6 values

    # Simple summary stats of the four vectors
    mean_vec = vecs.mean(axis=0)
    std_vec  = vecs.std(axis=0)

    # Compact interaction features
    feats = np.concatenate([
        pair,                         # 6
        [pair.mean(), pair.min(), pair.max(), pair.std()],  # 4
        mean_vec,                     # DIM
        std_vec,                      # DIM
    ])
    return feats.astype(np.float32)


FEATURE_DIM = 6 + 4 + DIM + DIM   # 6 + 4 + 384 + 384 = 778 for MiniLM


# ------------------------------------------------------------------
# 4. Build training examples (positives + hard negatives)
# ------------------------------------------------------------------

def make_group_dataset(puzzle_list, negatives_per_puzzle: int = 40):
    X, y = [], []
    rng = np.random.default_rng(42)

    for puzzle in puzzle_list:
        words = puzzle["words"]
        colors = puzzle["colors"]

        # True groups
        true_groups = []
        for c in ["yellow", "green", "blue", "purple"]:
            idxs = [i for i, col in enumerate(colors) if col == c]
            true_groups.append(tuple(sorted(idxs)))
            true_words = [words[i] for i in idxs]
            X.append(group_features(true_words))
            y.append(1.0)

        true_set = set(true_groups)

        # Random negatives from the same puzzle (harder than global random)
        all_combos = list(combinations(range(16), 4))
        rng.shuffle(all_combos)
        added = 0
        for combo in all_combos:
            if tuple(sorted(combo)) in true_set:
                continue
            X.append(group_features([words[i] for i in combo]))
            y.append(0.0)
            added += 1
            if added >= negatives_per_puzzle:
                break

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


train_puzzles, test_puzzles = train_test_split(
    puzzles, test_size=0.2, random_state=42
)

X_train, y_train = make_group_dataset(train_puzzles)
X_test,  y_test  = make_group_dataset(test_puzzles)

print(f"Train examples: {len(X_train)}  |  Test examples: {len(X_test)}")
print(f"Positive rate (train): {y_train.mean():.3f}")


# ------------------------------------------------------------------
# 5. Stronger neural network
# ------------------------------------------------------------------

class GroupScorer(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


model = GroupScorer()
# Positives are rarer → up-weight them
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=512,
    shuffle=True,
)


# ------------------------------------------------------------------
# 6. Training loop with simple validation
# ------------------------------------------------------------------

def evaluate_classifier(X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean().item()
        # balanced-ish metric
        pos = y == 1
        neg = y == 0
        pos_acc = (preds[pos] == y[pos]).float().mean().item() if pos.any() else 0
        neg_acc = (preds[neg] == y[neg]).float().mean().item() if neg.any() else 0
    return acc, pos_acc, neg_acc


for epoch in range(1, 151):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        train_acc, train_pos, train_neg = evaluate_classifier(X_train, y_train)
        test_acc,  test_pos,  test_neg  = evaluate_classifier(X_test,  y_test)
        print(
            f"Epoch {epoch:3d} | loss {total_loss:.1f} | "
            f"train acc {train_acc:.3f} (pos {train_pos:.3f}) | "
            f"test  acc {test_acc:.3f} (pos {test_pos:.3f})"
        )


# ------------------------------------------------------------------
# 7. Exact-cover solver that uses the group scorer
# ------------------------------------------------------------------

def solve_puzzle(words: list[str]):
    model.eval()
    words = [w.upper() for w in words]
    assert len(words) == 16 and len(set(words)) == 16

    candidates = list(combinations(range(16), 4))
    scores = []

    with torch.no_grad():
        for idxs in candidates:
            feats = group_features([words[i] for i in idxs])
            logit = model(torch.tensor(feats).unsqueeze(0)).item()
            # convert logit → probability for nicer interpretation
            scores.append(1 / (1 + np.exp(-logit)))

    scores = np.array(scores)

    # Build lookup for DP
    by_word = defaultdict(list)
    masks = []
    for cand_idx, idxs in enumerate(candidates):
        mask = sum(1 << i for i in idxs)
        masks.append(mask)
        for i in idxs:
            by_word[i].append(cand_idx)

    full_mask = (1 << 16) - 1

    @lru_cache(None)
    def best(remaining: int):
        if remaining == 0:
            return 0.0, ()
        # expand lowest bit
        first_bit = remaining & -remaining
        first = first_bit.bit_length() - 1
        best_score = -1e18
        best_sol = ()
        for cand_idx in by_word[first]:
            mask = masks[cand_idx]
            if (mask & remaining) != mask:
                continue
            rest_score, rest_sol = best(remaining ^ mask)
            total = scores[cand_idx] + rest_score
            if total > best_score:
                best_score = total
                best_sol = (cand_idx,) + rest_sol
        return best_score, best_sol

    total, chosen = best(full_mask)
    groups = [[words[i] for i in candidates[c]] for c in chosen]
    return groups, float(total)


# ------------------------------------------------------------------
# 8. Evaluation on held-out puzzles
# ------------------------------------------------------------------

def evaluate(puzzle_list):
    correct_groups = 0
    correct_puzzles = 0

    for puzzle in puzzle_list:
        predicted, _ = solve_puzzle(puzzle["words"])

        actual = []
        for c in ["yellow", "green", "blue", "purple"]:
            actual.append({
                w for w, col in zip(puzzle["words"], puzzle["colors"]) if col == c
            })

        hits = sum(set(g) in actual for g in predicted)
        correct_groups += hits
        correct_puzzles += (hits == 4)

    n = len(puzzle_list)
    print(f"Group accuracy : {correct_groups / (4 * n):.3f}")
    print(f"Puzzle accuracy: {correct_puzzles / n:.3f}")


print("\n=== Test set performance ===")
evaluate(test_puzzles)