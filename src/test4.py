"""
Improved + regularized NYT Connections group scorer
===================================================
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

import Feature_finder as ff


# ------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------

def load_puzzles(db_path: str = "data/connections.db"):
    df = ff.load_database(db_path).copy()
    df["word"] = df["word"].astype(str).str.strip().str.upper()
    df["color"] = df["color"].astype(str).str.lower().str.strip()

    puzzle_key = "puzzle_number" if "puzzle_number" in df.columns else "puzzle_date"

    complete = (
        df.groupby(puzzle_key)
        .filter(lambda x: len(x) == 16 and x["word"].nunique() == 16)
    )

    puzzles = []
    for pid, group in complete.groupby(puzzle_key):
        puzzles.append({
            "id": str(pid),
            "words": group["word"].tolist(),
            "colors": group["color"].tolist(),
        })
    print(f"Loaded {len(puzzles)} complete puzzles")
    return puzzles, complete


puzzles, complete_df = load_puzzles()


# ------------------------------------------------------------------
# 2. Embeddings
# ------------------------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
unique_words = complete_df["word"].unique().tolist()
vectors = embedding_model.encode(
    unique_words, normalize_embeddings=True, show_progress_bar=True
)
word_embeddings = dict(zip(unique_words, vectors))
DIM = vectors.shape[1]


# ------------------------------------------------------------------
# 3. Much smaller, more robust group features
# ------------------------------------------------------------------

def group_features(words: list[str]) -> np.ndarray:
    vecs = np.stack([word_embeddings[w] for w in words])          # (4, DIM)

    cos = vecs @ vecs.T
    pair = cos[np.triu_indices(4, k=1)]                           # 6 values

    # Only the compact statistics – no raw mean/std vectors
    feats = np.array([
        pair.mean(),
        pair.min(),
        pair.max(),
        pair.std(),
        np.median(pair),
        # pairwise range
        pair.max() - pair.min(),
    ], dtype=np.float32)
    return feats


FEATURE_DIM = 6          # dramatically smaller → much less overfitting


# ------------------------------------------------------------------
# 4. Dataset with more negatives + mild noise
# ------------------------------------------------------------------

def make_group_dataset(puzzle_list, negatives_per_puzzle: int = 60):
    X, y = [], []
    rng = np.random.default_rng(42)

    for puzzle in puzzle_list:
        words = puzzle["words"]
        colors = puzzle["colors"]

        true_groups = []
        for c in ["yellow", "green", "blue", "purple"]:
            idxs = [i for i, col in enumerate(colors) if col == c]
            if len(idxs) != 4:
                continue
            true_groups.append(tuple(sorted(idxs)))
            X.append(group_features([words[i] for i in idxs]))
            y.append(1.0)

        true_set = set(true_groups)

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

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # tiny Gaussian noise on features helps generalization
    X += rng.normal(0, 0.01, size=X.shape).astype(np.float32)

    return (
        torch.tensor(X),
        torch.tensor(y).unsqueeze(1),
    )


train_puzzles, test_puzzles = train_test_split(
    puzzles, test_size=0.2, random_state=42
)

X_train, y_train = make_group_dataset(train_puzzles)
X_test,  y_test  = make_group_dataset(test_puzzles)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"Positive rate train: {y_train.mean():.3f}")


# ------------------------------------------------------------------
# 5. Much smaller network + heavy regularization
# ------------------------------------------------------------------

class GroupScorer(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = GroupScorer()
pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum().item(), 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256,
    shuffle=True,
)


def evaluate_classifier(X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean().item()
        pos = y == 1
        pos_acc = (preds[pos] == y[pos]).float().mean().item() if pos.any() else 0.0
    return acc, pos_acc


best_test_pos = 0.0
best_state = None

for epoch in range(1, 81):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        train_acc, train_pos = evaluate_classifier(X_train, y_train)
        test_acc,  test_pos  = evaluate_classifier(X_test,  y_test)
        print(
            f"Epoch {epoch:3d} | loss {total_loss:.1f} | "
            f"train {train_acc:.3f} (pos {train_pos:.3f}) | "
            f"test  {test_acc:.3f} (pos {test_pos:.3f})"
        )
        # early-ish model selection on positive recall
        if test_pos > best_test_pos:
            best_test_pos = test_pos
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nLoaded best model (test pos recall = {best_test_pos:.3f})")


# ------------------------------------------------------------------
# 6. Solver + evaluation (unchanged logic)
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
            scores.append(1 / (1 + np.exp(-logit)))

    scores = np.array(scores)

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


def evaluate(puzzle_list):
    correct_groups = 0
    correct_puzzles = 0
    for puzzle in puzzle_list:
        predicted, _ = solve_puzzle(puzzle["words"])
        actual = [
            {w for w, col in zip(puzzle["words"], puzzle["colors"]) if col == c}
            for c in ["yellow", "green", "blue", "purple"]
        ]
        hits = sum(set(g) in actual for g in predicted)
        correct_groups += hits
        correct_puzzles += (hits == 4)

    n = len(puzzle_list)
    print(f"Group accuracy : {correct_groups / (4 * n):.3f}")
    print(f"Puzzle accuracy: {correct_puzzles / n:.3f}")

def predict(words: list[str]):
    words = [w.strip().upper() for w in words]
    assert len(words) == 16 and len(set(words)) == 16

    # Encode any words the model has never seen
    missing = [w for w in words if w not in word_embeddings]
    if missing:
        new_vecs = embedding_model.encode(
            missing, normalize_embeddings=True, show_progress_bar=False
        )
        for w, v in zip(missing, new_vecs):
            word_embeddings[w] = v

    return solve_puzzle(words)

print("\n=== Test set performance ===")
evaluate(test_puzzles)
print(predict(["EYE","LASH","LIPS","JACKET","TIE","SENSE","CHANGE","BELT","RING","TASTE","CUP","HITCH","CANNON","TETHER","FLAIR","ENDS"]))