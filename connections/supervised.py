"""Simple, explainable pairwise classifier for grouping 16 puzzle words.

Train on earlier puzzles and evaluate on later puzzles. This is a baseline,
not a language model: wordplay and semantic categories remain challenging.
"""
from itertools import combinations
import difflib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .paths import save_table

PAIRS = list(combinations(range(16), 2))


def pair_features(words, embeddings=None):
    """One row per pair: spelling overlap, lengths and optional semantic cosine."""
    vectors = None
    if embeddings is not None:
        vectors = np.asarray(embeddings, dtype=float)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    rows = []
    for i, j in PAIRS:
        a, b = words[i].lower(), words[j].lower()
        row = [difflib.SequenceMatcher(None, a, b).ratio(),
               abs(len(a) - len(b)),
               len(set(a) & set(b)),
               float(a[:2] == b[:2]),
               float(a[-2:] == b[-2:])]
        if vectors is not None:
            row.append(float(vectors[i] @ vectors[j]))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def labeled_pairs(puzzles, encoder=None):
    """Labels are 1 for words in the same correct group, otherwise 0."""
    X, y = [], []
    for puzzle in puzzles:
        vectors = encoder.encode(puzzle["words"], normalize_embeddings=True) if encoder else None
        X.append(pair_features(puzzle["words"], vectors))
        y.extend(int(puzzle["colors"][i] == puzzle["colors"][j]) for i, j in PAIRS)
    return np.vstack(X), np.asarray(y)


def greedy_partition(probabilities):
    """Choose four non-overlapping groups of four from pair compatibility."""
    scores = np.zeros((16, 16))
    for (i, j), value in zip(PAIRS, probabilities):
        scores[i, j] = scores[j, i] = value
    remaining = set(range(16))
    result = []
    while remaining:
        # Anchor the first unused word to avoid duplicate group candidates.
        anchor = min(remaining)
        candidates = combinations(sorted(remaining - {anchor}), 3)
        best = max(candidates, key=lambda triple: sum(
            scores[i, j] for i, j in combinations((anchor, *triple), 2)))
        group = (anchor, *best)
        result.append(group)
        remaining.difference_update(group)
    return result


def solve_with_classifier(words, model, encoder=None):
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError("Provide exactly 16 distinct words.")
    from .models.torch_solver import ContextualPairScorer
    if isinstance(model, ContextualPairScorer):
        from .mlp import solve_with_mlp
        if encoder is None:
            from .clustering import load_encoder
            encoder = load_encoder()
        return solve_with_mlp(words, model, encoder)
    vectors = encoder.encode(words, normalize_embeddings=True) if encoder else None
    scores = model.predict_proba(pair_features(words, vectors))[:, 1]
    return [[words[i] for i in group] for group in greedy_partition(scores)]


def compare_classifiers(puzzles, encoder=None, max_validation_puzzles=None, **kwargs):
    """Compatibility wrapper: train ONLY the PyTorch contextual solver.

    Returns (metrics, validation_puzzles, trained_model) in the legacy order.
    An embedding encoder is required; loaded automatically if omitted.
    """
    from .mlp import fit_pair_mlp
    if encoder is None:
        from .clustering import load_encoder
        encoder = load_encoder()
    model, metrics, results = fit_pair_mlp(
        puzzles, encoder, max_evaluation=max_validation_puzzles, **kwargs
    )
    return metrics, results, model
