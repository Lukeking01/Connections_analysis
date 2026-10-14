"""Simple, explainable pairwise classifier for grouping 16 puzzle words.

Train on earlier puzzles and evaluate on later puzzles. This is a baseline,
not a language model: wordplay and semantic categories remain challenging.
"""
from itertools import combinations
import difflib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
    vectors = encoder.encode(words, normalize_embeddings=True) if encoder else None
    scores = model.predict_proba(pair_features(words, vectors))[:, 1]
    return [[words[i] for i in group] for group in greedy_partition(scores)]


def compare_classifiers(puzzles, encoder=None, max_test_puzzles=80):
    """Chronological train/validation/test; compare lexical or semantic inputs."""
    n = len(puzzles)
    training, validation, testing = (puzzles[:int(n * .6)],
                                     puzzles[int(n * .6):int(n * .8)],
                                     puzzles[int(n * .8):])
    X_train, y_train = labeled_pairs(training, encoder)
    X_val, y_val = labeled_pairs(validation, encoder)
    X_test, y_test = labeled_pairs(testing, encoder)
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Random forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                                  random_state=42, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"model": name, "validation_AUC": validation_auc,
                     "test_AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])})
    comparison = pd.DataFrame(rows)
    # Choose the classifier using validation data only.
    winner = comparison.loc[comparison.validation_AUC.idxmax(), "model"]
    model = models[winner]
    evaluation = []
    for puzzle in testing[:max_test_puzzles]:
        predicted = solve_with_classifier(puzzle["words"], model, encoder)
        actual = [frozenset(w for w, c in zip(puzzle["words"], puzzle["colors"]) if c == color)
                  for color in set(puzzle["colors"])]
        correct = sum(frozenset(group) in actual for group in predicted)
        evaluation.append({"puzzle_date": puzzle["date"], "correct_groups": correct,
                           "fully_solved": int(correct == 4)})
    results = pd.DataFrame(evaluation)
    suffix = "semantic" if encoder else "lexical"
    save_table(comparison, f"{suffix}_model_comparison.csv")
    save_table(results, f"{suffix}_solver_evaluation.csv")
    return comparison, results, model
