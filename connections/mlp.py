"""Small neural-network extensions: word-pair solver and difficulty prediction.

Sentence Transformers supplies fixed word embeddings. Only scikit-learn MLPs
are trained; the pretrained language model is never fine-tuned.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .supervised import PAIRS, pair_features, greedy_partition
from .paths import save_table


def split_puzzles(puzzles):
    """Chronological 60/20/20 split; never split word pairs independently."""
    ordered = sorted(puzzles, key=lambda p: p["date"])
    n = len(ordered)
    return ordered[:int(.6*n)], ordered[int(.6*n):int(.8*n)], ordered[int(.8*n):]


def encode_puzzles(puzzles, encoder):
    """Encode each distinct word once; avoid repeated transformer calls."""
    vocabulary = list(dict.fromkeys(w for p in puzzles for w in p["words"]))
    vectors = encoder.encode(vocabulary, batch_size=128, normalize_embeddings=True,
                             show_progress_bar=True)
    lookup = dict(zip(vocabulary, np.asarray(vectors)))
    return {p["date"]: np.stack([lookup[w] for w in p["words"]]) for p in puzzles}


def pair_dataset(puzzles, vectors):
    """Lexical and semantic features, with known same-group labels."""
    features, labels = [], []
    for p in puzzles:
        features.append(pair_features(p["words"], vectors[p["date"]]))
        labels.extend(int(p["colors"][i] == p["colors"][j]) for i, j in PAIRS)
    return np.vstack(features), np.asarray(labels)


def fit_pair_mlp(puzzles, encoder, max_evaluation=80):
    """Train on early puzzles, validate, then evaluate on held-out later puzzles."""
    train, val, test = split_puzzles(puzzles)
    vectors = encode_puzzles(puzzles, encoder)
    x_train, y_train = pair_dataset(train, vectors)
    x_val, y_val = pair_dataset(val, vectors)
    x_test, y_test = pair_dataset(test, vectors)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0,
                      max_iter=200, early_stopping=True, random_state=42),
    )
    model.fit(x_train, y_train)
    metrics = pd.DataFrame([{
        "model": "MLP (lexical + semantic)",
        "validation_AUC": roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]),
        "test_AUC": roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
        "training_puzzles": len(train), "validation_puzzles": len(val),
        "test_puzzles": len(test),
    }])
    save_table(metrics, "mlp_pair_metrics.csv")

    results = []
    for p in test[:max_evaluation]:
        scores = model.predict_proba(pair_features(p["words"], vectors[p["date"]]))[:, 1]
        predicted = greedy_partition(scores)
        actual = {frozenset(i for i, c in enumerate(p["colors"]) if c == color)
                  for color in set(p["colors"])}
        correct = sum(frozenset(group) in actual for group in predicted)
        results.append({"puzzle_date": p["date"], "correct_groups": correct,
                        "fully_solved": int(correct == 4)})
    evaluation = pd.DataFrame(results)
    save_table(evaluation, "mlp_solver_evaluation.csv")
    return model, metrics, evaluation


def solve_with_mlp(words, model, encoder):
    """Predict four disjoint groups of four from a fresh set of 16 words."""
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError("Please provide exactly 16 distinct words.")
    embeddings = encoder.encode(words, normalize_embeddings=True)
    scores = model.predict_proba(pair_features(words, embeddings))[:, 1]
    return [[words[i] for i in group] for group in greedy_partition(scores)]


def difficulty_features(puzzles, vectors):
    """Summarize a puzzle without using its solution or difficulty as features.

    Pairwise lexical/semantic averages and spreads describe the 16 words.
    No true group assignments enter these predictors.
    """
    rows = []
    for p in puzzles:
        pair_matrix = pair_features(p["words"], vectors[p["date"]])
        row = {"puzzle_date": p["date"], "difficulty": p.get("difficulty", np.nan)}
        for i, name in enumerate(["spelling_overlap", "length_difference",
                                  "shared_letters", "same_prefix", "same_suffix",
                                  "semantic_cosine"]):
            row[f"{name}_mean"] = float(pair_matrix[:, i].mean())
            row[f"{name}_std"] = float(pair_matrix[:, i].std())
        lengths = np.array([len(w) for w in p["words"]], dtype=float)
        row["mean_word_length"] = lengths.mean()
        row["sd_word_length"] = lengths.std()
        rows.append(row)
    return pd.DataFrame(rows)


def fit_difficulty_mlp(puzzles, encoder):
    """Predict published difficulty using word-only features and a mean baseline."""
    ordered = sorted(puzzles, key=lambda p: p["date"])
    vectors = encode_puzzles(ordered, encoder)
    frame = difficulty_features(ordered, vectors)
    save_table(frame, "mlp_difficulty_features.csv")
    # Split by date BEFORE removing unrated puzzles.
    train, val, test = split_puzzles(ordered)
    dates = [{p["date"] for p in part} for part in (train, val, test)]
    subsets = [frame.loc[frame.puzzle_date.isin(part)].dropna(subset=["difficulty"])
               for part in dates]
    columns = [c for c in frame if c not in ("puzzle_date", "difficulty")]
    a, b, c = subsets
    if min(map(len, subsets)) == 0:
        raise ValueError("Each chronological split needs at least one rated puzzle.")
    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(16,), alpha=10.0, max_iter=500,
        early_stopping=True, random_state=42))
    baseline = DummyRegressor(strategy="mean")
    model.fit(a[columns], a.difficulty)
    baseline.fit(a[columns], a.difficulty)
    rows = []
    for name, predictor in (("MLP (lexical + semantic)", model),
                            ("Training-mean baseline", baseline)):
        rows.append({"model": name,
                     "validation_MAE": mean_absolute_error(b.difficulty, predictor.predict(b[columns])),
                     "test_MAE": mean_absolute_error(c.difficulty, predictor.predict(c[columns])),
                     "train_rated": len(a), "validation_rated": len(b), "test_rated": len(c)})
    metrics = pd.DataFrame(rows)
    save_table(metrics, "mlp_difficulty_metrics.csv")
    return model, metrics


def predict_difficulty(words, model, encoder):
    """Estimate difficulty for a new puzzle from its 16 words alone."""
    if len(words) != 16:
        raise ValueError("Provide exactly 16 words.")
    record = {"date": "new", "words": list(words), "difficulty": np.nan}
    embeddings = {"new": encoder.encode(words, normalize_embeddings=True)}
    features = difficulty_features([record], embeddings)
    return float(model.predict(features.drop(columns=["puzzle_date", "difficulty"]))[0])
