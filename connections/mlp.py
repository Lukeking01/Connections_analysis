"""PyTorch MLP entry points for puzzle grouping and difficulty regression.

The previous sklearn MLP implementations have been replaced. Public function
names are retained for compatibility; sklearn remains for metrics/baselines.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from .paths import save_table


def split_puzzles(puzzles, train_fraction=0.8):
    """Chronological puzzle-level train/validation split; no test split."""
    ordered = sorted(puzzles, key=lambda p: p['date'])
    cutoff = int(len(ordered) * train_fraction)
    if cutoff < 1 or cutoff >= len(ordered):
        raise ValueError('At least two puzzles are required for train/validation')
    return ordered[:cutoff], ordered[cutoff:]


def encode_puzzles(puzzles, encoder):
    vocabulary = list(dict.fromkeys(w for p in puzzles for w in p['words']))
    vectors = encoder.encode(vocabulary, batch_size=128, normalize_embeddings=True,
                             show_progress_bar=True)
    lookup = dict(zip(vocabulary, np.asarray(vectors)))
    return {p['date']: np.stack([lookup[w] for w in p['words']]) for p in puzzles}


def pair_dataset(puzzles, vectors):
    from .supervised import PAIRS, pair_features
    features, labels = [], []
    for p in puzzles:
        features.append(pair_features(p['words'], vectors[p['date']]))
        labels.extend(int(p['colors'][i] == p['colors'][j]) for i, j in PAIRS)
    return np.vstack(features), np.asarray(labels)


def fit_pair_mlp(puzzles, encoder, max_evaluation=None, **training_kwargs):
    """Fit the PyTorch contextual pair scorer and report train/validation metrics."""
    from .training.train import train_solver, evaluate_solver
    from .solvers.group_optimizer import optimal_partition
    model, result = train_solver(puzzles, encoder, **training_kwargs)
    summary = {row['split']: row for row in evaluate_solver(result)}
    metrics = pd.DataFrame([{'model': 'PyTorch contextual attention + pooling',
        'train_pair_auc': summary['train']['pair_auc'],
        'validation_pair_auc': summary['validation']['pair_auc'],
        'train_exact_puzzle_accuracy': summary['train']['exact_puzzle_accuracy'],
        'validation_exact_puzzle_accuracy': summary['validation']['exact_puzzle_accuracy'],
        'training_puzzles': len(result['train']), 'validation_puzzles': len(result['validation'])}])
    save_table(metrics, 'mlp_pair_metrics.csv')
    rows = []
    offset = len(result['train'])
    for i, puzzle in enumerate(result['validation'][:max_evaluation]):
        predicted = optimal_partition(result['scores'][offset+i])
        actual = {frozenset(j for j,c in enumerate(puzzle['colors']) if c==color)
                  for color in set(puzzle['colors'])}
        correct = sum(frozenset(group) in actual for group in predicted)
        rows.append({'puzzle_date': puzzle['date'], 'correct_groups': correct,
                     'fully_solved': int(correct == 4)})
    evaluation = pd.DataFrame(rows)
    save_table(evaluation, 'mlp_validation_puzzles.csv')
    return model, metrics, evaluation


def solve_with_mlp(words, model, encoder, use_synonyms=None):
    """Solve 16 distinct words using the trained contextual PyTorch model."""
    from .features.synonyms import synonym_matrix
    from .solvers.group_optimizer import optimal_partition
    if len(words) != 16 or len(set(words)) != 16:
        raise ValueError('Provide exactly 16 distinct words.')
    device = next(model.parameters()).device
    embeddings = torch.as_tensor(np.asarray(encoder.encode(words, normalize_embeddings=True)),
                                 dtype=torch.float32, device=device).unsqueeze(0)
    if use_synonyms is None:
        use_synonyms = model.synonym_branch
    synonyms = torch.as_tensor(synonym_matrix(words) if use_synonyms else np.zeros((16,16)),
                               dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        scores = model(embeddings, synonyms).sigmoid()[0].cpu().numpy()
    return [[words[i] for i in group] for group in optimal_partition(scores)]


def difficulty_features(puzzles, vectors):
    """Word-only difficulty features, never solution-derived features."""
    from .supervised import pair_features
    rows = []
    for p in puzzles:
        pair_matrix = pair_features(p['words'], vectors[p['date']])
        row = {'puzzle_date': p['date'], 'difficulty': p.get('difficulty', np.nan)}
        for i, name in enumerate(['spelling_overlap', 'length_difference',
                                  'shared_letters', 'same_prefix', 'same_suffix',
                                  'semantic_cosine']):
            row[f'{name}_mean'] = float(pair_matrix[:, i].mean())
            row[f'{name}_std'] = float(pair_matrix[:, i].std())
        lengths = np.array([len(w) for w in p['words']], dtype=float)
        row['mean_word_length'] = lengths.mean()
        row['sd_word_length'] = lengths.std()
        rows.append(row)
    return pd.DataFrame(rows)


class TorchDifficultyRegressor(nn.Module):
    """Standardized tabular features followed by a PyTorch regression MLP."""
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(),
                                     nn.Linear(16, 1))
        self.register_buffer('feature_mean', torch.zeros(n_features))
        self.register_buffer('feature_std', torch.ones(n_features))
        self.feature_columns = None

    def forward(self, x):
        return self.network((x - self.feature_mean) / self.feature_std).squeeze(-1)

    def predict(self, frame):
        self.eval()
        if isinstance(frame, pd.DataFrame):
            frame = frame[self.feature_columns].to_numpy()
        with torch.no_grad():
            return self(torch.as_tensor(np.asarray(frame), dtype=torch.float32)).numpy()


def fit_difficulty_mlp(puzzles, encoder, epochs=300, patience=25, lr=1e-3, seed=42):
    """Fit PyTorch regressor with chronological validation-based early stopping."""
    from sklearn.metrics import mean_absolute_error
    import copy
    torch.manual_seed(seed)
    ordered = sorted(puzzles, key=lambda p: p['date'])
    vectors = encode_puzzles(ordered, encoder)
    frame = difficulty_features(ordered, vectors)
    save_table(frame, 'mlp_difficulty_features.csv')
    parts = split_puzzles(ordered)
    dates = [{p['date'] for p in part} for part in parts]
    subsets = [frame.loc[frame.puzzle_date.isin(d)].dropna(subset=['difficulty']) for d in dates]
    if min(map(len, subsets)) == 0:
        raise ValueError('Training and validation both need rated puzzles.')
    columns = [c for c in frame if c not in ('puzzle_date','difficulty')]
    a,b = subsets
    arrays = [torch.tensor(s[columns].to_numpy(dtype='float32')) for s in subsets]
    targets = [torch.tensor(s.difficulty.to_numpy(dtype='float32')) for s in subsets]
    model = TorchDifficultyRegressor(len(columns))
    model.feature_columns = columns
    with torch.no_grad():
        model.feature_mean.copy_(arrays[0].mean(0))
        model.feature_std.copy_(arrays[0].std(0, unbiased=False).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, state, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(arrays[0]), targets[0])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(arrays[1]), targets[1]).item()
        if val_loss < best - 1e-6:
            best, state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    baseline = float(targets[0].mean())
    rows = []
    for name, predictions in [('PyTorch difficulty MLP', model.predict(b[columns])),
                              ('Training-mean baseline', np.full(len(b), baseline))]:
        rows.append({'model': name, 'validation_MAE': float(np.mean(np.abs(b.difficulty.to_numpy()-predictions))),
                     'train_rated': len(a), 'validation_rated': len(b)})
    metrics = pd.DataFrame(rows)
    save_table(metrics, 'mlp_difficulty_metrics.csv')
    return model, metrics


def predict_difficulty(words, model, encoder):
    if len(words) != 16:
        raise ValueError('Provide exactly 16 words.')
    record = {'date':'new','words':list(words),'difficulty':np.nan}
    vectors = {'new':encoder.encode(words, normalize_embeddings=True)}
    frame = difficulty_features([record], vectors)
    return float(model.predict(frame)[0])
