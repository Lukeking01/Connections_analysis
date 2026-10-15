"""All PyTorch training, lexical features, evaluation and exact grouping in one place.

Only two splits: earlier puzzles train, later puzzles validate (80/20).
The optional sklearn import is for AUC scoring only, not training.
"""
import copy
import difflib
from functools import lru_cache
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .paths import save_table

PAIRS = list(combinations(range(16), 2))


@lru_cache(maxsize=8192)
def synsets(word):
    try:
        from nltk.corpus import wordnet as wn
        return frozenset(s.name() for s in wn.synsets(word.replace(' ', '_').lower()))
    except (ImportError, LookupError):
        return frozenset()

def synonym_matrix(words):
    """Binary direct synset overlap, not a learned synonym classifier."""
    sets = [synsets(w) for w in words]
    return np.asarray([[float(i != j and bool(a & b)) for j, b in enumerate(sets)]
                       for i, a in enumerate(sets)], dtype=np.float32)

def optimal_partition(pair_scores):
    scores = np.asarray(pair_scores, dtype=float)
    if scores.shape != (16, 16) or not np.isfinite(scores).all():
        raise ValueError('Expected finite 16x16 pair score matrix')
    scores = (scores + scores.T) / 2
    @lru_cache(None)
    def solve(mask):
        if mask == 0:
            return 0.0, ()
        first = (mask & -mask).bit_length() - 1
        others = [i for i in range(first + 1, 16) if mask & (1 << i)]
        best = (-float('inf'), ())
        for triple in combinations(others, 3):
            group = (first, *triple)
            remainder = mask
            for i in group:
                remainder ^= 1 << i
            value, groups = solve(remainder)
            value += sum(scores[i, j] for i, j in combinations(group, 2))
            if value > best[0]:
                best = (value, (group,) + groups)
        return best
    _, groups = solve((1 << 16) - 1)
    return [list(g) for g in groups]

class ContextualPairScorer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, heads=4, dropout=0.1,
                 synonym_branch=True):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError('hidden_dim must be divisible by heads')
        self.synonym_branch = synonym_branch
        self.project = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pair_head = nn.Sequential(nn.Linear(hidden_dim * 4 + 1 + int(synonym_branch), hidden_dim),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, embeddings, synonyms=None):
        """embeddings [batch,16,dim], synonyms [batch,16,16]; returns [batch,16,16]."""
        if embeddings.ndim != 3 or embeddings.shape[1] != 16:
            raise ValueError('Expected embeddings of shape [batch,16,dimension]')
        x = self.project(embeddings)
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + attended)
        pooled = x.mean(dim=1, keepdim=True).expand_as(x)
        a, b = x.unsqueeze(2).expand(-1,-1,16,-1), x.unsqueeze(1).expand(-1,16,-1,-1)
        context = pooled.unsqueeze(2).expand_as(a)
        cosine = F.cosine_similarity(a, b, dim=-1).unsqueeze(-1)
        features = [a*b, torch.abs(a-b), (a+b)/2, context, cosine]
        if self.synonym_branch:
            if synonyms is None:
                synonyms = embeddings.new_zeros((embeddings.shape[0],16,16))
            features.append(synonyms.unsqueeze(-1).to(dtype=embeddings.dtype))
        logits = self.pair_head(torch.cat(features, dim=-1)).squeeze(-1)
        return (logits + logits.transpose(1,2))/2
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
    features, labels = [], []
    for p in puzzles:
        features.append(pair_features(p['words'], vectors[p['date']]))
        labels.extend(int(p['colors'][i] == p['colors'][j]) for i, j in PAIRS)
    return np.vstack(features), np.asarray(labels)


def fit_pair_mlp(puzzles, encoder, max_evaluation=None, **training_kwargs):
    """Fit the PyTorch contextual pair scorer and report train/validation metrics."""
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


def build_arrays(puzzles, encoder, use_synonyms=True):
    """Embed puzzles and build the pair-label and optional synonym tensors."""
    vectors = encode_puzzles(puzzles, encoder)
    x = np.stack([vectors[p['date']] for p in puzzles]).astype('float32')
    y = np.stack([np.equal.outer(p['colors'], p['colors']) for p in puzzles]).astype('float32')
    syn = (np.stack([synonym_matrix(p['words']) for p in puzzles]).astype('float32')
           if use_synonyms else np.zeros((len(puzzles), 16, 16), dtype='float32'))
    return x, syn, y


def train_solver(puzzles, encoder, epochs=30, patience=5, lr=1e-3,
                 hidden_dim=128, batch_size=16, seed=42, device='cpu', use_synonyms=True):
    ordered = sorted(puzzles, key=lambda p: p['date'])
    train, val = split_puzzles(ordered)
    if not train or not val:
        raise ValueError('Need enough puzzles for nonempty chronological train/validation splits')
    torch.manual_seed(seed)
    x, s, y = build_arrays(ordered, encoder, use_synonyms)
    tensors = [torch.from_numpy(a).to(device) for a in (x,s,y)]
    n_train, n_val = len(train), len(val)
    model = ContextualPairScorer(x.shape[-1], hidden_dim=hidden_dim,
                                  synonym_branch=use_synonyms).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    upper = torch.triu_indices(16,16,offset=1,device=device)
    def loss(indices):
        logits = model(tensors[0][indices], tensors[1][indices])
        target = tensors[2][indices]
        return F.binary_cross_entropy_with_logits(logits[:,upper[0],upper[1]],target[:,upper[0],upper[1]])
    best, weights, waiting = float('inf'), None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(n_train), max(1,int(np.ceil(n_train/batch_size)))):
            optimizer.zero_grad()
            value = loss(ids.tolist())
            value.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            current = float(loss(slice(n_train,n_train+n_val)).item())
        if current < best - 1e-5:
            best, weights, waiting = current, copy.deepcopy(model.state_dict()), 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        scores = model(tensors[0],tensors[1]).sigmoid().cpu().numpy()
    return model, {'train':train,'validation':val,
                   'scores':scores,'labels':y,'best_validation_loss':best,
                   'epochs_run':epoch+1}


def evaluate_solver(result):
    """Pair AUC and exact 4-group puzzle accuracy on each chronological split."""
    from sklearn.metrics import roc_auc_score
    from itertools import combinations
    output = []
    offset = 0
    pairs = list(combinations(range(16),2))
    for name in ('train','validation'):
        puzzles = result[name]
        correct, truth = [], []
        for k,p in enumerate(puzzles):
            i = offset+k
            predicted = optimal_partition(result['scores'][i])
            expected = [frozenset(j for j,c in enumerate(p['colors']) if c==color) for color in dict.fromkeys(p['colors'])]
            correct.append({frozenset(g) for g in predicted} == set(expected))
            truth.extend((result['labels'][i,a,b],result['scores'][i,a,b]) for a,b in pairs)
        labels, probs = np.array(truth).T
        output.append({'split':name,'puzzles':len(puzzles),'pair_auc':float(roc_auc_score(labels,probs)),
                       'exact_puzzle_accuracy':float(np.mean(correct))})
        offset += len(puzzles)
    return output

def compare_classifiers(puzzles, encoder=None, max_validation_puzzles=None, **kwargs):
    """Historical notebook spelling; still trains exclusively with PyTorch."""
    if encoder is None:
        from .clustering import load_encoder
        encoder = load_encoder()
    model, metrics, evaluation = fit_pair_mlp(puzzles, encoder, max_evaluation=max_validation_puzzles, **kwargs)
    return metrics, evaluation, model

solve_with_classifier = solve_with_mlp
