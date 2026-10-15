"""Training with chronological validation and early stopping; no test-set tuning."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from ..mlp import split_puzzles
from ..models.torch_solver import ContextualPairScorer
from .datasets import build_arrays
from ..solvers.group_optimizer import optimal_partition


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
