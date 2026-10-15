# pip install pandas numpy torch sentence-transformers scikit-learn
import itertools
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

def connection_solver():

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA = r"data/raw/connections.csv"
    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
    EPOCHS, NEGATIVES, LR = 60, 48, 1e-3
    
    # Load word strings without interpreting "NA" as missing.
    frame = pd.read_csv(DATA, keep_default_na=False).sort_values(
        ['puzzle_date', 'color', 'position']
    )
    frame = frame[frame['puzzle_date'] != '2024-04-01']
    
    COLORS = ('yellow', 'green', 'blue', 'purple')
    puzzles = []
    for date, sub in frame.groupby('puzzle_date', sort=True):
        if len(sub) != 16 or any((sub.color == color).sum() != 4 for color in COLORS):
            continue
        if sub.word.eq('').any():
            print(f'Skipping {date}: empty word')
            continue
        words = sub.word.astype(str).tolist()
        # Positions are taken directly from the sorted rows, avoiding words.index() ambiguity.
        groups = {
            color: tuple(i for i, c in enumerate(sub.color.tolist()) if c == color)
            for color in COLORS
        }
        puzzles.append((date, words, groups))
    
    print(f'{len(puzzles)} complete puzzles')
    
    # Chronological split avoids training on future puzzles.
    n = len(puzzles)
    train, val, test = puzzles[:int(.7*n)], puzzles[int(.7*n):int(.85*n)], puzzles[int(.85*n):]
    unique_words = sorted({w for _, words, _ in puzzles for w in words})
    encoder = SentenceTransformer(MODEL_NAME)
    cache = Path('mpnet_word_embeddings.npz')
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved['words'], np.array(unique_words)):
            vectors = saved['vectors']
        else:
            vectors = encoder.encode(unique_words, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    else:
        vectors = encoder.encode(unique_words, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    if not cache.exists() or not np.array_equal(np.load(cache)['words'], np.array(unique_words)):
        np.savez_compressed(cache, words=np.array(unique_words), vectors=vectors)
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    emb = torch.tensor(vectors, dtype=torch.float32, device=DEVICE)
    combinations = np.array(list(itertools.combinations(range(16), 4)), dtype=np.int64)
    comb_t = torch.tensor(combinations, device=DEVICE)
    
    # Permutation-invariant features: mean, dispersion, and sorted pair similarities.
    def features(word_ids, combos):
        e = emb[word_ids[combos]]  # [N, 4, 768]
        mean = e.mean(dim=1)
        spread = ((e - mean[:, None]) ** 2).mean(dim=1)
        pairs = torch.stack([(e[:, i] * e[:, j]).sum(-1) for i, j in itertools.combinations(range(4), 2)], dim=1)
        return torch.cat([mean, spread, pairs.sort(dim=1).values], dim=1)
    
    # Precompute candidate features once: only the small scorer is trained.
    all_features = []
    for _, words, _ in puzzles:
        ids = torch.tensor([word_to_idx[w] for w in words], device=DEVICE)
        with torch.no_grad():
            chunks = [features(ids, c) for c in comb_t.split(256)]
            all_features.append(torch.cat(chunks).cpu())
    all_features = torch.stack(all_features)
    # Every puzzle has FOUR correct combination indices, one per color.
    combination_to_index = {tuple(c): i for i, c in enumerate(combinations)}
    labels = torch.tensor([
        [combination_to_index[groups[color]] for color in COLORS]
        for _, _, groups in puzzles
    ], dtype=torch.long)
    
    class Scorer(nn.Module):
        def __init__(self, dimension):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dimension, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
    
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
    
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
    
                nn.Linear(128, 64),
                nn.ReLU(),
    
                nn.Linear(64, 1)
            )
    
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    model = Scorer(all_features.shape[-1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    train_indices = np.arange(len(train))
    val_indices = np.arange(len(train), len(train)+len(val))
    test_indices = np.arange(len(train)+len(val), len(puzzles))
    
    def evaluate(indices):
        model.eval()
        any_correct, yellow_correct, ranks = 0, 0, []
        color_counts = {color: 0 for color in COLORS}
        with torch.no_grad():
            for i in indices:
                scores = torch.cat([model(x.to(DEVICE)) for x in all_features[i].split(256)])
                target_indices = labels[i].tolist()
                pred = scores.argmax().item()
                any_correct += int(pred in target_indices)
                yellow_correct += int(pred == target_indices[0])
                if pred in target_indices:
                    color_counts[COLORS[target_indices.index(pred)]] += 1
                # Best (lowest) rank among the four real groups.
                best_true_score = scores[labels[i].to(DEVICE)].max()
                ranks.append(1 + (scores > best_true_score).sum().item())
        return (any_correct / len(indices), yellow_correct / len(indices),
                float(np.median(ranks)), color_counts)
    
    best, patience = -1, 0
    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(train_indices)
        losses = []
        for i in train_indices:
            positives = labels[i].numpy()  # four real category groups
            negative_pool = np.setdiff1d(np.arange(len(combinations)), positives)
            negatives = np.random.choice(negative_pool, size=NEGATIVES, replace=False)
            chosen = np.r_[positives, negatives]
            x = all_features[i, chosen].to(DEVICE)
            scores = model(x)
    
            # Multi-positive objective: reward probability assigned to ANY real group.
            # -log(sum_{real groups} softmax(scores))
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[:4], dim=0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        accuracy, yellow_accuracy, rank, _ = evaluate(val_indices)
        print(f'Epoch {epoch+1:02d}: loss={np.mean(losses):.3f} '
              f'val_any={accuracy:.3%} val_yellow={yellow_accuracy:.3%} '
              f'median_best_group_rank={rank:.0f}')
        if accuracy > best:
            best, patience = accuracy, 0
            torch.save(model.state_dict(), 'connections_group_scorer.pt')
        else:
            patience += 1
            if patience >= 10:
                print('Early stopping'); break
    
    model.load_state_dict(torch.load('connections_group_scorer.pt', map_location=DEVICE, weights_only=True))
    for name, indices in [('TRAIN', np.arange(len(train))),
                          ('VALIDATION', val_indices), ('TEST', test_indices)]:
        accuracy, yellow_accuracy, rank, color_counts = evaluate(indices)
        print(f'{name}: any-group accuracy={accuracy:.2%}, '
              f'yellow accuracy={yellow_accuracy:.2%}, '
              f'median best-group rank={rank:.0f}/{len(combinations)}')
        print('  Correct predictions by color:', color_counts)
    
    model.eval()
    for i in test_indices[:5]:
        with torch.no_grad():
            scores = torch.cat([model(x.to(DEVICE)) for x in all_features[i].split(256)])
        date, words, groups = puzzles[i]
        pred_idx = scores.argmax().item()
        predicted = combinations[pred_idx]
        matched_color = next((c for c in COLORS
                              if pred_idx == combination_to_index[groups[c]]), None)
        print(f'\n{date}\nPredicted: {[words[j] for j in predicted]}')
        print(f'Result: {matched_color.upper() if matched_color else "INCORRECT"}')
        for color in COLORS:
            print(f'  {color.upper():6} {[words[j] for j in groups[color]]}')

