"""Predict Connections groups using a previously trained Scorer.

Requirements: pip install torch numpy sentence-transformers
Place this file next to connections_group_scorer.pt, or edit CHECKPOINT.
"""
import itertools
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
CHECKPOINT = Path(r'C:\Users\Administrator\connections_group_scorer.pt')

class Scorer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def group_features(embeddings):
    """embeddings: [N,4,768], normalized, in candidate combination order."""
    mean = embeddings.mean(dim=1)
    spread = ((embeddings - mean[:, None]) ** 2).mean(dim=1)
    pairs = torch.stack([
        (embeddings[:, i] * embeddings[:, j]).sum(dim=-1)
        for i, j in itertools.combinations(range(4), 2)
    ], dim=1)
    return torch.cat([mean, spread, pairs.sort(dim=1).values], dim=1)


def load_models():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Missing trained checkpoint: {CHECKPOINT}\n'
                                'Run train_connections_any_group.py first, then copy the .pt file here.')
    encoder = SentenceTransformer(MODEL_NAME)
    dimension = encoder.get_sentence_embedding_dimension() * 2 + 6
    scorer = Scorer(dimension).to(DEVICE)
    
    print("Checkpoint:", CHECKPOINT.resolve())
    print("File size:", CHECKPOINT.stat().st_size, "bytes")

    with open(CHECKPOINT, "rb") as f:
        print("File header:", f.read(16))
    
    scorer.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))
    scorer.eval()
    return encoder, scorer


def predict(words, encoder, scorer, top_k=5):
    if len(words) != 16:
        raise ValueError(f'Expected exactly 16 words, got {len(words)}.')
    if any(not w.strip() for w in words):
        raise ValueError('Every word must be nonempty.')

    vectors = encoder.encode(words, normalize_embeddings=True, convert_to_numpy=True)
    emb = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=DEVICE)
    combos = list(itertools.combinations(range(16), 4))
    with torch.no_grad():
        score_parts = []
        for start in range(0, len(combos), 256):
            ids = torch.tensor(combos[start:start + 256], device=DEVICE)
            x = group_features(emb[ids])
            score_parts.append(scorer(x).cpu())
        scores = torch.cat(score_parts)

    ranked = torch.argsort(scores, descending=True)[:top_k].tolist()
    return [([words[i] for i in combos[j]], float(scores[j])) for j in ranked]


def main():
    encoder, scorer = load_models()
    print('Enter 16 words separated by commas. Type quit to exit.')
    while True:
        raw = input('\nWords: ').strip()
        if raw.lower() in ('quit', 'exit', 'q'):
            break
        words = [w.strip() for w in raw.split(',')]
        try:
            results = predict(words, encoder, scorer)
            print('\nTop suggested groups:')
            for rank, (group, score) in enumerate(results, 1):
                print(f'{rank}. {group}  (score={score:.3f})')
            print('Scores are relative ranking values, not calibrated probabilities.')
        except ValueError as exc:
            print(f'Input error: {exc}')


if __name__ == '__main__':
    main()
