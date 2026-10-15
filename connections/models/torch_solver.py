"""Puzzle-aware pair scorer with optional synonym features and attention pooling."""
import torch
from torch import nn
import torch.nn.functional as F

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
