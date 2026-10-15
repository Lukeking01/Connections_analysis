"""Chronological puzzle-level datasets with cached frozen encoder vectors."""
import numpy as np
from ..mlp import split_puzzles, encode_puzzles
from ..features.synonyms import synonym_matrix

def build_arrays(puzzles, encoder, use_synonyms=True):
    vectors = encode_puzzles(puzzles, encoder)
    x = np.stack([vectors[p['date']] for p in puzzles]).astype('float32')
    y = np.stack([np.equal.outer(p['colors'], p['colors']) for p in puzzles]).astype('float32')
    s = np.stack([synonym_matrix(p['words']) for p in puzzles]).astype('float32') if use_synonyms else np.zeros((len(puzzles),16,16),dtype='float32')
    return x, s, y
