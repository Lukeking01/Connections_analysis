"""Exact maximum-weight partition of 16 items into four unlabeled groups of four."""
from functools import lru_cache
from itertools import combinations
import numpy as np

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
