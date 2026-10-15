"""Optional WordNet lexical relations; no labels or future puzzles are consulted."""
from functools import lru_cache
import numpy as np

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
