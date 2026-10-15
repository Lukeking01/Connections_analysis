import numpy as np
import torch
from connections.ml import optimal_partition
from connections.ml import ContextualPairScorer
from connections.ml import synonym_matrix

def test_partition():
    scores = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            if i//4 == j//4: scores[i,j] = 1
    groups = optimal_partition(scores)
    assert {frozenset(g) for g in groups} == {frozenset(range(i,i+4)) for i in (0,4,8,12)}

def test_forward_backward():
    model = ContextualPairScorer(12,hidden_dim=16)
    output = model(torch.randn(2,16,12),torch.zeros(2,16,16))
    assert output.shape == (2,16,16)
    assert torch.allclose(output,output.transpose(1,2))
    output.sum().backward()
    assert model.project[0].weight.grad is not None

def test_synonyms_shape():
    assert synonym_matrix(['CAT']*16).shape == (16,16)
