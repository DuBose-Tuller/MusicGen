import pytest
import numpy as np
from codebooks.distributions import jensen_shannon_divergence, cosine_similarity

def test_jensen_shannon_divergence():
    """Test JSD calculation."""
    # Test identical distributions
    p = np.array([0.5, 0.5])
    assert np.isclose(jensen_shannon_divergence(p, p), 0.0)
    
    # Test completely different distributions
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jensen_shannon_divergence(p, q) > 0.0

def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Test identical vectors
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_similarity(v, v), 1.0)
    
    # Test orthogonal vectors
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(v1, v2), 0.0)