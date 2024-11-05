import pytest
import torch
from embeddings.state_manager import EmbeddingStateManager

def test_singleton_pattern():
    """Test that EmbeddingStateManager maintains singleton pattern."""
    manager1 = EmbeddingStateManager()
    manager2 = EmbeddingStateManager()
    assert manager1 is manager2

def test_set_method():
    """Test setting different embedding methods."""
    manager = EmbeddingStateManager()
    
    # Test valid methods
    valid_methods = ['last', 'first', 'mean', 'max']
    for method in valid_methods:
        manager.set_method(method)
        assert manager.current_method == method
    
    # Test invalid method
    with pytest.raises(ValueError):
        manager.set_method('invalid_method')

def test_set_embedding_from_tensor():
    """Test different embedding capture methods."""
    manager = EmbeddingStateManager()
    
    # Create a test tensor [B, T, C]
    test_tensor = torch.tensor([
        [
            [1.0, 2.0, 3.0],  # t=0
            [4.0, 5.0, 6.0],  # t=1
            [7.0, 8.0, 9.0]   # t=2
        ]
    ])
    
    # Test 'last' method
    manager.set_method('last')
    manager.set_embedding_from_tensor(test_tensor)
    expected_last = torch.tensor([7.0, 8.0, 9.0])
    assert torch.allclose(manager.get_embedding(), expected_last)
    
    # Test 'first' method
    manager.set_method('first')
    manager.set_embedding_from_tensor(test_tensor)
    expected_first = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(manager.get_embedding(), expected_first)
    
    # Test 'mean' method
    manager.set_method('mean')
    manager.set_embedding_from_tensor(test_tensor)
    expected_mean = torch.tensor([4.0, 5.0, 6.0])
    assert torch.allclose(manager.get_embedding(), expected_mean)