import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from tqdm import tqdm

class RatingsClassifier:
    """A classifier for multi-dimensional rating predictions using PyTorch for GPU acceleration.
    
    This classifier predicts ratings across multiple categories (e.g., dance, heal, baby, love),
    where each category can have a discrete rating (e.g., 1-4).
    """
    
    def __init__(self, n_categories: int = 4, n_ratings: int = 4, learning_rate: float = 0.01, 
                 max_iter: int = 100, tol: float = 1e-4, l1_penalty: float = 0.0,
                 batch_size: int = 1024, device: Optional[str] = None):
        """Initialize the classifier.
        
        Args:
            n_categories: Number of categories to rate
            n_ratings: Number of possible ratings per category
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of training iterations
            tol: Tolerance for stopping criterion
            l1_penalty: L1 regularization strength
            batch_size: Batch size for mini-batch gradient descent (0 for full batch)
            device: Device to use ('cuda', 'cpu', or None to auto-select)
        """
        self.n_categories = n_categories
        self.n_ratings = n_ratings
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l1_penalty = l1_penalty
        self.batch_size = batch_size
        
        # Auto-select device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.weights: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        
        print(f"Using device: {self.device}")

    def _create_target_matrix(self, y: torch.Tensor) -> torch.Tensor:
        """Create target matrix from raw ratings.
        
        Args:
            y: Rating values of shape (n_samples, n_categories) with values in [1, n_ratings]
            
        Returns:
            Target matrix of shape (n_samples, n_categories, n_ratings)
        """
        n_samples, n_cats = y.shape
        
        # Initialize target matrix
        targets = torch.zeros(n_samples, self.n_categories, self.n_ratings, device=self.device)
        
        # Convert ratings to 0-based index and clamp to valid range
        y_idx = torch.clamp(y - 1, 0, self.n_ratings - 1).long()
        
        # Create index arrays for advanced indexing
        sample_idx = torch.arange(n_samples, device=self.device).unsqueeze(1).expand(-1, n_cats)
        cat_idx = torch.arange(n_cats, device=self.device).unsqueeze(0).expand(n_samples, -1)
        
        # Set 1s using advanced indexing
        targets[sample_idx, cat_idx, y_idx] = 1
                
        return targets

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], 
            verbose: bool = False) -> 'RatingsClassifier':
        """Fit the model using mini-batch gradient descent with L1 regularization.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target ratings of shape (n_samples, n_categories) with values in [1, n_ratings]
            verbose: Whether to print progress
            
        Returns:
            self: The fitted classifier
        """
        # Convert numpy arrays to PyTorch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(device=self.device, dtype=torch.float32)
            
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(device=self.device, dtype=torch.float32)
        
        n_samples, n_features = X.shape
        n_outputs = self.n_categories * self.n_ratings
        
        # Initialize weights and bias if not already initialized
        if self.weights is None:
            self.weights = torch.randn(n_features, n_outputs, device=self.device) * 0.01
            self.weights.requires_grad = True
        
        if self.bias is None:
            self.bias = torch.zeros(n_outputs, device=self.device)
            self.bias.requires_grad = True
        
        # Create target matrix once
        targets = self._create_target_matrix(y)
        
        # Use mini-batches if batch_size > 0, otherwise use full batch
        batch_size = self.batch_size if self.batch_size > 0 else n_samples
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Setup for tracking progress
        prev_loss = float('inf')
        iterator = tqdm(range(self.max_iter)) if verbose else range(self.max_iter)
        
        # Create dataset and dataloader for efficient batch processing
        dataset = torch.utils.data.TensorDataset(X, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=False, num_workers=0  # Avoid extra CPU overhead
        )
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
        
        for iteration in iterator:
            # Track the total loss for this epoch
            total_loss = 0.0
            
            # Process mini-batches
            for X_batch, targets_batch in dataloader:
                batch_samples = X_batch.size(0)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits = F.linear(X_batch, self.weights.t(), self.bias)
                logits_reshaped = logits.view(batch_samples, self.n_categories, self.n_ratings)
                
                # Compute log probabilities with log_softmax for numerical stability
                log_probs = F.log_softmax(logits_reshaped, dim=2)
                
                # Compute cross-entropy loss (negative log likelihood)
                loss = -torch.sum(targets_batch * log_probs) / batch_samples
                
                # Add L1 regularization
                if self.l1_penalty > 0:
                    l1_term = self.l1_penalty * torch.sum(torch.abs(self.weights))
                    loss += l1_term
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                total_loss += loss.item() * batch_samples
            
            # Compute average loss for reporting
            avg_loss = total_loss / n_samples
            
            if verbose and iteration % 10 == 0:
                if isinstance(iterator, tqdm):
                    iterator.set_description(f"Loss: {avg_loss:.4f}")
                else:
                    print(f"Iteration {iteration}, Loss: {avg_loss:.4f}")
            
            # Check convergence with average loss
            if abs(prev_loss - avg_loss) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            prev_loss = avg_loss
        
        return self

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict probability distributions over ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_categories, n_ratings) containing
            probability distributions over possible ratings
        """
        # Convert to PyTorch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(device=self.device, dtype=torch.float32)
        
        # No gradient tracking needed for prediction
        with torch.no_grad():
            logits = F.linear(X, self.weights.t(), self.bias)
            logits_reshaped = logits.view(-1, self.n_categories, self.n_ratings)
            probs = F.softmax(logits_reshaped, dim=2)
            
        # Return as numpy array
        return probs.cpu().numpy()

    def predict_ratings(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Ratings array of shape (n_samples, n_categories) with values in [1, n_ratings]
        """
        # Convert to PyTorch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(device=self.device, dtype=torch.float32)
        
        # No gradient tracking needed for prediction
        with torch.no_grad():
            logits = F.linear(X, self.weights.t(), self.bias)
            logits_reshaped = logits.view(-1, self.n_categories, self.n_ratings)
            max_indices = torch.argmax(logits_reshaped, dim=2) + 1
            
        # Return as numpy array
        return max_indices.cpu().numpy()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Legacy method - returns the category with highest rating.
        
        Returns:
            Category indices of shape (n_samples,) with values in [0, n_categories-1]
        """
        ratings = self.predict_ratings(X)
        return np.argmax(ratings, axis=1)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the weights and bias as numpy arrays.
        
        Returns:
            Tuple of (weights, bias) as numpy arrays
        """
        with torch.no_grad():
            weights_np = self.weights.cpu().numpy()
            bias_np = self.bias.cpu().numpy()
        return weights_np, bias_np

    def to_device(self, device: str) -> 'RatingsClassifier':
        """Move the model to a different device.
        
        Args:
            device: Device to move to ('cuda', 'cpu')
            
        Returns:
            self: The model on the new device
        """
        self.device = torch.device(device)
        if self.weights is not None:
            self.weights = self.weights.to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)
        return self
