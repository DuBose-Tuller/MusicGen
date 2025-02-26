import numpy as np
from typing import Tuple, Optional
from scipy.special import softmax
from tqdm import tqdm

class RatingsClassifier:
    """A classifier for multi-dimensional rating predictions.
    
    This classifier predicts ratings across multiple categories (e.g., dance, heal, baby, love),
    where each category can have a discrete rating (e.g., 1-4).
    """
    
    def __init__(self, n_categories: int = 4, n_ratings: int = 4, learning_rate: float = 0.01, 
                 max_iter: int = 100, tol: float = 1e-4, l1_penalty: float = 0.0,
                 batch_size: int = 1024):
        """Initialize the classifier.
        
        Args:
            n_categories: Number of categories to rate
            n_ratings: Number of possible ratings per category
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of training iterations
            tol: Tolerance for stopping criterion
            l1_penalty: L1 regularization strength
            batch_size: Batch size for mini-batch gradient descent (0 for full batch)
        """
        self.n_categories = n_categories
        self.n_ratings = n_ratings
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l1_penalty = l1_penalty
        self.batch_size = batch_size
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def _create_target_matrix_vectorized(self, y: np.ndarray) -> np.ndarray:
        """Create target matrix from raw ratings using vectorized operations.
        
        Args:
            y: Rating values of shape (n_samples, n_categories) with values in [1, n_ratings]
            
        Returns:
            Target matrix of shape (n_samples, n_categories, n_ratings)
        """
        n_samples, n_cats = y.shape
        
        # Initialize target matrix
        targets = np.zeros((n_samples, self.n_categories, self.n_ratings))
        
        # Convert ratings to 0-based index and clamp to valid range
        y_idx = np.clip(y - 1, 0, self.n_ratings - 1).astype(np.int32)
        
        # Create index arrays for fancy indexing
        sample_idx = np.arange(n_samples)[:, np.newaxis]
        cat_idx = np.arange(n_cats)[np.newaxis, :]
        
        # Set 1s using fancy indexing
        targets[sample_idx, cat_idx, y_idx] = 1
                
        return targets

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss with L1 regularization"""
        # Reshape logits for per-category softmax
        batch_size = logits.shape[0]
        logits_reshaped = logits.reshape(batch_size, self.n_categories, self.n_ratings)
        
        # Compute probabilities
        probs = softmax(logits_reshaped, axis=-1)
        
        # Compute cross-entropy loss
        targets_reshaped = targets.reshape(batch_size, self.n_categories, self.n_ratings)
        log_probs = np.log(np.maximum(probs, 1e-10))  # Add small epsilon to avoid log(0)
        ce_loss = -np.sum(targets_reshaped * log_probs) / batch_size
        
        # Add L1 regularization term
        l1_term = 0
        if self.l1_penalty > 0:
            l1_term = self.l1_penalty * np.sum(np.abs(self.weights))
            
        return ce_loss + l1_term

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'RatingsClassifier':
        """Fit the model using mini-batch gradient descent with L1 regularization.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target ratings of shape (n_samples, n_categories) with values in [1, n_ratings]
            verbose: Whether to print progress
            
        Returns:
            self: The fitted classifier
        """
        n_samples, n_features = X.shape
        n_outputs = self.n_categories * self.n_ratings
        
        # Initialize weights and bias if not already initialized
        if self.weights is None:
            self.weights = np.random.randn(n_features, n_outputs) * 0.01
        if self.bias is None:
            self.bias = np.zeros(n_outputs)
        
        # Create target matrix once
        targets = self._create_target_matrix_vectorized(y)
        
        # Use mini-batches if batch_size > 0, otherwise use full batch
        batch_size = self.batch_size if self.batch_size > 0 else n_samples
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Setup for tracking progress
        prev_loss = float('inf')
        iterator = tqdm(range(self.max_iter)) if verbose else range(self.max_iter)
        
        for iteration in iterator:
            # Shuffle data for stochastic updates
            if batch_size < n_samples:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                targets_shuffled = targets[indices]
            else:
                X_shuffled = X
                targets_shuffled = targets
            
            # Track the total loss for this epoch
            total_loss = 0
            
            # Process mini-batches
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                # Get batch data
                X_batch = X_shuffled[start_idx:end_idx]
                targets_batch = targets_shuffled[start_idx:end_idx]
                batch_samples = end_idx - start_idx
                
                # Forward pass
                logits = X_batch @ self.weights + self.bias
                loss = self._compute_loss(logits, targets_batch)
                total_loss += loss * batch_samples
                
                # Backward pass (vectorized)
                logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
                probs = softmax(logits_reshaped, axis=-1)
                
                # Compute gradients
                grad = (probs - targets_batch).reshape(batch_samples, -1)
                grad_w = X_batch.T @ grad / batch_samples
                
                # Add L1 regularization gradient
                if self.l1_penalty > 0:
                    # Subgradient of L1 norm: sign(w)
                    l1_grad = self.l1_penalty * np.sign(self.weights)
                    grad_w += l1_grad
                    
                grad_b = np.mean(grad, axis=0)
                
                # Update weights and bias
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b
            
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distributions over ratings for each category."""
        logits = X @ self.weights + self.bias
        logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
        return softmax(logits_reshaped, axis=-1)

    def predict_ratings(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for each category.
        
        Returns:
            Ratings array of shape (n_samples, n_categories) with values in [1, n_ratings]
        """
        probs = self.predict_proba(X)
        # Get indices of max probability and convert to 1-based ratings
        raw_predictions = np.argmax(probs, axis=2) + 1
        return raw_predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Legacy method - returns the category with highest rating.
        
        Returns:
            Category indices of shape (n_samples,) with values in [0, n_categories-1]
        """
        ratings = self.predict_ratings(X)
        # Return the category with the highest predicted rating
        return np.argmax(ratings, axis=1)
