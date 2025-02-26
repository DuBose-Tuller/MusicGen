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
                 max_iter: int = 100, tol: float = 1e-4, l1_penalty: float = 0.0):
        """Initialize the classifier.
        
        Args:
            n_categories: Number of categories to rate
            n_ratings: Number of possible ratings per category
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of training iterations
            tol: Tolerance for stopping criterion
            l1_penalty: L1 regularization strength
        """
        self.n_categories = n_categories
        self.n_ratings = n_ratings
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l1_penalty = l1_penalty
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def _create_target_matrix(self, y: np.ndarray) -> np.ndarray:
        """Create target matrix from raw ratings.
        
        Args:
            y: Rating values of shape (n_samples, n_categories) with values in [1, n_ratings]
            
        Returns:
            Target matrix of shape (n_samples, n_categories, n_ratings)
        """
        n_samples = y.shape[0]
        
        # Initialize target matrix
        targets = np.zeros((n_samples, self.n_categories, self.n_ratings))
        
        # For each sample and each category, set a 1 at the index corresponding to the rating
        # Note: Ratings typically start at 1, but indices start at 0
        for i in range(n_samples):
            for j in range(self.n_categories):
                # Convert rating value to 0-based index
                rating_idx = int(y[i, j]) - 1
                # Clamp to valid range in case of invalid ratings
                rating_idx = max(0, min(rating_idx, self.n_ratings - 1))
                targets[i, j, rating_idx] = 1
                
        return targets

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss with L1 regularization"""
        # Reshape logits for per-category softmax
        logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
        
        # Compute probabilities
        probs = softmax(logits_reshaped, axis=-1)
        
        # Compute cross-entropy loss
        log_probs = np.log(probs + 1e-10)
        ce_loss = -np.sum(targets.reshape(-1, self.n_categories, self.n_ratings) * log_probs)
        
        # Add L1 regularization term (only for weights, not bias)
        l1_term = 0
        if self.l1_penalty > 0:
            l1_term = self.l1_penalty * np.sum(np.abs(self.weights))
            
        return ce_loss / logits.shape[0] + l1_term

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'RatingsClassifier':
        """Fit the model using gradient descent with L1 regularization.
        
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
        targets = self._create_target_matrix(y)
        targets_flat = targets.reshape(n_samples, -1)
        
        prev_loss = float('inf')
        iterator = tqdm(range(self.max_iter)) if verbose else range(self.max_iter)
        
        for iteration in iterator:
            # Forward pass
            logits = X @ self.weights + self.bias
            loss = self._compute_loss(logits, targets_flat)
            
            if verbose and iteration % 10 == 0:
                if isinstance(iterator, tqdm):
                    iterator.set_description(f"Loss: {loss:.4f}")
                else:
                    print(f"Iteration {iteration}, Loss: {loss:.4f}")
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            prev_loss = loss
            
            # Backward pass (vectorized)
            logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
            probs = softmax(logits_reshaped, axis=-1)
            
            # Compute gradients
            grad = (probs - targets).reshape(n_samples, -1)
            grad_w = X.T @ grad / n_samples
            
            # Add L1 regularization gradient
            if self.l1_penalty > 0:
                # Subgradient of L1 norm: sign(w)
                l1_grad = self.l1_penalty * np.sign(self.weights)
                grad_w += l1_grad
                
            grad_b = np.sum(grad, axis=0) / n_samples
            
            # Update weights and bias
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        
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
