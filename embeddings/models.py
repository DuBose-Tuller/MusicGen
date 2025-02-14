import numpy as np
from typing import Tuple, Optional
from scipy.special import softmax
from tqdm import tqdm

class RatingsClassifier:
    """A linear classifier for multi-dimensional rating predictions with single-category ground truth.
    
    This classifier predicts ratings across multiple categories (e.g., how likely an input belongs
    to each category on a 4-point scale), but trains using single-category ground truth labels.
    """
    
    def __init__(self, n_categories: int = 4, n_ratings: int = 4, learning_rate: float = 0.01, 
                 max_iter: int = 1000, tol: float = 1e-4):
        """Initialize the classifier.
        
        Args:
            n_categories: Number of categories to rate
            n_ratings: Number of possible ratings per category
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of training iterations
            tol: Tolerance for stopping criterion
        """
        self.n_categories = n_categories
        self.n_ratings = n_ratings
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def _create_target_matrix(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        """Create target matrix for all samples in vectorized form.
        
        Args:
            y: Ground truth labels of shape (n_samples,)
            n_samples: Number of samples
            
        Returns:
            Target matrix of shape (n_samples, n_categories, n_ratings)
        """
        # Initialize target matrix
        targets = np.zeros((n_samples, self.n_categories, self.n_ratings))
        
        # Create index arrays for efficient assignment
        sample_idx = np.arange(n_samples)
        category_idx = np.arange(self.n_categories)
        
        # Set highest rating (n_ratings-1) for true categories
        targets[sample_idx, y, -1] = 1
        
        # Create a mask for false categories
        false_categories = category_idx[None, :] != y[:, None]
        
        # Set lowest rating (0) for false categories
        targets[false_categories] = np.array([1] + [0] * (self.n_ratings - 1))
        
        return targets

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss using vectorized operations.
        
        Args:
            logits: Model predictions of shape (n_samples, n_categories * n_ratings)
            targets: Target matrix of shape (n_samples, n_categories * n_ratings)
            
        Returns:
            Loss value
        """
        # Reshape logits for per-category softmax
        logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
        
        # Compute probabilities
        probs = softmax(logits_reshaped, axis=-1)
        
        # Compute cross-entropy loss
        log_probs = np.log(probs + 1e-10)
        loss = -np.sum(targets.reshape(-1, self.n_categories, self.n_ratings) * log_probs)
        
        return loss / logits.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'RatingsClassifier':
        """Fit the model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,) with values in [0, n_categories-1]
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
        targets = self._create_target_matrix(y, n_samples)
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
            grad_b = np.sum(grad, axis=0) / n_samples
            
            # Update weights and bias
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        
        return self

    def predict_ratings(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Ratings array of shape (n_samples, n_categories) with values in [0, n_ratings-1]
        """
        logits = X @ self.weights + self.bias
        logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
        return np.argmax(logits_reshaped, axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely category for each input.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) with values in [0, n_categories-1]
        """
        ratings = self.predict_ratings(X)
        return np.argmax(ratings, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distributions over ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_categories, n_ratings) containing
            probability distributions over possible ratings
        """
        logits = X @ self.weights + self.bias
        logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
        return softmax(logits_reshaped, axis=-1)


if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 4, size=n_samples)  # Single category labels
    
    # Create and train the classifier
    clf = RatingsClassifier(n_categories=4, n_ratings=4, max_iter=100)
    clf.fit(X, y, verbose=True)
    
    # Make predictions
    y_pred = clf.predict(X)  # Predicted categories
    print("Category predictions shape:", y_pred.shape)
    
    ratings = clf.predict_ratings(X)  # Predicted ratings for all categories
    print("Ratings predictions shape:", ratings.shape)
    
    # Get probability distributions
    probs = clf.predict_proba(X)
    print("Probability distributions shape:", probs.shape)