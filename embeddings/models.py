import numpy as np
from typing import Tuple, Optional

class RatingsClassifier:
    """A linear classifier for multi-dimensional rating scales.
    
    This classifier handles multiple rating categories simultaneously, where each category
    has the same number of possible ratings (e.g., 4-point scale across 4 categories).
    It uses separate weights for each category but trains them jointly.
    
    Attributes:
        n_categories (int): Number of rating categories (e.g., 4 for four different aspects being rated)
        n_ratings (int): Number of possible ratings per category (e.g., 4 for a 4-point scale)
        weights (np.ndarray): Weight matrix of shape (n_features, n_categories * n_ratings)
        bias (np.ndarray): Bias vector of shape (n_categories * n_ratings,)
    """
    
    def __init__(self, n_categories: int = 4, n_ratings: int = 4, learning_rate: float = 0.01, 
                 max_iter: int = 1000, tol: float = 1e-4):
        """Initialize the classifier.
        
        Args:
            n_categories: Number of rating categories
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
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function to the input.
        
        Args:
            x: Input array of shape (n_samples, n_classes)
            
        Returns:
            Softmax probabilities of shape (n_samples, n_classes)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _to_one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot encoded format.
        
        Args:
            y: Labels array of shape (n_samples, n_categories)
            
        Returns:
            One-hot encoded labels of shape (n_samples, n_categories * n_ratings)
        """
        n_samples = y.shape[0]
        y_one_hot = np.zeros((n_samples, self.n_categories * self.n_ratings))
        
        for i in range(self.n_categories):
            start_idx = i * self.n_ratings
            end_idx = (i + 1) * self.n_ratings
            y_one_hot[:, start_idx:end_idx] = np.eye(self.n_ratings)[y[:, i]]
        
        return y_one_hot

    def _compute_loss(self, logits: np.ndarray, y_one_hot: np.ndarray) -> float:
        """Compute the total cross-entropy loss across all categories.
        
        Args:
            logits: Model predictions of shape (n_samples, n_categories * n_ratings)
            y_one_hot: One-hot encoded labels of shape (n_samples, n_categories * n_ratings)
            
        Returns:
            Total loss value
        """
        total_loss = 0
        for i in range(self.n_categories):
            start_idx = i * self.n_ratings
            end_idx = (i + 1) * self.n_ratings
            
            category_logits = logits[:, start_idx:end_idx]
            category_labels = y_one_hot[:, start_idx:end_idx]
            
            probs = self._softmax(category_logits)
            category_loss = -np.mean(
                np.sum(category_labels * np.log(probs + 1e-10), axis=1)
            )
            total_loss += category_loss
            
        return total_loss / self.n_categories

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RatingsClassifier':
        """Fit the model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples, n_categories) with values in [0, n_ratings-1]
            
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
        
        y_one_hot = self._to_one_hot(y)
        prev_loss = float('inf')
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Forward pass
            logits = X @ self.weights + self.bias
            loss = self._compute_loss(logits, y_one_hot)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            
            # Backward pass for each category
            total_grad_w = np.zeros_like(self.weights)
            total_grad_b = np.zeros_like(self.bias)
            
            for i in range(self.n_categories):
                start_idx = i * self.n_ratings
                end_idx = (i + 1) * self.n_ratings
                
                category_logits = logits[:, start_idx:end_idx]
                category_labels = y_one_hot[:, start_idx:end_idx]
                
                probs = self._softmax(category_logits)
                grad = probs - category_labels
                
                # Accumulate gradients
                total_grad_w[:, start_idx:end_idx] = X.T @ grad / n_samples
                total_grad_b[start_idx:end_idx] = np.mean(grad, axis=0)
            
            # Update weights and bias
            self.weights -= self.learning_rate * total_grad_w
            self.bias -= self.learning_rate * total_grad_b
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_categories) with values in [0, n_ratings-1]
        """
        logits = X @ self.weights + self.bias
        predictions = np.zeros((X.shape[0], self.n_categories), dtype=int)
        
        for i in range(self.n_categories):
            start_idx = i * self.n_ratings
            end_idx = (i + 1) * self.n_ratings
            category_logits = logits[:, start_idx:end_idx]
            predictions[:, i] = np.argmax(category_logits, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predict probability distributions over ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple of n_categories arrays, each of shape (n_samples, n_ratings)
            containing probability distributions over possible ratings
        """
        logits = X @ self.weights + self.bias
        probabilities = []
        
        for i in range(self.n_categories):
            start_idx = i * self.n_ratings
            end_idx = (i + 1) * self.n_ratings
            category_logits = logits[:, start_idx:end_idx]
            category_probs = self._softmax(category_logits)
            probabilities.append(category_probs)
        
        return tuple(probabilities)

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 4, size=(n_samples, 4))  # 4 categories, 4 ratings each
    
    # Create and train the classifier
    clf = RatingsClassifier(n_categories=4, n_ratings=4)
    clf.fit(X, y)
    
    # Make predictions
    y_pred = clf.predict(X)
    print("Predictions shape:", y_pred.shape)
    
    # Get probability distributions
    probs = clf.predict_proba(X)
    print("Number of probability distributions:", len(probs))
    print("Shape of each distribution:", probs[0].shape)