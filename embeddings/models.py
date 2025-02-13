import numpy as np
from typing import Tuple, Optional
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
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function to the input.
        
        Args:
            x: Input array of shape (n_samples, n_classes)
            
        Returns:
            Softmax probabilities of shape (n_samples, n_classes)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss using ground truth categories.
        
        Args:
            logits: Model predictions of shape (n_samples, n_categories * n_ratings)
            y: Ground truth labels of shape (n_samples,) with values in [0, n_categories-1]
            
        Returns:
            Loss value
        """
        n_samples = y.shape[0]
        total_loss = 0
        
        for i in range(n_samples):
            true_category = y[i]
            sample_loss = 0
            
            # Compute loss for each category
            for category in range(self.n_categories):
                start_idx = category * self.n_ratings
                end_idx = (category + 1) * self.n_ratings
                
                # Target: highest rating (3) for true category, lowest (0) for others
                target = np.zeros(self.n_ratings)
                if category == true_category:
                    target[-1] = 1  # Want rating of 3 for true category
                else:
                    target[0] = 1   # Want rating of 0 for other categories
                
                category_logits = logits[i, start_idx:end_idx]
                probs = self._softmax(category_logits.reshape(1, -1))[0]
                sample_loss -= np.sum(target * np.log(probs + 1e-10))
            
            total_loss += sample_loss
            
        return total_loss / n_samples

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=False) -> 'RatingsClassifier':
        """Fit the model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,) with values in [0, n_categories-1]
                indicating the true category for each sample
            
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
        
        prev_loss = float('inf')
        
        # Gradient descent
        for iteration in tqdm(range(self.max_iter)) if verbose else range(self.max_iter):
            # Forward pass
            logits = X @ self.weights + self.bias
            loss = self._compute_loss(logits, y)
            
            if verbose and iteration % 10 == 0:
                print(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            
            # Backward pass
            grad_w = np.zeros_like(self.weights)
            grad_b = np.zeros_like(self.bias)
            
            for i in range(n_samples):
                true_category = y[i]
                
                # Compute gradients for all categories
                for category in range(self.n_categories):
                    start_idx = category * self.n_ratings
                    end_idx = (category + 1) * self.n_ratings
                    
                    # Target: highest rating for true category, lowest for others
                    target = np.zeros(self.n_ratings)
                    if category == true_category:
                        target[-1] = 1  # Want rating of 3 for true category
                    else:
                        target[0] = 1   # Want rating of 0 for other categories
                    
                    category_logits = logits[i, start_idx:end_idx]
                    probs = self._softmax(category_logits.reshape(1, -1))[0]
                    
                    # Compute gradients
                    grad = probs - target
                    grad_w[:, start_idx:end_idx] += X[i:i+1].T @ grad.reshape(1, -1)
                    grad_b[start_idx:end_idx] += grad
            
            # Update weights and bias
            grad_w /= n_samples
            grad_b /= n_samples
            
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        
        return self
    
    def predict_ratings(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for each category.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Ratings array of shape (n_samples, n_categories) with values in [0, n_ratings-1]
            indicating the predicted rating for each category
        """
        logits = X @ self.weights + self.bias
        predictions = np.zeros((X.shape[0], self.n_categories), dtype=int)
        
        for i in range(self.n_categories):
            start_idx = i * self.n_ratings
            end_idx = (i + 1) * self.n_ratings
            category_logits = logits[:, start_idx:end_idx]
            predictions[:, i] = np.argmax(category_logits, axis=1)
        
        return predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely category for each input.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) with values in [0, n_categories-1]
            indicating the predicted category
        """
        ratings = self.predict_ratings(X)
        # Predict the category with the highest rating
        return np.argmax(ratings, axis=1)
    
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
    print("Number of probability distributions:", len(probs))
    print("Shape of each distribution:", probs[0].shape)