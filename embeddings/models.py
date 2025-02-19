import numpy as np
from typing import Tuple, Optional, Callable
from scipy.special import softmax
from scipy.optimize import minimize
import time
from tqdm import tqdm

class RatingsClassifier:
    """A linear classifier for multi-dimensional rating predictions with single-category ground truth.
    
    This classifier predicts ratings across multiple categories using L-BFGS optimization,
    similar to scikit-learn's LogisticRegression but with extended output capabilities.
    """
    
    def __init__(self, n_categories: int = 4, n_ratings: int = 4, 
                 max_iter: int = 100, tol: float = 1e-4, l1_penalty: float = 0.0):
        """Initialize the classifier.
        
        Args:
            n_categories: Number of categories to rate
            n_ratings: Number of possible ratings per category
            max_iter: Maximum number of iterations for optimization
            tol: Tolerance for stopping criterion
            l1_penalty: L1 regularization strength (higher values = more regularization)
        """
        self.n_categories = n_categories
        self.n_ratings = n_ratings
        self.max_iter = max_iter
        self.tol = tol
        self.l1_penalty = l1_penalty
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
        
        # Set highest rating (n_ratings-1) for true categories
        targets[sample_idx, y, -1] = 1
        
        # Set lowest rating (0) for false categories
        for i in range(n_samples):
            for j in range(self.n_categories):
                if j != y[i]:
                    targets[i, j, 0] = 1
        
        return targets.reshape(n_samples, -1)  # Flatten for optimization

    def _params_to_weights_bias(self, params: np.ndarray, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flat parameters array to weights and bias.
        
        Args:
            params: Flat array of parameters
            n_features: Number of input features
            
        Returns:
            Tuple of (weights, bias)
        """
        n_outputs = self.n_categories * self.n_ratings
        weights = params[:n_features * n_outputs].reshape(n_features, n_outputs)
        bias = params[n_features * n_outputs:]
        return weights, bias

    def _weights_bias_to_params(self, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Convert weights and bias to flat parameters array.
        
        Args:
            weights: Weights matrix
            bias: Bias vector
            
        Returns:
            Flat array of parameters
        """
        return np.concatenate([weights.flatten(), bias])

    def _objective_function(self, params: np.ndarray, X: np.ndarray, 
                           targets: np.ndarray, n_features: int) -> float:
        """Compute objective function (negative log-likelihood + regularization).
        
        Args:
            params: Flat array of parameters
            X: Input data
            targets: Target values
            n_features: Number of input features
            
        Returns:
            Objective function value
        """
        weights, bias = self._params_to_weights_bias(params, n_features)
        n_samples = X.shape[0]
        
        # Forward pass
        logits = X @ weights + bias
        logits_reshaped = logits.reshape(n_samples, self.n_categories, self.n_ratings)
        probs = softmax(logits_reshaped, axis=-1)
        
        # Compute cross-entropy loss
        log_probs = np.log(probs + 1e-10)
        targets_reshaped = targets.reshape(n_samples, self.n_categories, self.n_ratings)
        ce_loss = -np.sum(targets_reshaped * log_probs) / n_samples
        
        # Add L1 regularization if needed
        l1_term = 0
        if self.l1_penalty > 0:
            # Use smooth approximation of L1 for better optimization
            epsilon = 1e-8
            l1_term = self.l1_penalty * np.sum(np.sqrt(weights**2 + epsilon))
            
        return ce_loss + l1_term

    def _objective_gradient(self, params: np.ndarray, X: np.ndarray, 
                           targets: np.ndarray, n_features: int) -> np.ndarray:
        """Compute gradient of the objective function.
        
        Args:
            params: Flat array of parameters
            X: Input data
            targets: Target values
            n_features: Number of input features
            
        Returns:
            Gradient of objective function
        """
        weights, bias = self._params_to_weights_bias(params, n_features)
        n_samples = X.shape[0]
        
        # Forward pass
        logits = X @ weights + bias
        logits_reshaped = logits.reshape(n_samples, self.n_categories, self.n_ratings)
        probs = softmax(logits_reshaped, axis=-1)
        
        # Compute gradient
        targets_reshaped = targets.reshape(n_samples, self.n_categories, self.n_ratings)
        grad = (probs - targets_reshaped).reshape(n_samples, -1)
        grad_w = X.T @ grad / n_samples
        grad_b = np.sum(grad, axis=0) / n_samples
        
        # Add L1 regularization gradient if needed
        if self.l1_penalty > 0:
            # Gradient of smooth L1 approximation
            epsilon = 1e-8
            l1_grad = self.l1_penalty * weights / np.sqrt(weights**2 + epsilon)
            grad_w += l1_grad
        
        # Combine gradients into flat array
        return np.concatenate([grad_w.flatten(), grad_b])

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'RatingsClassifier':
        """Fit the model using L-BFGS optimization.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,) with values in [0, n_categories-1]
            verbose: Whether to print progress
            
        Returns:
            self: The fitted classifier
        """
        n_samples, n_features = X.shape
        
        # Auto-detect number of categories if needed
        if np.max(y) >= self.n_categories:
            self.n_categories = np.max(y) + 1
            if verbose:
                print(f"Auto-adjusting to {self.n_categories} categories based on input labels")
        
        n_outputs = self.n_categories * self.n_ratings
        
        # Create target matrix
        targets = self._create_target_matrix(y, n_samples)
        
        # Initialize parameters if not already initialized
        if self.weights is None or self.bias is None:
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (n_features + n_outputs))
            self.weights = np.random.randn(n_features, n_outputs) * scale
            self.bias = np.zeros(n_outputs)
        
        # Convert to flat parameter array
        initial_params = self._weights_bias_to_params(self.weights, self.bias)
        
        # Setup callback for progress reporting
        iteration = [0]
        start_time = time.time()
        
        def callback(params):
            iteration[0] += 1
            if verbose and iteration[0] % 5 == 0:
                elapsed = time.time() - start_time
                obj_value = self._objective_function(params, X, targets, n_features)
                print(f"Iteration {iteration[0]}, Objective: {obj_value:.6f}, Time: {elapsed:.2f}s")
        
        # Run optimization
        if verbose:
            print(f"Starting L-BFGS optimization with max_iter={self.max_iter}, l1_penalty={self.l1_penalty}")
        
        result = minimize(
            fun=self._objective_function,
            x0=initial_params,
            args=(X, targets, n_features),
            method='L-BFGS-B',
            jac=self._objective_gradient,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'disp': verbose
            },
            callback=callback if verbose else None
        )
        
        # Extract optimized parameters
        self.weights, self.bias = self._params_to_weights_bias(result.x, n_features)
        
        # Set attributes for sklearn-like interface
        self.coef_ = self.weights
        self.intercept_ = self.bias
        self.n_iter_ = result.nit
        
        if verbose:
            print(f"Optimization completed in {result.nit} iterations")
            print(f"Final loss: {result.fun:.6f}")
            if not result.success:
                print(f"Warning: Optimization did not converge. Message: {result.message}")
        
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
        """Predict probability distributions over categories.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_categories) containing category probabilities
        """
        logits = X @ self.weights + self.bias
        logits_reshaped = logits.reshape(-1, self.n_categories, self.n_ratings)
        probs = softmax(logits_reshaped, axis=-1)
        
        # Take highest rating probability as the category probability
        # This matches the training objective (correct category should have highest rating)
        category_probs = probs[:, :, -1]
        
        # Normalize to ensure probabilities sum to 1
        category_probs = category_probs / np.sum(category_probs, axis=1, keepdims=True)
        
        return category_probs


if __name__ == "__main__":
    # Test against sklearn's LogisticRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import time
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, 
                             n_informative=10, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train sklearn LogisticRegression
    start_time = time.time()
    sklearn_clf = LogisticRegression(max_iter=10000, tol=1e-4, 
                                    penalty='l1', solver='liblinear', C=1.0)
    sklearn_clf.fit(X_scaled, y)
    sklearn_time = time.time() - start_time
    sklearn_pred = sklearn_clf.predict(X_scaled)
    sklearn_acc = accuracy_score(y, sklearn_pred)
    
    # Train our RatingsClassifier
    start_time = time.time()
    our_clf = RatingsClassifier(n_categories=4, n_ratings=4, max_iter=10000, 
                              tol=1e-4, l1_penalty=0.01)
    our_clf.fit(X_scaled, y)
    our_time = time.time() - start_time
    our_pred = our_clf.predict(X_scaled)
    our_acc = accuracy_score(y, our_pred)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"{'Metric':<15} {'scikit-learn':<15} {'RatingsClassifier':<15}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15} {sklearn_acc:.4f}{'':<8} {our_acc:.4f}")
    print(f"{'Time (s)':<15} {sklearn_time:.4f}{'':<8} {our_time:.4f}")
    print(f"{'Iterations':<15} {sklearn_clf.n_iter_}{'':<12} {our_clf.n_iter_}")