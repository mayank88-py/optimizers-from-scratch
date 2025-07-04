"""
Utility Functions for Gradient Descent Testing
==============================================

This module provides utility functions for common optimization problems
that can be used to test the gradient descent implementations.

Author: Mayank Kumar Kashyap
"""

import numpy as np
from typing import Tuple, Optional


def linear_regression_cost(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cost function for linear regression (Mean Squared Error).
    
    Args:
        weights: Model parameters [intercept, slope1, slope2, ...]
        X: Feature matrix with bias term (shape: n_samples x n_features)
        y: Target values (shape: n_samples,)
        
    Returns:
        Cost value (MSE)
    """
    n_samples = X.shape[0]
    predictions = X @ weights
    cost = (1 / (2 * n_samples)) * np.sum((predictions - y) ** 2)
    return cost


def linear_regression_gradient(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the gradient for linear regression.
    
    Args:
        weights: Model parameters [intercept, slope1, slope2, ...]
        X: Feature matrix with bias term (shape: n_samples x n_features)
        y: Target values (shape: n_samples,)
        
    Returns:
        Gradient vector (shape: n_features,)
    """
    n_samples = X.shape[0]
    predictions = X @ weights
    gradients = (1 / n_samples) * X.T @ (predictions - y)
    return gradients


def logistic_regression_cost(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cost function for logistic regression (Binary Cross-Entropy).
    
    Args:
        weights: Model parameters [intercept, slope1, slope2, ...]
        X: Feature matrix with bias term (shape: n_samples x n_features)
        y: Binary target values (shape: n_samples,)
        
    Returns:
        Cost value (Binary Cross-Entropy)
    """
    n_samples = X.shape[0]
    z = X @ weights
    
    # Sigmoid function with numerical stability
    sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Binary cross-entropy loss with numerical stability
    epsilon = 1e-15
    sigmoid = np.clip(sigmoid, epsilon, 1 - epsilon)
    cost = -(1 / n_samples) * np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
    
    return cost


def logistic_regression_gradient(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the gradient for logistic regression.
    
    Args:
        weights: Model parameters [intercept, slope1, slope2, ...]
        X: Feature matrix with bias term (shape: n_samples x n_features)
        y: Binary target values (shape: n_samples,)
        
    Returns:
        Gradient vector (shape: n_features,)
    """
    n_samples = X.shape[0]
    z = X @ weights
    
    # Sigmoid function with numerical stability
    sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    gradients = (1 / n_samples) * X.T @ (sigmoid - y)
    return gradients


def polynomial_regression_cost(weights: np.ndarray, X: np.ndarray, y: np.ndarray,
                              lambda_reg: float = 0.01) -> float:
    """
    Compute the cost function for polynomial regression with L2 regularization.
    
    Args:
        weights: Model parameters
        X: Feature matrix with polynomial features
        y: Target values
        lambda_reg: L2 regularization parameter
        
    Returns:
        Cost value with regularization
    """
    n_samples = X.shape[0]
    predictions = X @ weights
    mse = (1 / (2 * n_samples)) * np.sum((predictions - y) ** 2)
    l2_penalty = lambda_reg * np.sum(weights[1:] ** 2)  # Don't regularize bias term
    return mse + l2_penalty


def polynomial_regression_gradient(weights: np.ndarray, X: np.ndarray, y: np.ndarray,
                                  lambda_reg: float = 0.01) -> np.ndarray:
    """
    Compute the gradient for polynomial regression with L2 regularization.
    
    Args:
        weights: Model parameters
        X: Feature matrix with polynomial features
        y: Target values
        lambda_reg: L2 regularization parameter
        
    Returns:
        Gradient vector with regularization
    """
    n_samples = X.shape[0]
    predictions = X @ weights
    gradients = (1 / n_samples) * X.T @ (predictions - y)
    
    # Add L2 regularization (don't regularize bias term)
    reg_gradient = np.zeros_like(weights)
    reg_gradient[1:] = 2 * lambda_reg * weights[1:]
    
    return gradients + reg_gradient


def generate_linear_data(n_samples: int = 100, n_features: int = 2, noise: float = 0.1,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (excluding bias)
        noise: Noise level
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y, true_weights) where X includes bias term
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights
    true_weights = np.random.randn(n_features + 1)  # +1 for bias
    
    # Add bias term to X
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    # Generate targets
    y = X_with_bias @ true_weights + noise * np.random.randn(n_samples)
    
    return X_with_bias, y, true_weights


def generate_classification_data(n_samples: int = 100, n_features: int = 2,
                                random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic binary classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (excluding bias)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y, true_weights) where X includes bias term
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights
    true_weights = np.random.randn(n_features + 1)  # +1 for bias
    
    # Add bias term to X
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    # Generate targets using logistic function
    z = X_with_bias @ true_weights
    probabilities = 1 / (1 + np.exp(-z))
    y = (probabilities > 0.5).astype(int)
    
    return X_with_bias, y, true_weights


def generate_polynomial_data(n_samples: int = 100, degree: int = 3, noise: float = 0.1,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic polynomial regression data.
    
    Args:
        n_samples: Number of samples
        degree: Polynomial degree
        noise: Noise level
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y, true_weights) where X includes polynomial features
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate single feature
    x = np.random.uniform(-2, 2, n_samples)
    
    # Create polynomial features
    X = np.column_stack([x ** i for i in range(degree + 1)])
    
    # Generate true weights
    true_weights = np.random.randn(degree + 1)
    
    # Generate targets
    y = X @ true_weights + noise * np.random.randn(n_samples)
    
    return X, y, true_weights


def normalize_features(X: np.ndarray, exclude_bias: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        X: Feature matrix
        exclude_bias: Whether to exclude the first column (bias term) from normalization
        
    Returns:
        Tuple of (X_normalized, means, stds)
    """
    X_normalized = X.copy()
    
    if exclude_bias:
        # Don't normalize bias term (first column)
        features = X[:, 1:]
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        
        # Avoid division by zero
        stds[stds == 0] = 1
        
        X_normalized[:, 1:] = (features - means) / stds
        
        # Pad means and stds with zero for bias term
        means = np.concatenate([[0], means])
        stds = np.concatenate([[1], stds])
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        
        # Avoid division by zero
        stds[stds == 0] = 1
        
        X_normalized = (X - means) / stds
    
    return X_normalized, means, stds


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Accuracy value
    """
    return np.mean(y_true == y_pred)


def predict_logistic(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using logistic regression.
    
    Args:
        weights: Trained model parameters
        X: Feature matrix
        
    Returns:
        Binary predictions
    """
    z = X @ weights
    probabilities = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    return (probabilities > 0.5).astype(int) 