"""
Gradient Descent Implementation from Scratch
===========================================

This module implements various gradient descent algorithms from scratch using numpy.
Includes batch gradient descent, stochastic gradient descent, mini-batch gradient descent,
and advanced optimizers like SGD with momentum, AdaGrad, Adam, and RMSprop.

Author: Mayank Kumar Kashyap
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class GradientDescentBase(ABC):
    """Base class for all gradient descent implementations."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, verbose: bool = False):
        """
        Initialize base gradient descent parameters.
        
        Args:
            learning_rate: Step size for parameter updates
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to print progress
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.cost_history = []
        self.weights_history = []
        
    @abstractmethod
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Abstract method to be implemented by subclasses."""
        pass
    
    def _check_convergence(self, cost_history: List[float], iteration: int) -> bool:
        """Check if the algorithm has converged."""
        if len(cost_history) < 2:
            return False
        
        cost_diff = abs(cost_history[-1] - cost_history[-2])
        return cost_diff < self.tolerance or iteration >= self.max_iterations


class BatchGradientDescent(GradientDescentBase):
    """Batch Gradient Descent implementation."""
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform batch gradient descent optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        self.cost_history = []
        self.weights_history = []
        
        for iteration in range(self.max_iterations):
            # Compute cost and gradients using entire dataset
            cost = cost_function(weights, X, y)
            gradients = gradient_function(weights, X, y)
            
            # Update weights
            weights = weights - self.learning_rate * gradients
            
            # Store history
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


class StochasticGradientDescent(GradientDescentBase):
    """Stochastic Gradient Descent implementation."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, verbose: bool = False, shuffle: bool = True):
        """
        Initialize SGD with additional parameters.
        
        Args:
            shuffle: Whether to shuffle the data at each epoch
        """
        super().__init__(learning_rate, max_iterations, tolerance, verbose)
        self.shuffle = shuffle
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform stochastic gradient descent optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients for single sample
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        self.cost_history = []
        self.weights_history = []
        
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Shuffle data if specified
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process each sample
            for i in range(n_samples):
                # Single sample gradient
                Xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                gradients = gradient_function(weights, Xi, yi)
                weights = weights - self.learning_rate * gradients
            
            # Compute cost on entire dataset for tracking
            cost = cost_function(weights, X, y)
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


class MiniBatchGradientDescent(GradientDescentBase):
    """Mini-batch Gradient Descent implementation."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, verbose: bool = False, batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize mini-batch GD with additional parameters.
        
        Args:
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle the data at each epoch
        """
        super().__init__(learning_rate, max_iterations, tolerance, verbose)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform mini-batch gradient descent optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        self.cost_history = []
        self.weights_history = []
        
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Shuffle data if specified
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                gradients = gradient_function(weights, X_batch, y_batch)
                weights = weights - self.learning_rate * gradients
            
            # Compute cost on entire dataset for tracking
            cost = cost_function(weights, X, y)
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


class SGDWithMomentum(GradientDescentBase):
    """SGD with Momentum implementation."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, verbose: bool = False, momentum: float = 0.9,
                 batch_size: int = 32, shuffle: bool = True):
        """
        Initialize SGD with momentum.
        
        Args:
            momentum: Momentum factor (typically 0.9)
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle the data at each epoch
        """
        super().__init__(learning_rate, max_iterations, tolerance, verbose)
        self.momentum = momentum
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform SGD with momentum optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        velocity = np.zeros_like(weights)
        self.cost_history = []
        self.weights_history = []
        
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Shuffle data if specified
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                gradients = gradient_function(weights, X_batch, y_batch)
                
                # Update velocity and weights
                velocity = self.momentum * velocity - self.learning_rate * gradients
                weights = weights + velocity
            
            # Compute cost on entire dataset for tracking
            cost = cost_function(weights, X, y)
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


class AdaGrad(GradientDescentBase):
    """AdaGrad optimizer implementation."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, verbose: bool = False, epsilon: float = 1e-8,
                 batch_size: int = 32, shuffle: bool = True):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            epsilon: Small constant for numerical stability
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle the data at each epoch
        """
        super().__init__(learning_rate, max_iterations, tolerance, verbose)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform AdaGrad optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        sum_squared_gradients = np.zeros_like(weights)
        self.cost_history = []
        self.weights_history = []
        
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Shuffle data if specified
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                gradients = gradient_function(weights, X_batch, y_batch)
                
                # Accumulate squared gradients
                sum_squared_gradients += gradients ** 2
                
                # AdaGrad update
                adapted_lr = self.learning_rate / (np.sqrt(sum_squared_gradients) + self.epsilon)
                weights = weights - adapted_lr * gradients
            
            # Compute cost on entire dataset for tracking
            cost = cost_function(weights, X, y)
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


class RMSprop(GradientDescentBase):
    """RMSprop optimizer implementation."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, verbose: bool = False, beta: float = 0.9,
                 epsilon: float = 1e-8, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize RMSprop optimizer.
        
        Args:
            beta: Decay rate for moving average
            epsilon: Small constant for numerical stability
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle the data at each epoch
        """
        super().__init__(learning_rate, max_iterations, tolerance, verbose)
        self.beta = beta
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform RMSprop optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        squared_gradients = np.zeros_like(weights)
        self.cost_history = []
        self.weights_history = []
        
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Shuffle data if specified
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                gradients = gradient_function(weights, X_batch, y_batch)
                
                # Update moving average of squared gradients
                squared_gradients = self.beta * squared_gradients + (1 - self.beta) * gradients ** 2
                
                # RMSprop update
                weights = weights - self.learning_rate * gradients / (np.sqrt(squared_gradients) + self.epsilon)
            
            # Compute cost on entire dataset for tracking
            cost = cost_function(weights, X, y)
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


class Adam(GradientDescentBase):
    """Adam optimizer implementation."""
    
    def __init__(self, learning_rate: float = 0.001, max_iterations: int = 1000,
                 tolerance: float = 1e-6, verbose: bool = False, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize Adam optimizer.
        
        Args:
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle the data at each epoch
        """
        super().__init__(learning_rate, max_iterations, tolerance, verbose)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def optimize(self, cost_function: Callable, gradient_function: Callable,
                 initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform Adam optimization.
        
        Args:
            cost_function: Function to compute the cost
            gradient_function: Function to compute gradients
            initial_weights: Initial parameter values
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (optimized_weights, cost_history)
        """
        weights = initial_weights.copy()
        m = np.zeros_like(weights)  # First moment
        v = np.zeros_like(weights)  # Second moment
        self.cost_history = []
        self.weights_history = []
        
        n_samples = X.shape[0]
        t = 0  # Time step
        
        for iteration in range(self.max_iterations):
            # Shuffle data if specified
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                gradients = gradient_function(weights, X_batch, y_batch)
                
                t += 1
                
                # Update biased first moment estimate
                m = self.beta1 * m + (1 - self.beta1) * gradients
                
                # Update biased second moment estimate
                v = self.beta2 * v + (1 - self.beta2) * gradients ** 2
                
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - self.beta1 ** t)
                
                # Compute bias-corrected second moment estimate
                v_hat = v / (1 - self.beta2 ** t)
                
                # Update weights
                weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Compute cost on entire dataset for tracking
            cost = cost_function(weights, X, y)
            self.cost_history.append(cost)
            self.weights_history.append(weights.copy())
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
            
            # Check convergence
            if self._check_convergence(self.cost_history, iteration):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return weights, self.cost_history


def visualize_convergence(optimizers_results: Dict[str, Tuple[np.ndarray, List[float]]],
                         title: str = "Convergence Comparison"):
    """
    Visualize the convergence of different optimizers.
    
    Args:
        optimizers_results: Dictionary with optimizer names as keys and 
                           (weights, cost_history) as values
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    for name, (weights, cost_history) in optimizers_results.items():
        plt.plot(cost_history, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()


def compare_optimizers(cost_function: Callable, gradient_function: Callable,
                      initial_weights: np.ndarray, X: np.ndarray, y: np.ndarray,
                      optimizers: Dict[str, GradientDescentBase]) -> Dict[str, Tuple[np.ndarray, List[float]]]:
    """
    Compare different optimizers on the same problem.
    
    Args:
        cost_function: Function to compute the cost
        gradient_function: Function to compute gradients
        initial_weights: Initial parameter values
        X: Feature matrix
        y: Target values
        optimizers: Dictionary of optimizer instances
        
    Returns:
        Dictionary with optimizer names as keys and (weights, cost_history) as values
    """
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nRunning {name}...")
        weights, cost_history = optimizer.optimize(cost_function, gradient_function,
                                                  initial_weights.copy(), X, y)
        results[name] = (weights, cost_history)
        print(f"{name} final cost: {cost_history[-1]:.6f}")
    
    return results 
