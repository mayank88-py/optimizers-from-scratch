#!/usr/bin/env python3
"""
Comprehensive Demo of Gradient Descent Implementations
=====================================================

This script demonstrates the key features of all gradient descent
implementations with practical examples.

Author: Mayank Kumar Kashyap
"""

import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import *
from utils import *

def demo_basic_usage():
    """Demonstrate basic usage of gradient descent."""
    print("=" * 60)
    print("BASIC USAGE DEMO")
    print("=" * 60)
    
    # Generate data
    X, y, true_weights = generate_linear_data(n_samples=500, n_features=2, 
                                             noise=0.1, random_state=42)
    X_norm, _, _ = normalize_features(X, exclude_bias=True)
    
    # Initialize weights
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True weights: {true_weights}")
    print()
    
    # Demonstrate different optimizers
    print("1. Batch Gradient Descent:")
    bgd = BatchGradientDescent(learning_rate=0.01, max_iterations=200, verbose=True)
    weights_bgd, cost_bgd = bgd.optimize(linear_regression_cost, linear_regression_gradient,
                                        initial_weights.copy(), X_norm, y)
    print(f"   Final weights: {weights_bgd}")
    print(f"   Final cost: {cost_bgd[-1]:.6f}")
    print()
    
    print("2. Adam Optimizer:")
    adam = Adam(learning_rate=0.01, max_iterations=100, batch_size=32, verbose=True)
    weights_adam, cost_adam = adam.optimize(linear_regression_cost, linear_regression_gradient,
                                           initial_weights.copy(), X_norm, y)
    print(f"   Final weights: {weights_adam}")
    print(f"   Final cost: {cost_adam[-1]:.6f}")
    print()


def demo_comparison():
    """Demonstrate optimizer comparison."""
    print("=" * 60)
    print("OPTIMIZER COMPARISON DEMO")
    print("=" * 60)
    
    # Generate data
    X, y, true_weights = generate_linear_data(n_samples=1000, n_features=3, 
                                             noise=0.1, random_state=42)
    X_norm, _, _ = normalize_features(X, exclude_bias=True)
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    # Set up optimizers
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.01, max_iterations=300, verbose=False),
        'SGD': StochasticGradientDescent(learning_rate=0.01, max_iterations=100, verbose=False),
        'Mini-batch GD': MiniBatchGradientDescent(learning_rate=0.01, max_iterations=100, 
                                                 batch_size=64, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.01, max_iterations=100, 
                                         momentum=0.9, batch_size=64, verbose=False),
        'Adam': Adam(learning_rate=0.01, max_iterations=100, batch_size=64, verbose=False)
    }
    
    # Compare optimizers
    results = compare_optimizers(linear_regression_cost, linear_regression_gradient,
                                initial_weights, X_norm, y, optimizers)
    
    print("\nDetailed Results:")
    print(f"{'Optimizer':<15} {'Final Cost':<12} {'R²':<8} {'Iterations':<10}")
    print("-" * 50)
    
    for name, (weights, cost_history) in results.items():
        predictions = X_norm @ weights
        r2 = compute_r_squared(y, predictions)
        iterations = len(cost_history)
        print(f"{name:<15} {cost_history[-1]:<12.6f} {r2:<8.4f} {iterations:<10}")


def demo_logistic_regression():
    """Demonstrate logistic regression example."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION DEMO")
    print("=" * 60)
    
    # Generate classification data
    X, y, true_weights = generate_classification_data(n_samples=800, n_features=2, 
                                                     random_state=42)
    X_norm, _, _ = normalize_features(X, exclude_bias=True)
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    print()
    
    # Test different optimizers
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.1, max_iterations=300, verbose=False),
        'Adam': Adam(learning_rate=0.1, max_iterations=100, batch_size=64, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.1, max_iterations=100, 
                                         momentum=0.9, batch_size=64, verbose=False)
    }
    
    results = compare_optimizers(logistic_regression_cost, logistic_regression_gradient,
                                initial_weights, X_norm, y, optimizers)
    
    print("\nClassification Results:")
    print(f"{'Optimizer':<15} {'Final Cost':<12} {'Accuracy':<8}")
    print("-" * 40)
    
    for name, (weights, cost_history) in results.items():
        predictions = predict_logistic(weights, X_norm)
        accuracy = compute_accuracy(y, predictions)
        print(f"{name:<15} {cost_history[-1]:<12.6f} {accuracy:<8.4f}")


def demo_learning_rate_effect():
    """Demonstrate the effect of learning rate."""
    print("\n" + "=" * 60)
    print("LEARNING RATE EFFECT DEMO")
    print("=" * 60)
    
    # Generate data
    X, y, true_weights = generate_linear_data(n_samples=300, n_features=2, 
                                             noise=0.1, random_state=42)
    X_norm, _, _ = normalize_features(X, exclude_bias=True)
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    print(f"{'Learning Rate':<15} {'Final Cost':<12} {'R²':<8} {'Converged':<10}")
    print("-" * 50)
    
    for lr in learning_rates:
        try:
            optimizer = BatchGradientDescent(learning_rate=lr, max_iterations=200, verbose=False)
            weights, cost_history = optimizer.optimize(linear_regression_cost, 
                                                      linear_regression_gradient,
                                                      initial_weights.copy(), X_norm, y)
            
            predictions = X_norm @ weights
            r2 = compute_r_squared(y, predictions)
            converged = len(cost_history) < optimizer.max_iterations
            
            print(f"{lr:<15} {cost_history[-1]:<12.6f} {r2:<8.4f} {'Yes' if converged else 'No':<10}")
            
        except Exception as e:
            print(f"{lr:<15} {'Failed':<12} {'N/A':<8} {'No':<10}")


def demo_polynomial_fitting():
    """Demonstrate polynomial regression with regularization."""
    print("\n" + "=" * 60)
    print("POLYNOMIAL REGRESSION DEMO")
    print("=" * 60)
    
    # Generate polynomial data
    X, y, true_weights = generate_polynomial_data(n_samples=150, degree=3, 
                                                 noise=0.2, random_state=42)
    X_norm, _, _ = normalize_features(X, exclude_bias=True)
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    print(f"Polynomial degree: {X.shape[1] - 1}")
    print(f"Dataset: {X.shape[0]} samples")
    print(f"True weights: {true_weights}")
    print()
    
    # Test with different regularization
    lambda_values = [0.0, 0.01, 0.1]
    
    print(f"{'Lambda':<10} {'Final Cost':<12} {'R²':<8} {'Weight L2 Norm':<15}")
    print("-" * 50)
    
    for lambda_reg in lambda_values:
        cost_func = lambda w, X, y: polynomial_regression_cost(w, X, y, lambda_reg)
        grad_func = lambda w, X, y: polynomial_regression_gradient(w, X, y, lambda_reg)
        
        optimizer = Adam(learning_rate=0.01, max_iterations=200, batch_size=32, verbose=False)
        weights, cost_history = optimizer.optimize(cost_func, grad_func,
                                                  initial_weights.copy(), X_norm, y)
        
        predictions = X_norm @ weights
        r2 = compute_r_squared(y, predictions)
        weight_norm = np.linalg.norm(weights)
        
        print(f"{lambda_reg:<10} {cost_history[-1]:<12.6f} {r2:<8.4f} {weight_norm:<15.4f}")


def main():
    """Run all demos."""
    print("GRADIENT DESCENT FROM SCRATCH - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all demos
    demo_basic_usage()
    demo_comparison()
    demo_logistic_regression()
    demo_learning_rate_effect()
    demo_polynomial_fitting()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Different optimizers have different convergence behaviors")
    print("2. Learning rate significantly affects convergence")
    print("3. Adam and SGD with momentum often perform well")
    print("4. Feature normalization is crucial for performance")
    print("5. Regularization helps prevent overfitting")
    print("\nFor more examples, run: python examples.py")


if __name__ == "__main__":
    main() 