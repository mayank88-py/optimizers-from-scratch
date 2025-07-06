"""
Comprehensive Examples for Gradient Descent Implementations
==========================================================

This module provides practical examples demonstrating how to use all
gradient descent variants on different optimization problems.

Author: Mayank Kumar Kashyap
"""

import numpy as np
import matplotlib.pyplot as plt
from optimizers import *
from utils import *


def example_linear_regression():
    """Example: Linear Regression with different optimizers."""
    print("=" * 60)
    print("LINEAR REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic data
    X, y, true_weights = generate_linear_data(n_samples=1000, n_features=3, 
                                             noise=0.1, random_state=42)
    
    # Normalize features
    X_norm, means, stds = normalize_features(X, exclude_bias=True)
    
    # Initialize weights
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    # Set up optimizers
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.01, max_iterations=1000, verbose=False),
        'SGD': StochasticGradientDescent(learning_rate=0.01, max_iterations=100, verbose=False),
        'Mini-batch GD': MiniBatchGradientDescent(learning_rate=0.01, max_iterations=200, 
                                                 batch_size=64, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.01, max_iterations=200, 
                                         momentum=0.9, batch_size=64, verbose=False),
        'AdaGrad': AdaGrad(learning_rate=0.1, max_iterations=200, batch_size=64, verbose=False),
        'RMSprop': RMSprop(learning_rate=0.01, max_iterations=200, batch_size=64, verbose=False),
        'Adam': Adam(learning_rate=0.01, max_iterations=200, batch_size=64, verbose=False)
    }
    
    # Compare optimizers
    results = compare_optimizers(linear_regression_cost, linear_regression_gradient,
                                initial_weights, X_norm, y, optimizers)
    
    # Visualize convergence
    visualize_convergence(results, "Linear Regression: Optimizer Comparison")
    
    # Show final results
    print("\nFinal Results:")
    print(f"True weights: {true_weights}")
    for name, (weights, cost_history) in results.items():
        predictions = X_norm @ weights
        r2 = compute_r_squared(y, predictions)
        print(f"{name:15s}: Final Cost = {cost_history[-1]:.6f}, R² = {r2:.4f}")
    
    return results


def example_logistic_regression():
    """Example: Logistic Regression with different optimizers."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic data
    X, y, true_weights = generate_classification_data(n_samples=1000, n_features=2, 
                                                     random_state=42)
    
    # Normalize features
    X_norm, means, stds = normalize_features(X, exclude_bias=True)
    
    # Initialize weights
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    # Set up optimizers
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.1, max_iterations=1000, verbose=False),
        'SGD': StochasticGradientDescent(learning_rate=0.1, max_iterations=100, verbose=False),
        'Mini-batch GD': MiniBatchGradientDescent(learning_rate=0.1, max_iterations=200, 
                                                 batch_size=64, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.1, max_iterations=200, 
                                         momentum=0.9, batch_size=64, verbose=False),
        'AdaGrad': AdaGrad(learning_rate=0.3, max_iterations=200, batch_size=64, verbose=False),
        'RMSprop': RMSprop(learning_rate=0.1, max_iterations=200, batch_size=64, verbose=False),
        'Adam': Adam(learning_rate=0.1, max_iterations=200, batch_size=64, verbose=False)
    }
    
    # Compare optimizers
    results = compare_optimizers(logistic_regression_cost, logistic_regression_gradient,
                                initial_weights, X_norm, y, optimizers)
    
    # Visualize convergence
    visualize_convergence(results, "Logistic Regression: Optimizer Comparison")
    
    # Show final results
    print("\nFinal Results:")
    print(f"True weights: {true_weights}")
    for name, (weights, cost_history) in results.items():
        predictions = predict_logistic(weights, X_norm)
        accuracy = compute_accuracy(y, predictions)
        print(f"{name:15s}: Final Cost = {cost_history[-1]:.6f}, Accuracy = {accuracy:.4f}")
    
    return results


def example_polynomial_regression():
    """Example: Polynomial Regression with regularization."""
    print("\n" + "=" * 60)
    print("POLYNOMIAL REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic polynomial data
    X, y, true_weights = generate_polynomial_data(n_samples=200, degree=4, 
                                                 noise=0.2, random_state=42)
    
    # Normalize features (except bias)
    X_norm, means, stds = normalize_features(X, exclude_bias=True)
    
    # Initialize weights
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    # Set up optimizers with regularization
    lambda_reg = 0.01
    
    cost_func = lambda w, X, y: polynomial_regression_cost(w, X, y, lambda_reg)
    grad_func = lambda w, X, y: polynomial_regression_gradient(w, X, y, lambda_reg)
    
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.01, max_iterations=1000, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.01, max_iterations=300, 
                                         momentum=0.9, batch_size=32, verbose=False),
        'Adam': Adam(learning_rate=0.01, max_iterations=300, batch_size=32, verbose=False),
        'RMSprop': RMSprop(learning_rate=0.01, max_iterations=300, batch_size=32, verbose=False)
    }
    
    # Compare optimizers
    results = compare_optimizers(cost_func, grad_func, initial_weights, X_norm, y, optimizers)
    
    # Visualize convergence
    visualize_convergence(results, "Polynomial Regression: Optimizer Comparison")
    
    # Show final results
    print("\nFinal Results:")
    print(f"True weights: {true_weights}")
    for name, (weights, cost_history) in results.items():
        predictions = X_norm @ weights
        r2 = compute_r_squared(y, predictions)
        print(f"{name:15s}: Final Cost = {cost_history[-1]:.6f}, R² = {r2:.4f}")
    
    return results


def example_learning_rate_sensitivity():
    """Example: Demonstrate learning rate sensitivity."""
    print("\n" + "=" * 60)
    print("LEARNING RATE SENSITIVITY EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic data
    X, y, true_weights = generate_linear_data(n_samples=500, n_features=2, 
                                             noise=0.1, random_state=42)
    X_norm, means, stds = normalize_features(X, exclude_bias=True)
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.3, 0.5]
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 3, i + 1)
        
        # Test with different optimizers
        optimizers = {
            'Batch GD': BatchGradientDescent(learning_rate=lr, max_iterations=500, verbose=False),
            'SGD + Momentum': SGDWithMomentum(learning_rate=lr, max_iterations=100, 
                                             momentum=0.9, batch_size=32, verbose=False),
            'Adam': Adam(learning_rate=lr, max_iterations=100, batch_size=32, verbose=False)
        }
        
        for name, optimizer in optimizers.items():
            try:
                weights, cost_history = optimizer.optimize(linear_regression_cost, 
                                                          linear_regression_gradient,
                                                          initial_weights.copy(), X_norm, y)
                plt.plot(cost_history, label=name, linewidth=2)
            except:
                # Handle cases where optimization fails (e.g., learning rate too high)
                plt.plot([np.inf], label=f"{name} (failed)", linewidth=2)
        
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


def example_batch_size_comparison():
    """Example: Compare different batch sizes."""
    print("\n" + "=" * 60)
    print("BATCH SIZE COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic data
    X, y, true_weights = generate_linear_data(n_samples=1000, n_features=3, 
                                             noise=0.1, random_state=42)
    X_norm, means, stds = normalize_features(X, exclude_bias=True)
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    # Test different batch sizes
    batch_sizes = [1, 16, 64, 256, 1000]  # 1 = SGD, 1000 = Batch GD
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size == 1:
            optimizer = StochasticGradientDescent(learning_rate=0.01, max_iterations=50, verbose=False)
            name = 'SGD (batch=1)'
        elif batch_size == 1000:
            optimizer = BatchGradientDescent(learning_rate=0.01, max_iterations=200, verbose=False)
            name = 'Batch GD (batch=1000)'
        else:
            optimizer = MiniBatchGradientDescent(learning_rate=0.01, max_iterations=100, 
                                               batch_size=batch_size, verbose=False)
            name = f'Mini-batch GD (batch={batch_size})'
        
        weights, cost_history = optimizer.optimize(linear_regression_cost, 
                                                  linear_regression_gradient,
                                                  initial_weights.copy(), X_norm, y)
        results[name] = (weights, cost_history)
    
    # Visualize convergence
    visualize_convergence(results, "Batch Size Comparison")
    
    # Show final results
    print("\nFinal Results:")
    for name, (weights, cost_history) in results.items():
        predictions = X_norm @ weights
        r2 = compute_r_squared(y, predictions)
        print(f"{name:20s}: Final Cost = {cost_history[-1]:.6f}, R² = {r2:.4f}")


def example_convergence_visualization():
    """Example: Visualize convergence paths in 2D."""
    print("\n" + "=" * 60)
    print("CONVERGENCE VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Simple 2D quadratic function: f(x,y) = x² + y²
    def quadratic_cost(weights, X=None, y=None):
        return weights[0]**2 + weights[1]**2
    
    def quadratic_gradient(weights, X=None, y=None):
        return 2 * weights
    
    # Initial point
    initial_weights = np.array([3.0, 2.0])
    
    # Set up optimizers
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.1, max_iterations=50, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.1, max_iterations=50, 
                                         momentum=0.9, batch_size=1, verbose=False),
        'Adam': Adam(learning_rate=0.3, max_iterations=50, batch_size=1, verbose=False)
    }
    
    # Run optimizers
    results = {}
    for name, optimizer in optimizers.items():
        # Create dummy data (not used in this example)
        X_dummy = np.array([[1]])
        y_dummy = np.array([0])
        
        weights, cost_history = optimizer.optimize(quadratic_cost, quadratic_gradient,
                                                  initial_weights.copy(), X_dummy, y_dummy)
        results[name] = (weights, cost_history, optimizer.weights_history)
    
    # Create contour plot
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-3, 3, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = X_grid**2 + Y_grid**2
    
    plt.figure(figsize=(12, 8))
    
    # Plot contours
    contours = plt.contour(X_grid, Y_grid, Z, levels=20, alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Plot optimization paths
    colors = ['red', 'blue', 'green']
    for i, (name, (final_weights, cost_history, weights_history)) in enumerate(results.items()):
        weights_array = np.array(weights_history)
        plt.plot(weights_array[:, 0], weights_array[:, 1], 
                'o-', color=colors[i], label=name, markersize=4, linewidth=2)
        plt.plot(final_weights[0], final_weights[1], 's', color=colors[i], 
                markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    plt.plot(initial_weights[0], initial_weights[1], 'ko', markersize=8, label='Start')
    plt.plot(0, 0, 'k*', markersize=12, label='Optimum')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Paths on 2D Quadratic Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def run_all_examples():
    """Run all examples."""
    print("Running all gradient descent examples...")
    print("This may take a few minutes...")
    
    # Run all examples
    example_linear_regression()
    example_logistic_regression()
    example_polynomial_regression()
    example_learning_rate_sensitivity()
    example_batch_size_comparison()
    example_convergence_visualization()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples() 
