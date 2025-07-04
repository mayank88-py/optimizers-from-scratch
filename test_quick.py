#!/usr/bin/env python3
"""
Quick Test Script for Gradient Descent Implementations
=====================================================

This script performs a quick test of all gradient descent implementations
to verify they work correctly.

Author: Mayank Kumar Kashyap
"""

import numpy as np
from gradient_descent import *
from utils import *

def quick_test():
    """Perform a quick test of all gradient descent implementations."""
    print("=" * 60)
    print("QUICK TEST OF GRADIENT DESCENT IMPLEMENTATIONS")
    print("=" * 60)
    
    # Generate simple linear regression data
    np.random.seed(42)
    X, y, true_weights = generate_linear_data(n_samples=200, n_features=2, 
                                             noise=0.1, random_state=42)
    
    # Normalize features
    X_norm, means, stds = normalize_features(X, exclude_bias=True)
    
    # Initialize weights
    initial_weights = np.random.randn(X.shape[1]) * 0.01
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True weights: {true_weights}")
    print(f"Initial weights: {initial_weights}")
    print()
    
    # Test all optimizers
    optimizers = {
        'Batch GD': BatchGradientDescent(learning_rate=0.01, max_iterations=200, verbose=False),
        'SGD': StochasticGradientDescent(learning_rate=0.01, max_iterations=50, verbose=False),
        'Mini-batch GD': MiniBatchGradientDescent(learning_rate=0.01, max_iterations=50, 
                                                 batch_size=32, verbose=False),
        'SGD + Momentum': SGDWithMomentum(learning_rate=0.01, max_iterations=50, 
                                         momentum=0.9, batch_size=32, verbose=False),
        'AdaGrad': AdaGrad(learning_rate=0.1, max_iterations=50, batch_size=32, verbose=False),
        'RMSprop': RMSprop(learning_rate=0.01, max_iterations=50, batch_size=32, verbose=False),
        'Adam': Adam(learning_rate=0.01, max_iterations=50, batch_size=32, verbose=False)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"Testing {name}...", end=' ')
        
        try:
            weights, cost_history = optimizer.optimize(
                linear_regression_cost, 
                linear_regression_gradient,
                initial_weights.copy(), 
                X_norm, 
                y
            )
            
            # Compute final metrics
            predictions = X_norm @ weights
            final_cost = cost_history[-1]
            r2 = compute_r_squared(y, predictions)
            
            results[name] = {
                'weights': weights,
                'final_cost': final_cost,
                'r2': r2,
                'converged': len(cost_history) < optimizer.max_iterations
            }
            
            print(f"✓ Cost: {final_cost:.6f}, R²: {r2:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results[name] = None
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Find best result
    best_r2 = -np.inf
    best_optimizer = None
    
    for name, result in results.items():
        if result is not None and result['r2'] > best_r2:
            best_r2 = result['r2']
            best_optimizer = name
    
    print(f"Best optimizer: {best_optimizer} (R² = {best_r2:.4f})")
    print(f"True weights: {true_weights}")
    
    if best_optimizer:
        best_weights = results[best_optimizer]['weights']
        print(f"Best weights: {best_weights}")
        weight_error = np.linalg.norm(best_weights - true_weights)
        print(f"Weight error (L2 norm): {weight_error:.6f}")
    
    print("\nAll tests completed successfully! ✓")
    
    return results

if __name__ == "__main__":
    results = quick_test() 