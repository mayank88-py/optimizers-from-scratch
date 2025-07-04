# Gradient Descent from Scratch

A comprehensive implementation of various gradient descent algorithms using NumPy, built from scratch for educational and practical purposes.

## 🚀 Features

This project implements the following gradient descent variants:

### Basic Algorithms
- **Batch Gradient Descent**: Uses the entire dataset for each parameter update
- **Stochastic Gradient Descent (SGD)**: Uses single samples for parameter updates
- **Mini-batch Gradient Descent**: Uses small batches of data for parameter updates

### Advanced Optimizers
- **SGD with Momentum**: Accelerates convergence by adding momentum to parameter updates
- **AdaGrad**: Adapts learning rate based on historical gradients
- **RMSprop**: Improves AdaGrad by using exponential moving average of squared gradients
- **Adam**: Combines momentum and adaptive learning rates for robust optimization

## 📋 Requirements

```
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
```

Install requirements:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

### Quick Start

```python
import numpy as np
from gradient_descent import BatchGradientDescent, Adam
from utils import generate_linear_data, linear_regression_cost, linear_regression_gradient

# Generate synthetic data
X, y, true_weights = generate_linear_data(n_samples=1000, n_features=2, random_state=42)

# Initialize weights
initial_weights = np.random.randn(X.shape[1]) * 0.01

# Create optimizer
optimizer = Adam(learning_rate=0.01, max_iterations=1000, verbose=True)

# Run optimization
weights, cost_history = optimizer.optimize(
    linear_regression_cost, 
    linear_regression_gradient,
    initial_weights, 
    X, 
    y
)

print(f"Final weights: {weights}")
print(f"Final cost: {cost_history[-1]}")
```

### Comparing Multiple Optimizers

```python
from gradient_descent import compare_optimizers, visualize_convergence

# Set up optimizers
optimizers = {
    'Batch GD': BatchGradientDescent(learning_rate=0.01, max_iterations=1000),
    'SGD': StochasticGradientDescent(learning_rate=0.01, max_iterations=200),
    'Adam': Adam(learning_rate=0.01, max_iterations=200, batch_size=64)
}

# Compare optimizers
results = compare_optimizers(
    linear_regression_cost, 
    linear_regression_gradient,
    initial_weights, 
    X, 
    y, 
    optimizers
)

# Visualize convergence
visualize_convergence(results, "Optimizer Comparison")
```

## 📊 Examples

The project includes comprehensive examples demonstrating:

1. **Linear Regression**: Basic regression with different optimizers
2. **Logistic Regression**: Binary classification example
3. **Polynomial Regression**: Regression with regularization
4. **Learning Rate Sensitivity**: How learning rate affects convergence
5. **Batch Size Comparison**: Impact of different batch sizes
6. **Convergence Visualization**: 2D visualization of optimization paths

Run all examples:
```python
from examples import run_all_examples
run_all_examples()
```

## 🧮 Algorithm Details

### 1. Batch Gradient Descent
- **Update Rule**: θ = θ - α∇J(θ)
- **Pros**: Stable convergence, exact gradients
- **Cons**: Slow for large datasets, can get stuck in local minima

### 2. Stochastic Gradient Descent (SGD)
- **Update Rule**: θ = θ - α∇J(θ; x⁽ⁱ⁾, y⁽ⁱ⁾)
- **Pros**: Fast updates, can escape local minima
- **Cons**: Noisy convergence, requires learning rate scheduling

### 3. Mini-batch Gradient Descent
- **Update Rule**: θ = θ - α∇J(θ; X_batch, y_batch)
- **Pros**: Balance between batch and SGD, vectorized operations
- **Cons**: Hyperparameter (batch size) to tune

### 4. SGD with Momentum
- **Update Rules**: 
  - v = βv - α∇J(θ)
  - θ = θ + v
- **Pros**: Faster convergence, reduces oscillations
- **Cons**: Additional hyperparameter (momentum)

### 5. AdaGrad
- **Update Rules**:
  - G = G + ∇J(θ)²
  - θ = θ - α∇J(θ)/√(G + ε)
- **Pros**: Adapts learning rate per parameter
- **Cons**: Learning rate decays too aggressively

### 6. RMSprop
- **Update Rules**:
  - G = βG + (1-β)∇J(θ)²
  - θ = θ - α∇J(θ)/√(G + ε)
- **Pros**: Fixes AdaGrad's learning rate decay
- **Cons**: Hyperparameter tuning required

### 7. Adam
- **Update Rules**:
  - m = β₁m + (1-β₁)∇J(θ)
  - v = β₂v + (1-β₂)∇J(θ)²
  - θ = θ - α(m̂)/(√v̂ + ε)
- **Pros**: Combines momentum and adaptive learning rates
- **Cons**: More hyperparameters, can fail to converge in some cases

## 🏗️ Project Structure

```
SGD-from-scratch/
├── gradient_descent.py    # Main gradient descent implementations
├── utils.py              # Utility functions for common problems
├── examples.py           # Comprehensive examples
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 🎯 Key Classes

### GradientDescentBase
Abstract base class for all optimizers with common functionality:
- Convergence checking
- History tracking
- Verbose output

### Optimizer Classes
All optimizers inherit from `GradientDescentBase` and implement:
- `optimize(cost_function, gradient_function, initial_weights, X, y)`

### Utility Functions
- `generate_linear_data()`: Create synthetic linear regression data
- `generate_classification_data()`: Create synthetic classification data
- `linear_regression_cost()`: MSE cost function
- `linear_regression_gradient()`: MSE gradient function
- `logistic_regression_cost()`: Binary cross-entropy cost function
- `logistic_regression_gradient()`: Logistic regression gradient
- `normalize_features()`: Feature normalization
- `compute_r_squared()`: R² metric calculation
- `compute_accuracy()`: Classification accuracy

## 📈 Performance Tips

1. **Feature Scaling**: Always normalize features for better convergence
2. **Learning Rate**: Start with 0.01 and adjust based on convergence
3. **Batch Size**: Use 32-256 for mini-batch methods
4. **Initialization**: Use small random weights (e.g., `np.random.randn(n) * 0.01`)
5. **Convergence**: Monitor cost function to detect convergence or divergence

## 🔧 Customization

### Adding New Optimizers
1. Inherit from `GradientDescentBase`
2. Implement the `optimize` method
3. Add any optimizer-specific parameters in `__init__`

### Custom Cost Functions
Create functions with signature:
```python
def custom_cost(weights, X, y):
    # Your cost computation
    return cost_value

def custom_gradient(weights, X, y):
    # Your gradient computation
    return gradient_vector
```

## 🎓 Educational Notes

This implementation focuses on:
- **Clarity**: Clean, readable code with detailed comments
- **Completeness**: All major gradient descent variants
- **Modularity**: Easy to extend and modify
- **Visualization**: Built-in plotting for understanding convergence

## 🤝 Contributing

Feel free to contribute by:
- Adding new optimizers
- Improving existing implementations
- Adding more examples
- Fixing bugs

## 📚 References

1. Ruder, S. (2016). An overview of gradient descent optimization algorithms
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
3. Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent

## 📄 License

This project is open source and available under the MIT License. 