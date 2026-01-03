# NumPyNet 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

A lightweight, educational deep learning framework built from scratch using only NumPy. Perfect for understanding neural network fundamentals and backpropagation.

## Features

- Pure NumPy implementation - no external ML frameworks
- PyTorch-inspired Sequential model API
- Multiple initialization strategies (He, Xavier, LeCun)
- Comprehensive test coverage with numerical gradient verification
- Clean, documented code designed for learning

## Installation

```bash
# From GitHub
pip install git+https://github.com/yusufafify/numpynet.git

# From local clone (for development)
git clone https://github.com/yusufafify/numpynet.git
cd numpynet
pip install -e .
```

## Quick Start

```python
import numpy as np
from numpy_nn import Network, Dense, ReLU, Sigmoid, MSE

# Create network
model = Network([
    Dense(2, 8),
    ReLU(),
    Dense(8, 1),
    Sigmoid()
])

# Train on XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model.train(X, y, MSE(), epochs=2000, learning_rate=0.1)

# Predict
predictions = model.predict(X)
```

## Components

**Layers:** Dense (fully connected with He/Xavier/LeCun initialization, L2 regularization)

**Activations:** ReLU, LeakyReLU, Sigmoid, Tanh, Softmax

**Loss Functions:** MSE, BinaryCrossEntropy, CrossEntropy, SoftmaxCrossEntropy

**Network Features:** Mini-batch training, validation monitoring, training history

## Testing

```bash
python run_tests.py
```

## Examples

See `examples/xor_example.py` for a complete working example.

## License

MIT License - Free to use for learning and teaching.

**Note**: This is an educational framework. For production use, please use established frameworks like PyTorch, TensorFlow, or JAX.

