"""
NumPyNet - A lightweight, educational deep learning framework.

This package provides a simple, pure NumPy implementation of neural networks
designed for educational purposes and understanding deep learning fundamentals.
"""

__version__ = "0.1.0"
__author__ = "NumPyNet Contributors"

# Import core components for easy access
from numpy_nn.layers import Layer, Dense
from numpy_nn.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from numpy_nn.losses import Loss, MSE, CrossEntropy, BinaryCrossEntropy, SoftmaxCrossEntropy
from numpy_nn.network import Network

__all__ = [
    # Core base classes
    "Layer",
    "Loss",
    # Layers
    "Dense",
    # Activations
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Losses
    "MSE",
    "CrossEntropy",
    "BinaryCrossEntropy",
    "SoftmaxCrossEntropy",
    # Network
    "Network",
]
