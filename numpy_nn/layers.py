from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
class Layer(ABC):
    def __init__(self) -> None:
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass


class Dense(Layer):
    """
    Fully connected (dense) neural network layer.
    
    Implements: output = input @ weights + bias
    
    Parameters
    ----------
    input_size : int
        Number of input features
    output_size : int
        Number of output features (neurons)
    initialization : str, optional
        Weight initialization method. Options:
        - 'he': He/Kaiming initialization (best for ReLU), std = sqrt(2/input_size)
        - 'xavier': Xavier/Glorot initialization (best for tanh/sigmoid), std = sqrt(1/input_size)
        - 'lecun': LeCun initialization (best for SELU), std = sqrt(1/input_size)
        - 'standard': Basic initialization, std = 1/sqrt(input_size)
        Default: 'he'
    use_bias : bool, optional
        Whether to include bias term. Default: True
    weight_decay : float, optional
        L2 regularization coefficient (weight decay). Default: 0.0 (no regularization)
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        initialization: str = 'he',
        use_bias: bool = True,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        
        # Initialize weights based on chosen strategy
        if initialization == 'he':
            # He/Kaiming initialization - good for ReLU activations
            std = np.sqrt(2.0 / input_size)
        elif initialization == 'xavier':
            # Xavier/Glorot initialization - good for tanh/sigmoid
            std = np.sqrt(1.0 / input_size)
        elif initialization == 'lecun':
            # LeCun initialization - good for SELU
            std = np.sqrt(1.0 / input_size)
        elif initialization == 'standard':
            # Standard initialization
            std = 1.0 / np.sqrt(input_size)
        else:
            raise ValueError(
                f"Unknown initialization: {initialization}. "
                f"Choose from: 'he', 'xavier', 'lecun', 'standard'"
            )
        
        self.weights = np.random.randn(input_size, output_size) * std
        self.bias = np.zeros((1, output_size)) if use_bias else None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the dense layer.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input tensor of shape (batch_size, input_size)
            
        Returns
        -------
        np.ndarray
            Output tensor of shape (batch_size, output_size)
        """
        self.input = input_data
        # Y = X . W + B
        self.output = np.dot(self.input, self.weights)
        if self.use_bias:
            self.output += self.bias
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the dense layer.
        
        Computes gradients and updates weights and bias using gradient descent.
        
        Parameters
        ----------
        output_gradient : np.ndarray
            Gradient of loss w.r.t layer output, shape (batch_size, output_size)
        learning_rate : float
            Learning rate for gradient descent update
            
        Returns
        -------
        np.ndarray
            Gradient of loss w.r.t layer input, shape (batch_size, input_size)
        """
        # 1. Calculate gradient w.r.t Weights: dE/dW = X_transpose . dE/dY
        weights_gradient = np.dot(self.input.T, output_gradient)
        
        # Apply L2 regularization (weight decay) if specified
        if self.weight_decay > 0:
            weights_gradient += self.weight_decay * self.weights
        
        # 2. Calculate gradient w.r.t Input: dE/dX = dE/dY . W_transpose
        input_gradient = np.dot(output_gradient, self.weights.T)

        # 3. Update parameters (Gradient Descent)
        # W_new = W_old - (lr * dE/dW)
        self.weights -= learning_rate * weights_gradient
        
        if self.use_bias:
            self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient
    
    def get_parameters_count(self) -> int:
        """
        Get total number of trainable parameters in this layer.
        
        Returns
        -------
        int
            Total number of parameters (weights + biases)
        """
        params = self.weights.size
        if self.use_bias:
            params += self.bias.size
        return params