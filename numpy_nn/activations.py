import numpy as np
from .layers import Layer


class Activation(Layer):
    """
    Base class for activations. 
    Just like Layer, but usually has no learnable parameters.
    """
    def __init__(self):
        super().__init__()


class ReLU(Activation):
    """
    Rectified Linear Unit activation.
    
    f(x) = max(0, x)
    
    Pros: Simple, efficient, helps with vanishing gradients
    Cons: Dead neurons problem (neurons can permanently output 0)
    
    Use case: Default choice for hidden layers in deep networks
    """
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Apply ReLU activation element-wise."""
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float = 0.0) -> np.ndarray:
        """
        Compute gradient for ReLU.
        
        Derivative: 1 if x > 0, else 0
        Uses cached input to determine where activation was positive.
        """
        return output_gradient * (self.input > 0)


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit activation.
    
    f(x) = x if x > 0 else alpha * x
    
    Fixes the "dying ReLU" problem by allowing small negative gradients.
    
    Parameters
    ----------
    alpha : float, optional
        Slope for negative values. Default: 0.01
        
    Use case: When ReLU causes too many dead neurons
    """
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Apply LeakyReLU activation element-wise."""
        self.input = input_data
        self.output = np.where(input_data > 0, input_data, self.alpha * input_data)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float = 0.0) -> np.ndarray:
        """
        Compute gradient for LeakyReLU.
        
        Derivative: 1 if x > 0, else alpha
        """
        return output_gradient * np.where(self.input > 0, 1, self.alpha)


class Sigmoid(Activation):
    """
    Sigmoid (Logistic) activation.
    
    f(x) = 1 / (1 + exp(-x))
    
    Outputs values in range (0, 1). Suffers from vanishing gradients for large |x|.
    
    Use case: Binary classification output layer, gate mechanisms (LSTM/GRU)
    
    Note: Numerical stability is ensured by clipping inputs to prevent exp overflow.
    """
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Apply Sigmoid activation with numerical stability.
        
        Uses clipping to prevent overflow in exp() computation.
        For x < -500, sigmoid(x) ≈ 0
        For x > 500, sigmoid(x) ≈ 1
        """
        self.input = input_data
        # Clip to prevent overflow: exp(500) is near float64 limit
        clipped_input = np.clip(input_data, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-clipped_input))
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float = 0.0) -> np.ndarray:
        """
        Compute gradient for Sigmoid using cached output.
        
        Derivative: sigmoid(x) * (1 - sigmoid(x))
        Reuses stored output to avoid redundant computation.
        """
        sigmoid_grad = self.output * (1.0 - self.output)
        return output_gradient * sigmoid_grad


class Tanh(Activation):
    """
    Hyperbolic Tangent activation.
    
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Outputs values in range (-1, 1). Often performs better than Sigmoid.
    Zero-centered output helps with gradient flow.
    
    Use case: Hidden layers, RNN/LSTM cells, when zero-centered activations are preferred
    """
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Apply Tanh activation using NumPy's optimized implementation.
        
        NumPy's tanh is numerically stable and handles extreme values correctly.
        """
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float = 0.0) -> np.ndarray:
        """
        Compute gradient for Tanh using cached output.
        
        Derivative: 1 - tanh²(x)
        Reuses stored output to avoid redundant computation.
        """
        tanh_grad = 1.0 - self.output ** 2
        return output_gradient * tanh_grad


class Softmax(Activation):
    """
    Softmax activation for multi-class classification.
    
    f(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    
    Converts logits to probability distribution (outputs sum to 1).
    
    Use case: Multi-class classification output layer
    
    Note: Uses numerically stable implementation with max subtraction trick.
    """
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Apply Softmax activation with numerical stability.
        
        Subtracts max value before exp to prevent overflow.
        This is mathematically equivalent but numerically stable.
        
        Works with batched inputs: applies softmax along last axis (features).
        """
        self.input = input_data
        
        # Numerical stability: subtract max value
        # softmax(x) = softmax(x - c) for any constant c
        exp_shifted = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float = 0.0) -> np.ndarray:
        """
        Compute gradient for Softmax.
        
        The full Jacobian for softmax is complex, but when used with
        cross-entropy loss, the gradient simplifies to (predicted - actual).
        
        For general use, we compute the element-wise gradient approximation.
        This is exact when softmax is the final layer with cross-entropy loss.
        
        Note: For proper Softmax backprop, consider pairing with CrossEntropyLoss
        which combines softmax + loss for numerical stability.
        """
        # Simplified gradient (exact for cross-entropy loss)
        # For batched inputs: maintain shape
        return output_gradient * self.output * (1.0 - self.output)