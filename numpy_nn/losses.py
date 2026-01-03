import numpy as np


class Loss:
    """Base class for loss functions."""
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute loss value.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            True values
            
        Returns
        -------
        float
            Loss value
        """
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t predictions.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            True values
            
        Returns
        -------
        np.ndarray
            Gradient w.r.t predictions
        """
        raise NotImplementedError


class MSE(Loss):
    """
    Mean Squared Error loss.
    
    L = mean((y_pred - y_true)²)
    
    Used for regression tasks where we predict continuous values.
    
    Reduction:
    - Uses 'mean' reduction: divides by total number of elements
    - Matches PyTorch's MSELoss(reduction='mean')
    
    Gradient:
    - dL/dy_pred = 2 * (y_pred - y_true) / N
    - Where N = total number of elements (batch_size * num_features)
    
    Use case: Regression problems (predicting house prices, temperatures, etc.)
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute MSE loss.
        
        Returns mean of squared differences across ALL elements.
        """
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute MSE gradient.
        
        Derivative: 2 * (y_pred - y_true) / N
        where N is total number of elements (not just batch_size).
        
        This matches PyTorch's reduction='mean' behavior.
        """
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    """
    Categorical Cross Entropy loss.
    
    L = -mean(sum(y_true * log(y_pred)))
    
    Used for multi-class classification. Expects:
    - y_pred: Probability distribution (e.g., from Softmax), shape (batch, classes)
    - y_true: One-hot encoded labels, shape (batch, classes)
    
    Numerical Stability:
    - Clips predictions to [1e-15, 1 - 1e-15] to prevent log(0)
    - Without clipping, log(0) = -inf would break training
    
    Important: Usually paired with Softmax activation.
    
    Gradient:
    - dL/dy_pred = -y_true / y_pred / batch_size
    
    Note: When combined with Softmax, the gradient simplifies to
    (y_pred - y_true). Consider using SoftmaxCrossEntropy for this.
    
    Use case: Multi-class classification (image classification, NLP tasks)
    """
    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize CrossEntropy loss.
        
        Parameters
        ----------
        epsilon : float, optional
            Small constant for numerical stability. Default: 1e-15
            Clips predictions to [epsilon, 1 - epsilon]
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss with numerical stability.
        
        Clips predictions to prevent log(0) which would give -inf.
        
        Example:
        - Perfect prediction: y_pred=[0, 1, 0], y_true=[0, 1, 0] → loss ≈ 0
        - Wrong prediction: y_pred=[1, 0, 0], y_true=[0, 1, 0] → loss ≈ 34.5
        """
        # Clip to prevent log(0) which gives -inf
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        # Calculate loss: -mean(sum(y_true * log(y_pred)))
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute cross-entropy gradient with numerical stability.
        
        Derivative: -y_true / y_pred / batch_size
        
        Clips predictions to prevent division by zero.
        """
        # Clip for stability
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        # Derivative: -y_true / y_pred, averaged over batch
        return -y_true / y_pred_clipped / y_true.shape[0]


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy loss.
    
    L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    Used for binary classification (2 classes).
    
    Expects:
    - y_pred: Probability of positive class (e.g., from Sigmoid), shape (batch, 1)
    - y_true: Binary labels (0 or 1), shape (batch, 1)
    
    Use case: Binary classification (spam detection, sentiment analysis)
    """
    def __init__(self, epsilon: float = 1e-15):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(
            y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute binary cross-entropy gradient.
        
        When used with Sigmoid activation, gradient simplifies to:
        (y_pred - y_true) / batch_size
        """
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (
            (-y_true / y_pred_clipped + (1 - y_true) / (1 - y_pred_clipped)) /
            y_true.shape[0]
        )


class SoftmaxCrossEntropy(Loss):
    """
    Combined Softmax + Cross Entropy loss.
    
    More numerically stable and efficient than applying them separately.
    
    Takes raw logits (before softmax) and computes loss directly.
    The gradient simplifies to: y_pred - y_true (after softmax)
    
    This is the famous simplification used in most deep learning frameworks!
    
    Expects:
    - Input: Raw logits (before softmax), shape (batch, classes)
    - y_true: One-hot encoded labels, shape (batch, classes)
    
    Advantages over separate Softmax + CrossEntropy:
    1. More numerically stable (uses log-sum-exp trick)
    2. More efficient (avoids computing full softmax backward)
    3. Simpler gradient (y_pred - y_true)
    
    Use case: Multi-class classification (recommended over separate Softmax + CE)
    """
    def __init__(self, epsilon: float = 1e-15):
        super().__init__()
        self.epsilon = epsilon
        self.softmax_output = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute softmax + cross-entropy loss from raw logits.
        
        Uses numerically stable implementation with max subtraction.
        
        Parameters
        ----------
        logits : np.ndarray
            Raw logits (before softmax), shape (batch, classes)
        y_true : np.ndarray
            One-hot encoded labels, shape (batch, classes)
        """
        # Numerically stable softmax
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.softmax_output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy loss
        clipped = np.clip(self.softmax_output, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(clipped), axis=1))

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient: simply (y_pred - y_true) / batch_size.
        
        This is the famous simplification! Much easier than computing
        the full Jacobian of softmax followed by cross-entropy gradient.
        
        Parameters
        ----------
        logits : np.ndarray
            Raw logits (same as forward pass)
        y_true : np.ndarray
            One-hot encoded labels
            
        Returns
        -------
        np.ndarray
            Gradient w.r.t logits (before softmax)
        """
        # The gradient is simply: (softmax_output - y_true) / batch_size
        return (self.softmax_output - y_true) / y_true.shape[0]