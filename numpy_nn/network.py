from typing import List, Optional, Tuple, Callable
from .layers import Layer
from .losses import Loss
import numpy as np


class Network:
    """
    A Sequential neural network container.
    
    Stacks layers sequentially and provides training capabilities.
    Similar to PyTorch's nn.Sequential and Keras Sequential model.
    
    Parameters
    ----------
    layers : List[Layer]
        List of layers to stack sequentially
        
    Example
    -------
    >>> model = Network([
    ...     Dense(784, 128),
    ...     ReLU(),
    ...     Dense(128, 10)
    ... ])
    >>> model.train(X_train, y_train, MSE(), epochs=100, learning_rate=0.01)
    """
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.training_history = {
            'loss': [],
            'val_loss': []
        }

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run forward pass through all layers (inference mode).
        
        Parameters
        ----------
        input_data : np.ndarray
            Input data of shape (batch_size, input_features)
            
        Returns
        -------
        np.ndarray
            Network output of shape (batch_size, output_features)
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        loss_fn: Loss, 
        epochs: int, 
        learning_rate: float, 
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 100
    ) -> dict:
        """
        Train the network using Gradient Descent.
        
        Performs batch gradient descent (full batch) or mini-batch gradient
        descent if batch_size is specified.
        
        Parameters
        ----------
        x_train : np.ndarray
            Training data of shape (n_samples, n_features)
        y_train : np.ndarray
            Training labels of shape (n_samples, n_outputs)
        loss_fn : Loss
            Loss function to optimize
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for gradient descent
        x_val : np.ndarray, optional
            Validation data for monitoring generalization
        y_val : np.ndarray, optional
            Validation labels
        batch_size : int, optional
            Mini-batch size. If None, uses full batch gradient descent.
        verbose : bool, optional
            Whether to print training progress. Default: True
        log_interval : int, optional
            Print loss every N epochs. Default: 100
            
        Returns
        -------
        dict
            Training history with 'loss' and 'val_loss' (if validation data provided)
            
        Notes
        -----
        The backward pass iterates through layers in REVERSE order.
        This is essential for correct backpropagation:
        - Forward: Input → Layer1 → Layer2 → ... → Output
        - Backward: Loss ← Layer1 ← Layer2 ← ... ← Output
        
        Without reversed(), the chain rule would be broken and training
        would fail (shape mismatches and incorrect gradients).
        """
        # Reset history
        self.training_history = {'loss': [], 'val_loss': []}
        
        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch or full-batch training
            if batch_size is None:
                # Full batch gradient descent
                batches = [(x_train, y_train)]
            else:
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)
                batches = [
                    (x_train[indices[i:i+batch_size]], 
                     y_train[indices[i:i+batch_size]])
                    for i in range(0, n_samples, batch_size)
                ]
            
            # Train on each batch
            for x_batch, y_batch in batches:
                # 1. Forward Pass
                output = self.predict(x_batch)
                
                # 2. Compute Loss
                loss = loss_fn.forward(output, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # 3. Backward Pass
                # First, compute gradient of loss w.r.t network output
                grad = loss_fn.backward(output, y_batch)
                
                # Then propagate backward through all layers in REVERSE order
                # This is CRITICAL for correct backpropagation!
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            
            # Average loss over batches
            avg_loss = epoch_loss / n_batches
            self.training_history['loss'].append(avg_loss)
            
            # Validation loss
            if x_val is not None and y_val is not None:
                val_output = self.predict(x_val)
                val_loss = loss_fn.forward(val_output, y_val)
                self.training_history['val_loss'].append(val_loss)
            
            # Logging
            if verbose and (epoch + 1) % log_interval == 0:
                log_msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}"
                if x_val is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                print(log_msg)
        
        return self.training_history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, loss_fn: Loss) -> float:
        """
        Evaluate the network on test data.
        
        Parameters
        ----------
        x_test : np.ndarray
            Test data
        y_test : np.ndarray
            Test labels
        loss_fn : Loss
            Loss function for evaluation
            
        Returns
        -------
        float
            Test loss
        """
        predictions = self.predict(x_test)
        return loss_fn.forward(predictions, y_test)
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters in the network.
        
        Returns
        -------
        int
            Total parameter count
        """
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'get_parameters_count'):
                total += layer.get_parameters_count()
        return total
    
    def __repr__(self) -> str:
        """String representation of the network."""
        layer_strs = [f"  ({i}): {layer.__class__.__name__}" 
                      for i, layer in enumerate(self.layers)]
        return f"Network(\n" + "\n".join(layer_strs) + "\n)"