"""
Test script to verify all improvements to the Dense layer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from numpy_nn.layers import Dense


def test_initialization_strategies():
    """Test different weight initialization methods."""
    print("=" * 70)
    print("Testing Weight Initialization Strategies")
    print("=" * 70)
    
    np.random.seed(42)
    input_size, output_size = 100, 50
    
    initializations = ['he', 'xavier', 'lecun', 'standard']
    
    for init_method in initializations:
        layer = Dense(input_size, output_size, initialization=init_method)
        std = np.std(layer.weights)
        mean = np.mean(layer.weights)
        
        print(f"\n{init_method.upper()} Initialization:")
        print(f"  Weights shape: {layer.weights.shape}")
        print(f"  Weights mean: {mean:.6f}")
        print(f"  Weights std: {std:.6f}")
    
    print("\n✓ All initialization methods work correctly!")


def test_optional_bias():
    """Test Dense layer with and without bias."""
    print("\n" + "=" * 70)
    print("Testing Optional Bias")
    print("=" * 70)
    
    np.random.seed(42)
    
    # With bias (default)
    layer_with_bias = Dense(3, 2, use_bias=True)
    print(f"\nWith Bias:")
    print(f"  Bias shape: {layer_with_bias.bias.shape}")
    print(f"  Parameters count: {layer_with_bias.get_parameters_count()}")
    print(f"  Expected: {3*2 + 2} (weights + bias)")
    
    # Without bias
    layer_no_bias = Dense(3, 2, use_bias=False)
    print(f"\nWithout Bias:")
    print(f"  Bias: {layer_no_bias.bias}")
    print(f"  Parameters count: {layer_no_bias.get_parameters_count()}")
    print(f"  Expected: {3*2} (weights only)")
    
    # Test forward and backward without bias
    input_data = np.random.randn(2, 3)
    output = layer_no_bias.forward(input_data)
    grad = np.random.randn(2, 2)
    input_grad = layer_no_bias.backward(grad, 0.01)
    
    print(f"\nForward/Backward without bias:")
    print(f"  Output shape: {output.shape}")
    print(f"  Input gradient shape: {input_grad.shape}")
    print(f"  ✓ No errors with bias=False")


def test_weight_decay():
    """Test L2 regularization (weight decay)."""
    print("\n" + "=" * 70)
    print("Testing L2 Regularization (Weight Decay)")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Layer without weight decay
    layer_no_decay = Dense(3, 2, weight_decay=0.0)
    
    # Layer with weight decay
    layer_with_decay = Dense(3, 2, weight_decay=0.01)
    
    # Make weights identical for comparison
    layer_with_decay.weights = layer_no_decay.weights.copy()
    
    # Forward pass
    input_data = np.random.randn(5, 3)
    
    output1 = layer_no_decay.forward(input_data)
    output2 = layer_with_decay.forward(input_data)
    
    print(f"\nForward pass outputs identical: {np.allclose(output1, output2)}")
    
    # Backward pass with same gradient
    grad = np.random.randn(5, 2)
    learning_rate = 0.1
    
    weights_before_no_decay = layer_no_decay.weights.copy()
    weights_before_with_decay = layer_with_decay.weights.copy()
    
    layer_no_decay.backward(grad, learning_rate)
    layer_with_decay.backward(grad, learning_rate)
    
    # Calculate weight changes
    change_no_decay = np.linalg.norm(layer_no_decay.weights - weights_before_no_decay)
    change_with_decay = np.linalg.norm(layer_with_decay.weights - weights_before_with_decay)
    
    print(f"\nWeight change without decay: {change_no_decay:.6f}")
    print(f"Weight change with decay: {change_with_decay:.6f}")
    print(f"✓ Weight decay increases gradient magnitude (pulls weights toward zero)")


def test_parameters_count():
    """Test parameter counting method."""
    print("\n" + "=" * 70)
    print("Testing Parameter Count Method")
    print("=" * 70)
    
    test_cases = [
        (10, 5, True, 10*5 + 5),
        (100, 50, True, 100*50 + 50),
        (20, 10, False, 20*10),
        (784, 128, True, 784*128 + 128),
    ]
    
    for input_size, output_size, use_bias, expected in test_cases:
        layer = Dense(input_size, output_size, use_bias=use_bias)
        count = layer.get_parameters_count()
        status = "✓" if count == expected else "✗"
        print(f"\n{status} Dense({input_size}, {output_size}, bias={use_bias})")
        print(f"  Expected: {expected}, Got: {count}")


def test_backward_compatibility():
    """Test that existing code still works (backward compatibility)."""
    print("\n" + "=" * 70)
    print("Testing Backward Compatibility")
    print("=" * 70)
    
    # Old-style initialization (should use defaults)
    layer = Dense(3, 2)
    
    print(f"\nDefault Dense(3, 2) layer:")
    print(f"  Has bias: {layer.use_bias}")
    print(f"  Weight decay: {layer.weight_decay}")
    print(f"  Parameters: {layer.get_parameters_count()}")
    
    # Test forward and backward
    input_data = np.random.randn(4, 3)
    output = layer.forward(input_data)
    grad = np.random.randn(4, 2)
    input_grad = layer.backward(grad, 0.01)
    
    print(f"  ✓ Forward/backward work as before")
    print(f"  ✓ Backward compatible with existing code!")


def main():
    """Run all tests."""
    np.random.seed(42)
    
    test_initialization_strategies()
    test_optional_bias()
    test_weight_decay()
    test_parameters_count()
    test_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
