"""
Test script for improved and new activation functions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from numpy_nn.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax


def test_leaky_relu():
    """Test LeakyReLU activation."""
    print("=" * 70)
    print("Testing LeakyReLU Activation")
    print("=" * 70)
    
    alpha = 0.01
    leaky_relu = LeakyReLU(alpha=alpha)
    
    # Test forward pass
    input_data = np.array([[-10, -1, 0, 1, 10]])
    output = leaky_relu.forward(input_data)
    
    expected = np.array([[-10 * alpha, -1 * alpha, 0, 1, 10]])
    
    print(f"\nForward Pass (alpha={alpha}):")
    print(f"  Input:    {input_data}")
    print(f"  Output:   {output}")
    print(f"  Expected: {expected}")
    assert np.allclose(output, expected), "LeakyReLU forward failed!"
    print(f"  ✓ Forward pass correct")
    
    # Test backward pass
    grad_output = np.ones_like(input_data)
    grad_input = leaky_relu.backward(grad_output)
    
    expected_grad = np.array([[alpha, alpha, alpha, 1, 1]])
    
    print(f"\nBackward Pass:")
    print(f"  Output gradient: {grad_output}")
    print(f"  Input gradient:  {grad_input}")
    print(f"  Expected:        {expected_grad}")
    assert np.allclose(grad_input, expected_grad), "LeakyReLU backward failed!"
    print(f"  ✓ Backward pass correct")
    
    # Test alpha validation
    try:
        LeakyReLU(alpha=0)
        assert False, "Should reject alpha=0"
    except ValueError:
        print(f"  ✓ Alpha validation works (rejects invalid values)")


def test_tanh():
    """Test Tanh activation."""
    print("\n" + "=" * 70)
    print("Testing Tanh Activation")
    print("=" * 70)
    
    tanh = Tanh()
    
    # Test forward pass
    input_data = np.array([[-2, -1, 0, 1, 2]])
    output = tanh.forward(input_data)
    
    expected = np.tanh(input_data)
    
    print(f"\nForward Pass:")
    print(f"  Input:    {input_data}")
    print(f"  Output:   {output}")
    print(f"  Expected: {expected}")
    assert np.allclose(output, expected), "Tanh forward failed!"
    print(f"  ✓ Forward pass correct")
    
    # Test backward pass
    grad_output = np.ones_like(input_data)
    grad_input = tanh.backward(grad_output)
    
    expected_grad = 1 - np.tanh(input_data) ** 2
    
    print(f"\nBackward Pass:")
    print(f"  Input gradient:  {grad_input}")
    print(f"  Expected:        {expected_grad}")
    assert np.allclose(grad_input, expected_grad), "Tanh backward failed!"
    print(f"  ✓ Backward pass correct")
    
    # Test output range
    extreme_input = np.array([[-100, -10, 0, 10, 100]])
    output = tanh.forward(extreme_input)
    assert np.all(output >= -1) and np.all(output <= 1), "Tanh output out of range!"
    print(f"  ✓ Output correctly bounded in [-1, 1]")


def test_softmax():
    """Test Softmax activation."""
    print("\n" + "=" * 70)
    print("Testing Softmax Activation")
    print("=" * 70)
    
    softmax = Softmax()
    
    # Test single sample
    input_data = np.array([[1, 2, 3, 4, 5]])
    output = softmax.forward(input_data)
    
    print(f"\nForward Pass (single sample):")
    print(f"  Input:  {input_data}")
    print(f"  Output: {output}")
    print(f"  Sum:    {np.sum(output, axis=-1)}")
    
    # Check probability distribution properties
    assert np.allclose(np.sum(output, axis=-1), 1.0), "Softmax doesn't sum to 1!"
    assert np.all(output >= 0) and np.all(output <= 1), "Softmax output out of range!"
    print(f"  ✓ Outputs sum to 1.0")
    print(f"  ✓ All outputs in [0, 1]")
    
    # Test batch processing
    batch_input = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    batch_output = softmax.forward(batch_input)
    
    print(f"\nForward Pass (batch):")
    print(f"  Input shape:  {batch_input.shape}")
    print(f"  Output shape: {batch_output.shape}")
    print(f"  Row sums:     {np.sum(batch_output, axis=-1)}")
    
    assert np.allclose(np.sum(batch_output, axis=-1), 1.0), "Batch softmax rows don't sum to 1!"
    print(f"  ✓ Each row sums to 1.0")
    
    # Test numerical stability with extreme values
    extreme_input = np.array([[1000, 0, -1000]])
    output = softmax.forward(extreme_input)
    
    print(f"\nNumerical Stability Test:")
    print(f"  Extreme input: {extreme_input}")
    print(f"  Output:        {output}")
    
    assert not np.any(np.isnan(output)), "Softmax produced NaN!"
    assert not np.any(np.isinf(output)), "Softmax produced Inf!"
    assert np.allclose(np.sum(output), 1.0), "Softmax with extreme values doesn't sum to 1!"
    print(f"  ✓ No NaN or Inf with extreme values")
    print(f"  ✓ Numerically stable")


def test_relu_output_caching():
    """Test that ReLU now caches output properly."""
    print("\n" + "=" * 70)
    print("Testing ReLU Output Caching")
    print("=" * 70)
    
    relu = ReLU()
    input_data = np.array([[1, -2, 3, -4, 5]])
    output = relu.forward(input_data)
    
    # Check that output is cached
    assert hasattr(relu, 'output'), "ReLU doesn't cache output!"
    assert np.array_equal(relu.output, output), "Cached output doesn't match returned output!"
    print(f"  ✓ ReLU now properly caches output")


def test_gradient_flow():
    """Test gradient flow through all activations."""
    print("\n" + "=" * 70)
    print("Testing Gradient Flow")
    print("=" * 70)
    
    activations = [
        ('ReLU', ReLU()),
        ('LeakyReLU', LeakyReLU(alpha=0.01)),
        ('Sigmoid', Sigmoid()),
        ('Tanh', Tanh()),
        ('Softmax', Softmax())
    ]
    
    input_data = np.random.randn(4, 10)
    grad_output = np.random.randn(4, 10)
    
    print(f"\nInput shape: {input_data.shape}")
    print(f"Gradient shape: {grad_output.shape}\n")
    
    for name, activation in activations:
        output = activation.forward(input_data)
        grad_input = activation.backward(grad_output)
        
        # Check shapes
        assert output.shape == input_data.shape, f"{name} output shape mismatch!"
        assert grad_input.shape == input_data.shape, f"{name} gradient shape mismatch!"
        
        # Check for NaN or Inf
        assert not np.any(np.isnan(output)), f"{name} forward produced NaN!"
        assert not np.any(np.isnan(grad_input)), f"{name} backward produced NaN!"
        assert not np.any(np.isinf(output)), f"{name} forward produced Inf!"
        assert not np.any(np.isinf(grad_input)), f"{name} backward produced Inf!"
        
        print(f"  ✓ {name:12s} - shapes match, no NaN/Inf")


def test_comparison_relu_vs_leaky():
    """Compare ReLU and LeakyReLU behavior."""
    print("\n" + "=" * 70)
    print("Comparing ReLU vs LeakyReLU")
    print("=" * 70)
    
    relu = ReLU()
    leaky = LeakyReLU(alpha=0.01)
    
    input_data = np.array([[-5, -1, 0, 1, 5]])
    
    relu_out = relu.forward(input_data)
    leaky_out = leaky.forward(input_data)
    
    print(f"\nInput:       {input_data}")
    print(f"ReLU:        {relu_out}")
    print(f"LeakyReLU:   {leaky_out}")
    
    grad = np.ones_like(input_data)
    
    relu_grad = relu.backward(grad)
    leaky_grad = leaky.backward(grad)
    
    print(f"\nGradient (ReLU):      {relu_grad}")
    print(f"Gradient (LeakyReLU): {leaky_grad}")
    
    # LeakyReLU should have non-zero gradients for negative inputs
    negative_mask = input_data < 0
    assert np.all(relu_grad[negative_mask] == 0), "ReLU gradient should be 0 for negatives"
    assert np.all(leaky_grad[negative_mask] != 0), "LeakyReLU gradient should be non-zero for negatives"
    
    print(f"\n  ✓ LeakyReLU prevents dying neurons (non-zero negative gradients)")


def test_sigmoid_tanh_relationship():
    """Verify mathematical relationship: sigmoid(x) = (tanh(x/2) + 1) / 2."""
    print("\n" + "=" * 70)
    print("Testing Sigmoid-Tanh Relationship")
    print("=" * 70)
    
    sigmoid = Sigmoid()
    tanh = Tanh()
    
    x = np.array([[-4, -2, 0, 2, 4]])
    
    sig_out = sigmoid.forward(x)
    tanh_out = tanh.forward(x / 2)
    
    # sigmoid(x) = (tanh(x/2) + 1) / 2
    expected = (tanh_out + 1) / 2
    
    print(f"\nInput:                    {x}")
    print(f"Sigmoid(x):               {sig_out}")
    print(f"(Tanh(x/2) + 1) / 2:      {expected}")
    print(f"Match: {np.allclose(sig_out, expected)}")
    
    assert np.allclose(sig_out, expected, rtol=1e-6), "Sigmoid-Tanh relationship violated!"
    print(f"  ✓ Mathematical relationship verified")


def main():
    """Run all tests."""
    np.random.seed(42)
    
    test_leaky_relu()
    test_tanh()
    test_softmax()
    test_relu_output_caching()
    test_gradient_flow()
    test_comparison_relu_vs_leaky()
    test_sigmoid_tanh_relationship()
    
    print("\n" + "=" * 70)
    print("ALL IMPROVEMENT TESTS PASSED! ✓")
    print("=" * 70)
    print("\nSummary of Improvements:")
    print("  1. ✓ LeakyReLU added (prevents dying neurons)")
    print("  2. ✓ Tanh added (zero-centered activation)")
    print("  3. ✓ Softmax added (multi-class classification)")
    print("  4. ✓ ReLU now caches output")
    print("  5. ✓ Comprehensive docstrings added")
    print("  6. ✓ Numerical stability verified")
    print("  7. ✓ Input validation for parametric activations")
    print("=" * 70)


if __name__ == "__main__":
    main()
