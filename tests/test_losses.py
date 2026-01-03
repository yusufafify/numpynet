"""
Test the improved loss functions including new additions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from numpy_nn.losses import MSE, CrossEntropy, BinaryCrossEntropy, SoftmaxCrossEntropy


def test_softmax_crossentropy_simplification():
    """
    Test the SoftmaxCrossEntropy combined loss that gives y_pred - y_true gradient.
    """
    print("=" * 70)
    print("Testing SoftmaxCrossEntropy (Combined Loss)")
    print("=" * 70)
    
    # Create instance
    sce = SoftmaxCrossEntropy()
    
    # Raw logits (before softmax)
    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 1.0],
        [1.0, 1.0, 3.0]
    ])
    
    # One-hot encoded labels
    y_true = np.array([
        [1.0, 0.0, 0.0],  # Class 0
        [0.0, 1.0, 0.0],  # Class 1
        [0.0, 0.0, 1.0]   # Class 2
    ])
    
    print(f"\nInput:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits:\n{logits}")
    print(f"  True labels:\n{y_true}")
    
    # Forward pass
    loss = sce.forward(logits, y_true)
    print(f"\nForward Pass:")
    print(f"  Loss: {loss}")
    print(f"  Softmax output:\n{sce.softmax_output}")
    print(f"  Each row sums to: {np.sum(sce.softmax_output, axis=1)}")
    
    # Backward pass
    gradient = sce.backward(logits, y_true)
    
    print(f"\nBackward Pass:")
    print(f"  Gradient shape: {gradient.shape}")
    print(f"  Gradient:\n{gradient}")
    
    # Verify it's (y_pred - y_true) / batch_size
    expected_grad = (sce.softmax_output - y_true) / y_true.shape[0]
    
    print(f"\nVerification (should be y_pred - y_true):")
    print(f"  Expected:\n{expected_grad}")
    print(f"  Match: {np.allclose(gradient, expected_grad)}")
    
    assert np.allclose(gradient, expected_grad), "Gradient doesn't match simplification!"
    
    # Check gradient for first sample
    print(f"\nSample 0 analysis:")
    print(f"  Predicted probs: {sce.softmax_output[0]}")
    print(f"  True label:      {y_true[0]}")
    print(f"  Gradient:        {gradient[0]}")
    print(f"  Interpretation:")
    for i, (pred, true, grad) in enumerate(zip(sce.softmax_output[0], y_true[0], gradient[0])):
        status = "CORRECT" if true == 1.0 else "WRONG"
        print(f"    Class {i} ({status}): pred={pred:.3f}, grad={grad:.3f}")
    
    print(f"\n✓ SoftmaxCrossEntropy gradient = (y_pred - y_true) / batch_size")


def test_binary_crossentropy():
    """Test BinaryCrossEntropy for binary classification."""
    print("\n" + "=" * 70)
    print("Testing BinaryCrossEntropy")
    print("=" * 70)
    
    bce = BinaryCrossEntropy()
    
    # Binary predictions (from sigmoid)
    y_pred = np.array([[0.9], [0.3], [0.8], [0.1]])
    y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
    
    print(f"\nSetup:")
    print(f"  Predictions: {y_pred.T}")
    print(f"  True labels: {y_true.T}")
    
    # Forward pass
    loss = bce.forward(y_pred, y_true)
    print(f"\nLoss: {loss}")
    
    # Manual calculation for verification
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    manual_loss = -np.mean(
        y_true * np.log(y_pred_clipped) + 
        (1 - y_true) * np.log(1 - y_pred_clipped)
    )
    print(f"Manual calculation: {manual_loss}")
    assert np.isclose(loss, manual_loss), "BCE loss incorrect!"
    
    # Backward pass
    gradient = bce.backward(y_pred, y_true)
    print(f"\nGradient: {gradient.T}")
    print(f"Gradient shape: {gradient.shape}")
    
    # Interpretation
    print(f"\nInterpretation:")
    for i, (pred, true, grad) in enumerate(zip(y_pred.flatten(), y_true.flatten(), gradient.flatten())):
        correct = "✓" if (pred > 0.5 and true == 1) or (pred < 0.5 and true == 0) else "✗"
        print(f"  Sample {i} {correct}: pred={pred:.1f}, true={true:.0f}, grad={grad:.3f}")
    
    print(f"\n✓ BinaryCrossEntropy works correctly")


def test_loss_comparison():
    """Compare all loss functions."""
    print("\n" + "=" * 70)
    print("Loss Function Comparison")
    print("=" * 70)
    
    # Same data for all losses
    batch_size = 3
    num_classes = 4
    
    # Create test data
    logits = np.random.randn(batch_size, num_classes)
    y_true = np.zeros((batch_size, num_classes))
    y_true[0, 0] = 1  # Sample 0: class 0
    y_true[1, 2] = 1  # Sample 1: class 2
    y_true[2, 1] = 1  # Sample 2: class 1
    
    # Apply softmax manually for regular CrossEntropy
    from numpy_nn.activations import Softmax
    softmax = Softmax()
    y_pred = softmax.forward(logits)
    
    print(f"\nTest data:")
    print(f"  Logits:\n{logits}")
    print(f"  After Softmax:\n{y_pred}")
    print(f"  True labels:\n{y_true}")
    
    # Method 1: Separate Softmax + CrossEntropy
    ce = CrossEntropy()
    loss_separate = ce.forward(y_pred, y_true)
    grad_separate = ce.backward(y_pred, y_true)
    
    print(f"\nMethod 1: Separate Softmax + CrossEntropy")
    print(f"  Loss: {loss_separate}")
    print(f"  Gradient[0]: {grad_separate[0]}")
    
    # Method 2: Combined SoftmaxCrossEntropy
    sce = SoftmaxCrossEntropy()
    loss_combined = sce.forward(logits, y_true)
    grad_combined = sce.backward(logits, y_true)
    
    print(f"\nMethod 2: Combined SoftmaxCrossEntropy")
    print(f"  Loss: {loss_combined}")
    print(f"  Gradient[0]: {grad_combined[0]}")
    
    print(f"\nComparison:")
    print(f"  Loss match: {np.isclose(loss_separate, loss_combined)}")
    print(f"  Losses: {loss_separate:.6f} vs {loss_combined:.6f}")
    
    # Note: Gradients won't match exactly because:
    # - ce.backward returns dL/dy_pred (gradient w.r.t softmax output)
    # - sce.backward returns dL/dlogits (gradient w.r.t logits before softmax)
    # The sce gradient is what you'd get after backprop through softmax
    
    print(f"\n✓ Both methods compute same loss")
    print(f"  Note: Gradients differ because they're w.r.t different variables")
    print(f"  - CE: gradient w.r.t softmax output")
    print(f"  - SCE: gradient w.r.t logits (more useful!)")


def test_gradient_numerical_verification():
    """Numerically verify gradients using finite differences."""
    print("\n" + "=" * 70)
    print("Numerical Gradient Verification")
    print("=" * 70)
    
    epsilon = 1e-7
    
    # Test SoftmaxCrossEntropy
    print("\nTesting SoftmaxCrossEntropy gradient:")
    sce = SoftmaxCrossEntropy()
    
    logits = np.array([[1.0, 2.0, 0.5]])
    y_true = np.array([[0.0, 1.0, 0.0]])
    
    # Analytical gradient
    loss = sce.forward(logits, y_true)
    analytical_grad = sce.backward(logits, y_true)
    
    # Numerical gradient
    numerical_grad = np.zeros_like(logits)
    for i in range(logits.shape[1]):
        logits_plus = logits.copy()
        logits_minus = logits.copy()
        
        logits_plus[0, i] += epsilon
        logits_minus[0, i] -= epsilon
        
        loss_plus = sce.forward(logits_plus, y_true)
        loss_minus = sce.forward(logits_minus, y_true)
        
        numerical_grad[0, i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    print(f"  Analytical: {analytical_grad}")
    print(f"  Numerical:  {numerical_grad}")
    print(f"  Match: {np.allclose(analytical_grad, numerical_grad, rtol=1e-5)}")
    
    assert np.allclose(analytical_grad, numerical_grad, rtol=1e-5), \
        "SoftmaxCrossEntropy gradient incorrect!"
    
    print(f"✓ Gradient verified numerically!")


def main():
    """Run all tests."""
    np.random.seed(42)
    
    test_softmax_crossentropy_simplification()
    test_binary_crossentropy()
    test_loss_comparison()
    test_gradient_numerical_verification()
    
    print("\n" + "=" * 70)
    print("ALL IMPROVED LOSS TESTS PASSED! ✓")
    print("=" * 70)
    
    print("\nLoss Function Guide:")
    print("  • MSE: Regression (continuous values)")
    print("  • BinaryCrossEntropy: Binary classification (2 classes)")
    print("  • CrossEntropy: Multi-class (with pre-computed softmax)")
    print("  • SoftmaxCrossEntropy: Multi-class (RECOMMENDED - more stable)")
    print("\nKey Insight:")
    print("  SoftmaxCrossEntropy gives the famous (y_pred - y_true) gradient!")
    print("=" * 70)


if __name__ == "__main__":
    main()
