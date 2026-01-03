"""
Run all tests for NumPyNet.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run all tests
from tests import test_activations, test_layers, test_losses, test_network

def run_all_tests():
    """Run all test modules."""
    print("=" * 80)
    print("RUNNING NUMPYNET TEST SUITE")
    print("=" * 80)
    
    test_modules = [
        ("Activations", test_activations),
        ("Layers", test_layers),
        ("Losses", test_losses),
        ("Network", test_network)
    ]
    
    for name, module in test_modules:
        print(f"\n{'=' * 80}")
        print(f"Testing {name}...")
        print(f"{'=' * 80}\n")
        module.main()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
