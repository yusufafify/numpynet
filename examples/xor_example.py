import sys
from pathlib import Path

# Add parent directory to path so we can import numpy_nn
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# Import our library
from numpy_nn.network import Network
from numpy_nn.layers import Dense
from numpy_nn.activations import Tanh, Sigmoid
from numpy_nn.losses import MSE

def main():
    print("--------------------------------------------------")
    print("      NumPyNet: Training on XOR Problem")
    print("--------------------------------------------------")

    # 1. Prepare Data (XOR Truth Table)
    # Inputs: (0,0), (0,1), (1,0), (1,1)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    # Targets: 0, 1, 1, 0
    Y = np.array([[0], [1], [1], [0]]).reshape(4, 1)

    # 2. Build the Model
    # Architecture: Input(2) -> Dense(3) -> Tanh -> Dense(1) -> Tanh
    # We use Tanh here because it converges faster than ReLU for this specific tiny problem.
    model = Network([
        Dense(input_size=2, output_size=5, initialization='xavier'),
        Tanh(),
        Dense(input_size=5, output_size=1, initialization='xavier'),
        Sigmoid()
    ])

    print(f"Model Architecture:\n{model}")
    print(f"Total Parameters: {model.get_num_parameters()}")

    # 3. Train
    print("\nStarting Training...")
    history = model.train(
        x_train=X, 
        y_train=Y, 
        loss_fn=MSE(), 
        epochs=5000, 
        learning_rate=0.5,
        verbose=True,
        log_interval=500
    )

    # 4. Predict and Evaluate
    print("\n--------------------------------------------------")
    print("      Final Predictions vs Ground Truth")
    print("--------------------------------------------------")
    
    predictions = model.predict(X)
    
    for i in range(len(X)):
        input_val = X[i]
        true_val = Y[i][0]
        pred_val = predictions[i][0]
        print(f"Input: {input_val} | Truth: {true_val} | Pred: {pred_val:.4f}")

    # 5. Visualizing the Decision Boundary (Optional)
    plot_decision_boundary(model, X, Y)

def plot_decision_boundary(model, X, Y):
    """Visualizes how the network separates the space."""
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the whole grid
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_data)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.Spectral, edgecolors='k', s=100)
    plt.title("XOR Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

if __name__ == "__main__":
    main()