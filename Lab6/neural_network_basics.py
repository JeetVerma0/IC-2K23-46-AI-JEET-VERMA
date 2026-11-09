"""
Lab 6: Deep Learning using PyTorch
Experiment 3: Neural Network Fundamentals (neural_network_basics.py)

Implements a binary classification neural network using PyTorch torch.nn.Module,
featuring input, hidden (ReLU), and output layers, Adam optimization, autograd,
evaluation metrics, and visualization of training loss and decision boundaries.
"""

from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for strict reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleNeuralNetwork(nn.Module):
    """Feedforward Neural Network with one hidden layer and ReLU activation."""

    def __init__(self, in_features: int = 2, hidden_dim: int = 16, out_features: int = 1):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_dim)   # Input -> Hidden
        self.relu = nn.ReLU()                                    # Activation
        self.output_layer = nn.Linear(hidden_dim, out_features) # Hidden -> Output Logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through input, hidden, and output layers."""
        x = self.input_layer(x)
        x = self.relu(x)
        return self.output_layer(x)


def generate_data(n_samples: int = 600) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Generate synthetic non-linear 2D classification dataset and convert to tensors."""
    X, y = make_moons(n_samples=n_samples, noise=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_t, y_train_t, X_test_t, y_test_t, X_test, y_test


def train_neural_network(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100, lr: float = 0.05) -> list:
    """Train the model using BCEWithLogitsLoss and Adam optimizer, logging epoch losses."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()               # Reset gradients
        outputs = model(X_train)            # Forward pass
        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()                     # Autograd backward pass
        optimizer.step()                    # Update weights
        losses.append(loss.item())

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{epochs}] - Loss: {loss.item():.4f}")

    return losses


def evaluate_model(model: nn.Module, X_test_t: torch.Tensor, y_test: np.ndarray) -> np.ndarray:
    """Evaluate accuracy, precision, recall, and confusion matrix on unseen test data."""
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test_t))
        predictions = (probabilities >= 0.5).int().numpy().flatten()

    acc = accuracy_score(y_test, predictions)
    print("\n=== MODEL EVALUATION (TEST SET) ===")
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    return predictions


def plot_results(losses: list, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, output_path: Path):
    """Plot training loss curve and non-linear decision boundary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Training Loss Progression
    axes[0].plot(range(1, len(losses) + 1), losses, color="#2980b9", lw=2)
    axes[0].set_title("Neural Network Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Decision Boundary on Test Data
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        grid_preds = torch.sigmoid(model(grid_tensor)).reshape(xx.shape).numpy()

    axes[1].contourf(xx, yy, grid_preds >= 0.5, alpha=0.25, cmap="coolwarm")
    scatter = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", s=40)
    axes[1].set_title("Decision Boundary on Test Data")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\nPlot saved successfully to: {output_path}")


def main():
    print("=== LAB 6: PYTORCH NEURAL NETWORK FUNDAMENTALS ===")
    X_train_t, y_train_t, X_test_t, _, X_test, y_test = generate_data()

    model = SimpleNeuralNetwork(in_features=2, hidden_dim=16, out_features=1)
    print("Neural Network Architecture:\n", model)

    losses = train_neural_network(model, X_train_t, y_train_t, epochs=100, lr=0.05)
    evaluate_model(model, X_test_t, y_test)

    output_dir = Path("Lab6/outputs") if Path("Lab6").exists() else Path("outputs")
    plot_results(losses, model, X_test, y_test, output_dir / "neural_network_results.png")


if __name__ == "__main__":
    main()

