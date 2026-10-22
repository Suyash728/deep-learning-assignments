"""
Practical Implementation 1: Backpropagation Algorithm from Scratch
Multi-Layer Perceptron (MLP) with One Hidden Layer using NumPy
Course Outcome: CO2 (5.00 Marks)
Loss Function: Mean Squared Error (MSE)
Optimization: Standard Gradient Descent
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure visualization styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class MLPBackpropFromScratch:
    """
    Multi-Layer Perceptron with 1 hidden layer implemented from scratch using NumPy.
    - Architecture: Input (d_in) -> Hidden (d_hidden, Sigmoid) -> Output (d_out, Sigmoid)
    - Loss: Mean Squared Error (MSE)
    - Optimizer: Standard Gradient Descent
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 learning_rate: float = 0.5, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        
        # Xavier / Glorot uniform-style initialization
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))
        
        self.cache = {}
        self.grads = {}

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid activation function."""
        z_safe = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_safe))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid with respect to pre-activation z, given a = sigmoid(z)."""
        return a * (1.0 - a)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through hidden layer and output layer.
        Z1 = X @ W1 + b1,   A1 = sigma(Z1)
        Z2 = A1 @ W2 + b2,  A2 = sigma(Z2)
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error: L = 1/(2N) * sum((y_pred - y_true)^2)."""
        N = y_true.shape[0]
        return float((1.0 / (2.0 * N)) * np.sum((y_pred - y_true) ** 2))

    def backward(self, y_true: np.ndarray) -> dict:
        """
        Backpropagation algorithm computing exact analytical gradients via Chain Rule:
        dZ2 = (1/N) * (A2 - y_true) * sigma'(Z2)
        dW2 = A1.T @ dZ2
        db2 = sum(dZ2, axis=0)
        dZ1 = (dZ2 @ W2.T) * sigma'(Z1)
        dW1 = X.T @ dZ1
        db1 = sum(dZ1, axis=0)
        """
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        N = X.shape[0]

        # 1. Output Layer Gradients
        dZ2 = (1.0 / N) * (A2 - y_true) * self.sigmoid_derivative(A2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 2. Hidden Layer Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return self.grads

    def step(self):
        """Update parameters using standard Gradient Descent: theta = theta - lr * grad."""
        self.W1 -= self.lr * self.grads['dW1']
        self.b1 -= self.lr * self.grads['db1']
        self.W2 -= self.lr * self.grads['dW2']
        self.b2 -= self.lr * self.grads['db2']

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000, 
            print_every: int = 1000, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the MLP for a specified number of epochs."""
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            train_loss_history.append(loss)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss_history.append(self.compute_loss(y_val, y_val_pred))
                self.forward(X)  # restore cache
            
            self.backward(y)
            self.step()
            
            if print_every and (epoch % print_every == 0 or epoch == epochs - 1):
                val_str = f" | Val Loss: {val_loss_history[-1]:.6f}" if val_loss_history else ""
                print(f"Epoch {epoch:5d}/{epochs:5d} | Train Loss (MSE): {loss:.6f}{val_str}")
                
        return {'train_loss': train_loss_history, 'val_loss': val_loss_history}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def check_gradients(model: MLPBackpropFromScratch, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-5):
    """Finite difference numerical gradient checking."""
    model.forward(X)
    model.backward(y)
    param_names = ['W1', 'b1', 'W2', 'b2']
    print("=" * 70)
    print(f"{'Parameter':<10} | {'Shape':<15} | {'Relative Difference':<22} | {'Status'}")
    print("=" * 70)
    
    for name in param_names:
        param = getattr(model, name)
        grad_analytic = model.grads['d' + name]
        grad_numeric = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]
            
            param[idx] = orig_val + epsilon
            loss_plus = model.compute_loss(y, model.forward(X))
            
            param[idx] = orig_val - epsilon
            loss_minus = model.compute_loss(y, model.forward(X))
            
            param[idx] = orig_val
            grad_numeric[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
            it.iternext()
            
        model.forward(X)
        model.backward(y)
        
        norm_diff = np.linalg.norm(grad_analytic - grad_numeric)
        norm_sum = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric) + 1e-12
        rel_diff = norm_diff / norm_sum
        status = "PASSED (< 1e-7)" if rel_diff < 1e-7 else "FAILED"
        print(f"{name:<10} | {str(param.shape):<15} | {rel_diff:<22.4e} | {status}")
    print("=" * 70)


def main():
    print("\n" + "=" * 65)
    print(" 1. FINITE DIFFERENCES NUMERICAL GRADIENT VERIFICATION")
    print("=" * 65)
    X_check = np.array([[0.2, 0.7], [0.8, 0.1], [0.5, 0.5]])
    y_check = np.array([[1.0], [0.0], [1.0]])
    test_model = MLPBackpropFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.1, seed=123)
    check_gradients(test_model, X_check, y_check)

    print("\n" + "=" * 65)
    print(" 2. EXPERIMENT 1: CANONICAL XOR PROBLEM")
    print("=" * 65)
    X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_xor = np.array([[0.0], [1.0], [1.0], [0.0]])
    
    mlp_xor = MLPBackpropFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=1.0, seed=42)
    history_xor = mlp_xor.fit(X_xor, y_xor, epochs=4000, print_every=1000)
    preds_xor = mlp_xor.predict(X_xor)
    probs_xor = mlp_xor.predict_proba(X_xor)

    print("\n--- XOR Truth Table vs Model Predictions ---")
    print(f"{'Input (x1, x2)':<16} | {'Target':<8} | {'Prob':<10} | {'Prediction':<10} | {'Status'}")
    print("-" * 60)
    for i in range(len(X_xor)):
        x_str = f"({int(X_xor[i, 0])}, {int(X_xor[i, 1])})"
        target = int(y_xor[i, 0])
        prob = probs_xor[i, 0]
        pred = int(preds_xor[i, 0])
        status = "CORRECT" if pred == target else "WRONG"
        print(f"{x_str:<16} | {target:<8} | {prob:<10.4f} | {pred:<10} | {status}")
    print("-" * 60)

    # XOR Visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    ax1.plot(history_xor['train_loss'], color='#1f77b4', lw=2.5, label='MSE Loss')
    ax1.set_title("XOR Training Convergence Curve", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax1.legend(loc='upper right')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 250), np.linspace(-0.3, 1.3, 250))
    probs_grid = mlp_xor.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cf = ax2.contourf(xx, yy, probs_grid, levels=50, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(cf, ax=ax2).set_label('$\hat{Y} = P(Y=1)$')
    ax2.contour(xx, yy, probs_grid, levels=[0.5], colors='black', linewidths=2.0, linestyles='--')
    colors = {0: 'blue', 1: 'red'}
    markers = {0: 's', 1: 'o'}
    for i in range(len(X_xor)):
        lbl = int(y_xor[i, 0])
        ax2.scatter(X_xor[i, 0], X_xor[i, 1], c=colors[lbl], marker=markers[lbl], s=180, edgecolors='k', lw=1.5,
                    label=f"Class {lbl} ({int(X_xor[i,0])},{int(X_xor[i,1])})")
    ax2.set_title("Learned Decision Boundary (XOR)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Feature $x_1$")
    ax2.set_ylabel("Feature $x_2$")
    ax2.legend(loc='lower left')

    mlp_xor.forward(X_xor)
    hidden_acts = mlp_xor.cache['A1']
    for i in range(len(X_xor)):
        lbl = int(y_xor[i, 0])
        ax3.scatter(hidden_acts[i, 0], hidden_acts[i, 1], c=colors[lbl], marker=markers[lbl], s=200, edgecolors='k', lw=1.5,
                    label=f"Class {lbl} {list(X_xor[i].astype(int))}")
        ax3.annotate(f"({int(X_xor[i,0])},{int(X_xor[i,1])})", (hidden_acts[i,0]+0.02, hidden_acts[i,1]+0.02), weight='bold')
    ax3.set_title("Hidden Feature Space Transformation ($A_1^{(1)}$ vs $A_1^{(2)}$)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Hidden Neuron 1 Activation")
    ax3.set_ylabel("Hidden Neuron 2 Activation")
    ax3.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("mlp_scratch_xor_visualizations.png", dpi=300)
    plt.close()
    print("Saved 'mlp_scratch_xor_visualizations.png'")

    print("\n" + "=" * 65)
    print(" 3. EXPERIMENT 2: SYNTHETIC NOISY MOONS DATASET")
    print("=" * 65)
    X_syn, y_syn = make_moons(n_samples=400, noise=0.20, random_state=42)
    y_syn = y_syn.reshape(-1, 1).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.20, random_state=42, stratify=y_syn)

    mlp_synthetic = MLPBackpropFromScratch(input_dim=2, hidden_dim=8, output_dim=1, learning_rate=0.4, seed=42)
    history_syn = mlp_synthetic.fit(X_train, y_train, epochs=3500, print_every=700, X_val=X_test, y_val=y_test)
    y_test_pred = mlp_synthetic.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}% | Final Test Loss: {history_syn['val_loss'][-1]:.6f}")

    # Synthetic Visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    ax1.plot(history_syn['train_loss'], color='#2ca02c', lw=2.2, label='Train MSE')
    ax1.plot(history_syn['val_loss'], color='#d62728', lw=2.0, ls='--', label='Test MSE')
    ax1.set_title("Train & Test Loss Convergence", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()

    xx, yy = np.meshgrid(np.linspace(X_syn[:, 0].min()-0.5, X_syn[:, 0].max()+0.5, 250),
                         np.linspace(X_syn[:, 1].min()-0.5, X_syn[:, 1].max()+0.5, 250))
    probs_syn = mlp_synthetic.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cf = ax2.contourf(xx, yy, probs_syn, levels=40, cmap='coolwarm', alpha=0.75)
    plt.colorbar(cf, ax=ax2).set_label('$\hat{Y} = P(Class=1)$')
    ax2.contour(xx, yy, probs_syn, levels=[0.5], colors='black', lw=2.2)
    ax2.scatter(X_test[y_test.ravel()==0, 0], X_test[y_test.ravel()==0, 1], c='blue', edgecolors='k', label='Class 0 (Test)')
    ax2.scatter(X_test[y_test.ravel()==1, 0], X_test[y_test.ravel()==1, 1], c='red', edgecolors='k', label='Class 1 (Test)')
    ax2.set_title(f"Non-Linear Decision Boundary (Acc: {test_acc*100:.1f}%)", fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left')

    cm = confusion_matrix(y_test, y_test_pred)
    ax3.imshow(cm, cmap=plt.cm.Blues)
    ax3.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['0', '1'], yticklabels=['0', '1'],
            title="Confusion Matrix (Test Set)", xlabel="Predicted", ylabel="Actual")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("mlp_scratch_synthetic_visualizations.png", dpi=300)
    plt.close()
    print("Saved 'mlp_scratch_synthetic_visualizations.png'")

    print("\n" + "=" * 65)
    print(" 4. LEARNING RATE SENSITIVITY")
    print("=" * 65)
    plt.figure(figsize=(10, 5.5))
    rates = [0.05, 0.20, 0.50, 1.00, 2.50]
    for lr_val in rates:
        m = MLPBackpropFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=lr_val, seed=42)
        h = m.fit(X_xor, y_xor, epochs=2500, print_every=0)
        plt.plot(h['train_loss'], label=f"$\eta={lr_val:.2f}$ (Final Loss: {h['train_loss'][-1]:.4f})", lw=2.2)
    plt.title("Effect of Learning Rate ($\eta$) on Standard GD Convergence", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("mlp_learning_rate_sensitivity.png", dpi=300)
    plt.close()
    print("Saved 'mlp_learning_rate_sensitivity.png'")
    print("\nAll Program 1 tasks completed successfully!\n")


if __name__ == "__main__":
    main()
