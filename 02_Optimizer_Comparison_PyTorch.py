"""
Practical Implementation 2: Optimizer Comparison in Deep Learning (PyTorch)
Feedforward Neural Network on Synthetic Dataset: SGD with Momentum vs. Adam Optimizer
Course Outcome: CO2 (5.00 Marks)
Framework: PyTorch
"""

import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model Definition
class FeedforwardNN(nn.Module):
    def __init__(self, in_features: int = 20, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super(FeedforwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# 2D Model Definition for Decision Boundary Visualization
class Model2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)


def main():
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    
    # 1. Dataset Generation
    X_raw, y_raw = make_classification(
        n_samples=2500, n_features=20, n_informative=15, n_redundant=5,
        n_classes=2, weights=[0.5, 0.5], flip_y=0.03, random_state=SEED
    )
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                  torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), 
                                torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initial weights template to ensure identical starting points
    torch.manual_seed(SEED)
    template_model = FeedforwardNN(in_features=20, hidden_dim1=64, hidden_dim2=32)
    initial_state_dict = copy.deepcopy(template_model.state_dict())

    def train_model(optimizer_type: str, lr: float, optimizer_kwargs: dict, epochs: int = 40, threshold: float = 0.18):
        model = FeedforwardNN(in_features=20, hidden_dim1=64, hidden_dim2=32).to(device)
        model.load_state_dict(copy.deepcopy(initial_state_dict))
        
        if optimizer_type == 'SGD_Momentum':
            optimizer = optim.SGD(model.parameters(), lr=lr, **optimizer_kwargs)
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, **optimizer_kwargs)
            
        criterion = nn.BCEWithLogitsLoss()
        history = {
            'epoch_train_loss': [], 'epoch_val_loss': [],
            'epoch_train_acc': [], 'epoch_val_acc': [],
            'step_loss': [], 'time_per_epoch': [],
            'convergence_epoch': None, 'total_time': 0.0
        }
        
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            model.train()
            r_loss, correct, total = 0.0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
                history['step_loss'].append(loss.item())
                r_loss += loss.item() * len(bx)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == by).sum().item()
                total += len(by)
                
            ep_train_loss = r_loss / total
            ep_train_acc = correct / total
            
            model.eval()
            vr_loss, v_corr, v_tot = 0.0, 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vl = criterion(model(vx), vy)
                    vr_loss += vl.item() * len(vx)
                    v_preds = (torch.sigmoid(model(vx)) >= 0.5).float()
                    v_corr += (v_preds == vy).sum().item()
                    v_tot += len(vy)
                    
            ep_val_loss = vr_loss / v_tot
            ep_val_acc = v_corr / v_tot
            ep_dur = time.time() - ep_start
            
            history['epoch_train_loss'].append(ep_train_loss)
            history['epoch_val_loss'].append(ep_val_loss)
            history['epoch_train_acc'].append(ep_train_acc)
            history['epoch_val_acc'].append(ep_val_acc)
            history['time_per_epoch'].append(ep_dur)
            
            if history['convergence_epoch'] is None and ep_val_loss <= threshold:
                history['convergence_epoch'] = epoch
                
        history['total_time'] = time.time() - start_time
        return model, history

    EPOCHS = 40
    THRESHOLD = 0.18
    print("\n--- Training Model 1: SGD with Momentum (0.9, lr=0.01) ---")
    model_sgd, hist_sgd = train_model('SGD_Momentum', lr=0.01, optimizer_kwargs={'momentum': 0.9}, epochs=EPOCHS, threshold=THRESHOLD)
    print(f"SGD+Momentum: Final Val Loss = {hist_sgd['epoch_val_loss'][-1]:.4f}, Final Val Acc = {hist_sgd['epoch_val_acc'][-1]*100:.2f}%, Time = {hist_sgd['total_time']:.2f}s")

    print("\n--- Training Model 2: Adam Optimizer (lr=0.001) ---")
    model_adam, hist_adam = train_model('Adam', lr=0.001, optimizer_kwargs={'betas': (0.9, 0.999), 'eps': 1e-8}, epochs=EPOCHS, threshold=THRESHOLD)
    print(f"Adam:         Final Val Loss = {hist_adam['epoch_val_loss'][-1]:.4f}, Final Val Acc = {hist_adam['epoch_val_acc'][-1]*100:.2f}%, Time = {hist_adam['total_time']:.2f}s")

    # Visualizations Part 1: Curves
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes[0, 0].plot(epochs_range, hist_sgd['epoch_train_loss'], color='#d62728', lw=2.5, marker='o', markersize=4, label='SGD + Momentum')
    axes[0, 0].plot(epochs_range, hist_adam['epoch_train_loss'], color='#1f77b4', lw=2.5, marker='s', markersize=4, label='Adam')
    axes[0, 0].set_title("Training Loss Convergence Curve", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, hist_sgd['epoch_val_loss'], color='#d62728', lw=2.5, marker='o', markersize=4, label='SGD + Momentum')
    axes[0, 1].plot(epochs_range, hist_adam['epoch_val_loss'], color='#1f77b4', lw=2.5, marker='s', markersize=4, label='Adam')
    axes[0, 1].axhline(y=THRESHOLD, color='gray', linestyle='--', label=f'Threshold ({THRESHOLD})')
    axes[0, 1].set_title("Validation Loss Comparison", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Validation Loss")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs_range, [a * 100 for a in hist_sgd['epoch_val_acc']], color='#d62728', lw=2.5, marker='o', markersize=4, label='SGD + Momentum')
    axes[1, 0].plot(epochs_range, [a * 100 for a in hist_adam['epoch_val_acc']], color='#1f77b4', lw=2.5, marker='s', markersize=4, label='Adam')
    axes[1, 0].set_title("Validation Accuracy (%)", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()

    steps = 250
    axes[1, 1].plot(hist_sgd['step_loss'][:steps], color='#d62728', alpha=0.5, label='SGD + Momentum Steps')
    axes[1, 1].plot(hist_adam['step_loss'][:steps], color='#1f77b4', alpha=0.5, label='Adam Steps')
    axes[1, 1].plot(pd.Series(hist_sgd['step_loss'][:steps]).rolling(10).mean(), color='#8c0000', lw=2.5, label='SGD (MA-10)')
    axes[1, 1].plot(pd.Series(hist_adam['step_loss'][:steps]).rolling(10).mean(), color='#003366', lw=2.5, label='Adam (MA-10)')
    axes[1, 1].set_title("Early Optimization Dynamics (First 250 Steps)", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("Mini-batch Step")
    axes[1, 1].set_ylabel("Step Loss")
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig("optimizer_comparison_curves.png", dpi=300)
    plt.close()
    print("Saved 'optimizer_comparison_curves.png'")

    # Visualizations Part 2: Bar charts
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    labels = ['SGD + Momentum', 'Adam']
    colors = ['#d62728', '#1f77b4']

    vloss = [hist_sgd['epoch_val_loss'][-1], hist_adam['epoch_val_loss'][-1]]
    axes[0].bar(labels, vloss, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[0].set_title("Final Validation Loss (Lower is Better)", fontweight='bold')
    for i, v in enumerate(vloss): axes[0].text(i, v+0.005, f"{v:.4f}", ha='center', fontweight='bold')
    axes[0].set_ylim(0, max(vloss)*1.25)

    vacc = [hist_sgd['epoch_val_acc'][-1]*100, hist_adam['epoch_val_acc'][-1]*100]
    axes[1].bar(labels, vacc, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[1].set_title("Final Validation Accuracy (%)", fontweight='bold')
    for i, v in enumerate(vacc): axes[1].text(i, v+0.8, f"{v:.2f}%", ha='center', fontweight='bold')
    axes[1].set_ylim(80, 103)

    conv = [hist_sgd['convergence_epoch'] or EPOCHS, hist_adam['convergence_epoch'] or EPOCHS]
    axes[2].bar(labels, conv, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[2].set_title("Epochs to Loss <= 0.18 (Speed)", fontweight='bold')
    for i, v in enumerate(conv): axes[2].text(i, v+0.5, f"Epoch {v}", ha='center', fontweight='bold')
    axes[2].set_ylim(0, max(conv)*1.35)

    ttime = [hist_sgd['total_time'], hist_adam['total_time']]
    axes[3].bar(labels, ttime, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[3].set_title("Total Training Time (s)", fontweight='bold')
    for i, v in enumerate(ttime): axes[3].text(i, v+0.04, f"{v:.2f}s", ha='center', fontweight='bold')
    axes[3].set_ylim(0, max(ttime)*1.35)

    plt.tight_layout()
    plt.savefig("optimizer_comparison_metrics_bar.png", dpi=300)
    plt.close()
    print("Saved 'optimizer_comparison_metrics_bar.png'")

    # Visualizations Part 3: Decision Surfaces
    X_2d, y_2d = make_moons(n_samples=500, noise=0.22, random_state=SEED)
    X_2d_train, X_2d_val, y_2d_train, y_2d_val = train_test_split(X_2d, y_2d, test_size=0.2, random_state=SEED)
    scaler_2d = StandardScaler()
    X_2d_train = scaler_2d.fit_transform(X_2d_train)
    X_2d_val = scaler_2d.transform(X_2d_val)

    train_2d_loader = DataLoader(
        TensorDataset(torch.tensor(X_2d_train, dtype=torch.float32), torch.tensor(y_2d_train, dtype=torch.float32).unsqueeze(1)),
        batch_size=32, shuffle=True
    )
    torch.manual_seed(SEED)
    w_2d_init = copy.deepcopy(Model2D().state_dict())

    def train_2d(opt_type, lr, kwargs, epochs=35):
        m = Model2D().to(device)
        m.load_state_dict(copy.deepcopy(w_2d_init))
        crit = nn.BCEWithLogitsLoss()
        opt = optim.SGD(m.parameters(), lr=lr, **kwargs) if opt_type=='SGD' else optim.Adam(m.parameters(), lr=lr, **kwargs)
        for _ in range(epochs):
            m.train()
            for bx, by in train_2d_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(m(bx), by).backward()
                opt.step()
        return m

    m_2d_sgd = train_2d('SGD', lr=0.03, kwargs={'momentum': 0.9}, epochs=35)
    m_2d_adam = train_2d('Adam', lr=0.005, kwargs={'betas': (0.9, 0.999)}, epochs=35)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    xx, yy = np.meshgrid(np.linspace(X_2d_val[:, 0].min()-0.6, X_2d_val[:, 0].max()+0.6, 250),
                         np.linspace(X_2d_val[:, 1].min()-0.6, X_2d_val[:, 1].max()+0.6, 250))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    for ax, model_obj, opt_title in zip([ax1, ax2], [m_2d_sgd, m_2d_adam], ['SGD with Momentum', 'Adam Optimizer']):
        model_obj.eval()
        with torch.no_grad():
            preds_grid = torch.sigmoid(model_obj(grid_tensor)).cpu().numpy().reshape(xx.shape)
        cf = ax.contourf(xx, yy, preds_grid, levels=40, cmap='RdBu_r', alpha=0.75)
        ax.contour(xx, yy, preds_grid, levels=[0.5], colors='black', linewidths=2.2, linestyles='--')
        plt.colorbar(cf, ax=ax).set_label('$P(Y=1)$')
        ax.scatter(X_2d_val[y_2d_val==0, 0], X_2d_val[y_2d_val==0, 1], c='blue', edgecolors='k', s=55, label='Class 0')
        ax.scatter(X_2d_val[y_2d_val==1, 0], X_2d_val[y_2d_val==1, 1], c='red', edgecolors='k', s=55, label='Class 1')
        ax.set_title(f"Decision Surface: {opt_title}", fontweight='bold')
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig("optimizer_decision_surfaces.png", dpi=300)
    plt.close()
    print("Saved 'optimizer_decision_surfaces.png'")

    # Summary Table
    summary_data = {
        "Optimizer": ["SGD + Momentum", "Adam"],
        "Hyperparameters": ["lr=0.01, momentum=0.9", "lr=0.001, betas=(0.9, 0.999)"],
        "Initial Loss": [f"{hist_sgd['epoch_train_loss'][0]:.4f}", f"{hist_adam['epoch_train_loss'][0]:.4f}"],
        "Final Train Loss": [f"{hist_sgd['epoch_train_loss'][-1]:.4f}", f"{hist_adam['epoch_train_loss'][-1]:.4f}"],
        "Final Val Loss": [f"{hist_sgd['epoch_val_loss'][-1]:.4f}", f"{hist_adam['epoch_val_loss'][-1]:.4f}"],
        "Final Val Accuracy": [f"{hist_sgd['epoch_val_acc'][-1]*100:.2f}%", f"{hist_adam['epoch_val_acc'][-1]*100:.2f}%"],
        "Convergence Epoch": [f"Epoch {hist_sgd['convergence_epoch']}", f"Epoch {hist_adam['convergence_epoch']}"],
        "Total Time": [f"{hist_sgd['total_time']:.2f} s", f"{hist_adam['total_time']:.2f} s"]
    }
    df = pd.DataFrame(summary_data)
    print("\n" + "=" * 90)
    print("                     SUMMARY COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print("\nAll Program 2 tasks completed successfully!\n")


if __name__ == "__main__":
    main()
