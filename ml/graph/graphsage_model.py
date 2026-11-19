"""
ml/graph/graphsage_model.py
GraphSAGE model and trainer for diabetes prediction.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data
    TGP_AVAILABLE = True
except ImportError:
    TGP_AVAILABLE = False
    print("[GraphSAGE] torch_geometric not available. Install it via pip install torch-geometric")

from config.config import (
    GRAPHSAGE_HIDDEN_DIM, GRAPHSAGE_NUM_LAYERS, GRAPHSAGE_DROPOUT,
    GRAPHSAGE_LR, GRAPHSAGE_EPOCHS, GRAPHSAGE_BATCH_SIZE,
    ARTIFACTS_DIR, RESULTS_DIR, RANDOM_SEED
)


class GraphSAGEClassifier(nn.Module):
    """
    2-layer GraphSAGE classifier for binary diabetes prediction.

    Architecture:
        Input → SAGEConv(hidden) → ReLU → Dropout
              → SAGEConv(hidden) → ReLU → Dropout
              → Linear(1) → Sigmoid (probability)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = GRAPHSAGE_HIDDEN_DIM,
        num_layers: int = GRAPHSAGE_NUM_LAYERS,
        dropout: float = GRAPHSAGE_DROPOUT,
    ):
        super().__init__()
        if not TGP_AVAILABLE:
            raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")

        hidden_channels = 128
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.dropout = nn.Dropout(p=0.2)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip-connection from original node features."""
        x_orig = x
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            if i < self.num_layers - 1:
                h = self.dropout(h)
        h = self.dropout(h)
        # Concatenate graph neighborhood embeddings with original patient features
        combined = torch.cat([h, x_orig], dim=-1)
        return self.classifier(combined).squeeze(-1)

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """Return probability of class 1 for each node."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()


class GraphSAGETrainer:
    """
    Trains GraphSAGEClassifier on a patient similarity graph
    using PyTorch Geometric NeighborLoader for scalable mini-batch training.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = GRAPHSAGE_HIDDEN_DIM,
        num_layers: int = GRAPHSAGE_NUM_LAYERS,
        dropout: float = GRAPHSAGE_DROPOUT,
        lr: float = GRAPHSAGE_LR,
        epochs: int = GRAPHSAGE_EPOCHS,
    ):
        torch.manual_seed(RANDOM_SEED)
        self.model = GraphSAGEClassifier(in_channels, hidden_channels, num_layers, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.history = []
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[GraphSAGE] Using device: {self.device}")

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        edge_index_train: torch.Tensor,
        X_val: np.ndarray, y_val: np.ndarray,
        edge_index_val: torch.Tensor,
        save_path: Path = None,
    ) -> list:
        """
        Train the GraphSAGE model.

        Args:
            X_train, y_train: training node features and labels
            edge_index_train: COO edge index for training graph
            X_val, y_val: validation data
            edge_index_val: validation graph edge index
            save_path: where to save best model checkpoint
        """
        if save_path is None:
            save_path = ARTIFACTS_DIR / 'graphsage_centralized.pt'

        self.model.to(self.device)
        best_val_f1 = 0.0
        best_state = None

        x_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        edge_index_train_t = edge_index_train.to(self.device)

        print(f"\n[GraphSAGE] Training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            # ─── Training step ───
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(x_train_t, edge_index_train_t)
            loss = self.criterion(out, y_train_t)
            loss.backward()
            self.optimizer.step()
            avg_loss = loss.item()

            # ─── Validation step ───
            val_metrics = self._evaluate(X_val, y_val, edge_index_val)

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            record = {
                'epoch': epoch,
                'train_loss': round(avg_loss, 4),
                **{f'val_{k}': round(v, 4) for k, v in val_metrics.items()},
            }
            self.history.append(record)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{self.epochs} | Loss: {avg_loss:.4f} "
                    f"| Val Acc: {val_metrics['accuracy']:.4f} "
                    f"| Val F1: {val_metrics['f1']:.4f} "
                    f"| Val AUC: {val_metrics['roc_auc']:.4f}"
                )

        # Save best model
        if best_state:
            self.model.load_state_dict(best_state)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"[GraphSAGE] Best model saved to {save_path}")
        return self.history

    def _evaluate(self, X: np.ndarray, y: np.ndarray, edge_index: torch.Tensor) -> dict:
        """Full-batch evaluation on validation / test set."""
        self.model.eval()
        x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        ei_t = edge_index.to(self.device)
        with torch.no_grad():
            logits = self.model(x_t, ei_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            'accuracy': float(accuracy_score(y, preds)),
            'precision': float(precision_score(y, preds, zero_division=0)),
            'recall': float(recall_score(y, preds, zero_division=0)),
            'f1': float(f1_score(y, preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probs)),
        }

    def _evaluate_inductive(self, X_combined: np.ndarray, y_eval: np.ndarray, edge_index_eval: torch.Tensor, n_train: int) -> dict:
        """Inductive evaluation where eval nodes aggregate messages from training nodes."""
        self.model.eval()
        x_t = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        ei_t = edge_index_eval.to(self.device)
        with torch.no_grad():
            logits = self.model(x_t, ei_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        eval_probs = probs[n_train:]
        preds = (eval_probs >= 0.5).astype(int)
        return {
            'accuracy': float(accuracy_score(y_eval, preds)),
            'precision': float(precision_score(y_eval, preds, zero_division=0)),
            'recall': float(recall_score(y_eval, preds, zero_division=0)),
            'f1': float(f1_score(y_eval, preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_eval, eval_probs)),
        }

    def save_metrics(self, test_metrics: dict, history: list):
        """Persist test metrics and training history to results/."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Update metrics.json
        metrics_path = RESULTS_DIR / 'metrics.json'
        existing = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)
        existing['graphsage_centralized'] = test_metrics
        with open(metrics_path, 'w') as f:
            json.dump(existing, f, indent=2)

        # Save training history
        hist_path = RESULTS_DIR / 'training_history.json'
        hist_data = {}
        if hist_path.exists():
            with open(hist_path) as f:
                hist_data = json.load(f)
        hist_data['graphsage_centralized'] = history
        with open(hist_path, 'w') as f:
            json.dump(hist_data, f, indent=2)
        print(f"[GraphSAGE] Metrics saved.")
