"""
ml/federated/federated_trainer.py
Federated learning simulation: FedAvg and FedProx.
Hospitals are SIMULATED by partitioning a single dataset.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import copy
import json
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config.config import (
    NUM_CLIENTS, CLIENT_NAMES, FEDERATED_ROUNDS, LOCAL_EPOCHS,
    FEDPROX_MU, RANDOM_SEED, ARTIFACTS_DIR, RESULTS_DIR,
    GRAPHSAGE_HIDDEN_DIM, GRAPHSAGE_NUM_LAYERS, GRAPHSAGE_DROPOUT, GRAPHSAGE_LR
)
from ml.graph.graph_builder import PatientGraphBuilder
from ml.graph.graphsage_model import GraphSAGEClassifier


class DataPartitioner:
    """
    Splits training data into N client shards.
    Supports IID (random equal) and Non-IID (demographically biased) splits.

    NOTE: These are SIMULATED hospitals from a single dataset.
    """

    def __init__(self, feature_columns: list):
        self.feature_columns = feature_columns

    def iid_partition(
        self, X: np.ndarray, y: np.ndarray, n_clients: int = NUM_CLIENTS
    ) -> list:
        """Random equal split across clients (IID)."""
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(len(X))
        shards = np.array_split(indices, n_clients)
        return [(X[idx], y[idx]) for idx in shards]

    def non_iid_partition(
        self, X: np.ndarray, y: np.ndarray, n_clients: int = NUM_CLIENTS
    ) -> list:
        """
        Non-IID partition based on demographic attributes.
        Each simulated hospital has different patient distributions.

        Hospital A: Younger patients (Age category 1-4)
        Hospital B: Middle-aged, high BMI
        Hospital C: Senior patients (Age category 9-13)
        Hospital D: Low income patients (Income 1-3)
        Hospital E: Remaining / mixed population

        This creates realistic heterogeneous data distribution.
        """
        age_idx = self._col_idx('Age')
        bmi_idx = self._col_idx('BMI')
        income_idx = self._col_idx('Income')

        all_indices = np.arange(len(X))
        assigned = np.zeros(len(X), dtype=bool)

        hospital_indices = []

        # Hospital A: Young (Age 1-4)
        mask_a = (X[:, age_idx] <= 4) & (~assigned)
        idx_a = all_indices[mask_a]
        assigned[idx_a] = True

        # Hospital B: Middle-aged + high BMI (Age 5-8, BMI > 0.5 standardised)
        mask_b = (X[:, age_idx] >= 5) & (X[:, age_idx] <= 8) & (X[:, bmi_idx] > 0.5) & (~assigned)
        idx_b = all_indices[mask_b]
        assigned[idx_b] = True

        # Hospital C: Senior (Age 9-13)
        mask_c = (X[:, age_idx] >= 9) & (~assigned)
        idx_c = all_indices[mask_c]
        assigned[idx_c] = True

        # Hospital D: Low income (Income 1-3)
        mask_d = (X[:, income_idx] <= 3) & (~assigned) if income_idx is not None else np.zeros(len(X), dtype=bool)
        idx_d = all_indices[mask_d] if income_idx is not None else np.array([], dtype=int)
        if income_idx is not None:
            assigned[idx_d] = True

        # Hospital E: Remaining
        idx_e = all_indices[~assigned]

        shards = [idx_a, idx_b, idx_c, idx_d, idx_e]
        # Balance: if any shard is too small, redistribute
        min_size = max(200, len(X) // (n_clients * 4))
        extra_pool = []
        for i, shard in enumerate(shards):
            if len(shard) < min_size:
                extra_pool.extend(shard.tolist())
                shards[i] = np.array([], dtype=int)

        extra_pool = np.array(extra_pool)
        np.random.shuffle(extra_pool)
        extras = np.array_split(extra_pool, n_clients)
        balanced = [
            np.concatenate([s, e]) if len(s) > 0 else e
            for s, e in zip(shards[:n_clients], extras)
        ]

        print(f"[Partitioner] Non-IID split:")
        for i, (name, idx) in enumerate(zip(CLIENT_NAMES[:n_clients], balanced)):
            print(f"  {name}: {len(idx)} patients")

        return [(X[idx.astype(int)], y[idx.astype(int)]) for idx in balanced]

    def _col_idx(self, col_name: str):
        try:
            return self.feature_columns.index(col_name)
        except ValueError:
            return None


class HospitalClient:
    """
    Simulates one federated hospital client.
    Holds local data, builds local graph, performs local training.
    """

    def __init__(self, client_id: int, name: str, X: np.ndarray, y: np.ndarray, k: int = 10):
        self.client_id = client_id
        self.name = name
        self.X = X
        self.y = y
        self.k = k
        self.n_samples = len(X)
        self._build_local_graph()

    def _build_local_graph(self):
        """Build local patient similarity graph on client data."""
        builder = PatientGraphBuilder(k=min(self.k, self.n_samples - 1))
        self.edge_index = builder.build_graph(self.X)

    def train_local(
        self,
        global_model: GraphSAGEClassifier,
        local_epochs: int = LOCAL_EPOCHS,
        mu: float = 0.0,          # mu=0.0 → FedAvg, mu>0 → FedProx
        lr: float = GRAPHSAGE_LR,
        device: torch.device = None,
    ) -> dict:
        """
        Local training on client data.

        For FedProx, adds proximal term: mu/2 * ||w - w_global||^2

        Returns:
            dict with 'state_dict', 'n_samples', 'loss'
        """
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # Deep-copy global model for local update
        local_model = copy.deepcopy(global_model).to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        x_t = torch.tensor(self.X, dtype=torch.float32).to(device)
        y_t = torch.tensor(self.y, dtype=torch.float32).to(device)
        ei_t = self.edge_index.to(device)

        # Capture global parameters for proximal term
        global_params = [p.data.clone() for p in local_model.parameters()]

        local_model.train()
        total_loss = 0.0
        for _ in range(local_epochs):
            optimizer.zero_grad()
            out = local_model(x_t, ei_t)
            loss = criterion(out, y_t)

            # FedProx proximal regularisation
            if mu > 0.0:
                prox = sum(
                    torch.norm(p - g_p) ** 2
                    for p, g_p in zip(local_model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return {
            'state_dict': {k: v.cpu().clone() for k, v in local_model.state_dict().items()},
            'n_samples': self.n_samples,
            'loss': total_loss / local_epochs,
        }


class FederatedServer:
    """
    Federated Learning server.
    Manages global model, orchestrates rounds, performs aggregation.
    """

    def __init__(self, in_channels: int, n_clients: int = NUM_CLIENTS):
        torch.manual_seed(RANDOM_SEED)
        self.global_model = GraphSAGEClassifier(in_channels)
        self.n_clients = n_clients
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def fedavg_aggregate(self, client_updates: list) -> None:
        """
        FedAvg: weighted average of client model parameters.
        Weight = n_samples_client / n_samples_total
        """
        total_samples = sum(u['n_samples'] for u in client_updates)
        new_state = {}
        for key in client_updates[0]['state_dict'].keys():
            weighted = sum(
                u['state_dict'][key] * (u['n_samples'] / total_samples)
                for u in client_updates
            )
            new_state[key] = weighted
        self.global_model.load_state_dict(new_state)

    def evaluate_global(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        edge_index_test: torch.Tensor,
    ) -> dict:
        """Evaluate global model on held-out test set."""
        self.global_model.eval().to(self.device)
        x_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        ei_t = edge_index_test.to(self.device)
        with torch.no_grad():
            logits = self.global_model(x_t, ei_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probs)),
        }

    def run_federated(
        self,
        clients: list,
        X_test: np.ndarray,
        y_test: np.ndarray,
        edge_index_test: torch.Tensor,
        method: str = 'fedavg',
        rounds: int = FEDERATED_ROUNDS,
        local_epochs: int = LOCAL_EPOCHS,
        mu: float = FEDPROX_MU,
    ) -> list:
        """
        Main federated training loop.

        Args:
            clients: list of HospitalClient objects
            method: 'fedavg' or 'fedprox'
            rounds: number of communication rounds
            local_epochs: local training epochs per round
            mu: proximal coefficient (FedProx only)

        Returns:
            history: list of dicts with per-round metrics
        """
        mu_effective = mu if method == 'fedprox' else 0.0
        history = []

        print(f"\n{'='*60}")
        print(f"Federated Training: {method.upper()}")
        print(f"Clients: {len(clients)} | Rounds: {rounds} | Local epochs: {local_epochs}")
        if method == 'fedprox':
            print(f"Proximal mu: {mu}")
        print('='*60)

        for rnd in range(1, rounds + 1):
            # ─── 1. Distribute global model to clients ───
            client_updates = []
            round_client_losses = []

            for client in clients:
                update = client.train_local(
                    self.global_model,
                    local_epochs=local_epochs,
                    mu=mu_effective,
                    device=self.device,
                )
                client_updates.append(update)
                round_client_losses.append({
                    'client': client.name,
                    'loss': round(update['loss'], 4),
                    'n_samples': update['n_samples'],
                })

            # ─── 2. Aggregate ───
            self.fedavg_aggregate(client_updates)   # same aggregation for both methods

            # ─── 3. Global evaluation ───
            global_metrics = self.evaluate_global(X_test, y_test, edge_index_test)

            record = {
                'round': rnd,
                'method': method,
                'global_accuracy': round(global_metrics['accuracy'], 4),
                'global_f1': round(global_metrics['f1'], 4),
                'global_roc_auc': round(global_metrics['roc_auc'], 4),
                'client_losses': round_client_losses,
                'avg_client_loss': round(
                    sum(u['loss'] for u in client_updates) / len(client_updates), 4
                ),
            }
            history.append(record)

            print(
                f"  Round {rnd:3d}/{rounds} | "
                f"Acc: {global_metrics['accuracy']:.4f} | "
                f"F1: {global_metrics['f1']:.4f} | "
                f"AUC: {global_metrics['roc_auc']:.4f}"
            )

        return history

    def save_model(self, method: str):
        """Save the global model."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / f'{method}_graphsage.pt'
        torch.save(self.global_model.state_dict(), path)
        print(f"[FederatedServer] Global model saved to {path}")

    def save_history(self, history: list, method: str, test_metrics: dict):
        """Persist training history and final metrics."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Training history
        hist_path = RESULTS_DIR / 'training_history.json'
        hist_data = {}
        if hist_path.exists():
            with open(hist_path) as f:
                hist_data = json.load(f)
        hist_data[method] = history
        with open(hist_path, 'w') as f:
            json.dump(hist_data, f, indent=2)

        # Metrics
        metrics_path = RESULTS_DIR / 'metrics.json'
        existing = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)
        existing[f'federated_{method}'] = {
            **test_metrics,
            'method': method,
            'rounds': len(history),
            'n_clients': self.n_clients,
        }
        with open(metrics_path, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"[FederatedServer] History saved.")
