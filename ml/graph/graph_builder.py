"""
ml/graph/graph_builder.py
Scalable patient similarity graph construction using k-NN.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from config.config import GRAPH_K_NEIGHBORS, ARTIFACTS_DIR


class PatientGraphBuilder:
    """
    Builds a patient similarity graph using k-nearest neighbours.

    Each patient becomes a node. Patients within each other's
    k-NN neighbourhood are connected by an edge.

    Uses sklearn NearestNeighbors (efficient, does NOT build
    a dense N×N matrix) for scalability on ~70k nodes.
    """

    def __init__(self, k: int = GRAPH_K_NEIGHBORS):
        self.k = k
        self.edge_index = None      # shape (2, num_edges)
        self.num_nodes = 0
        self.num_edges = 0
        self._positions = None      # 2-D layout for sample visualisation

    def build_graph(self, X: np.ndarray) -> torch.Tensor:
        """
        Construct kNN graph from feature matrix X.

        Args:
            X: numpy array of shape (n_patients, n_features), already scaled.

        Returns:
            edge_index: LongTensor of shape (2, num_edges).
        """
        n = X.shape[0]
        print(f"[GraphBuilder] Building {self.k}-NN graph on {n} nodes...")

        # k+1 because the node itself is its own 1st neighbour
        nbrs = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm='ball_tree',
            metric='euclidean',
            n_jobs=-1
        )
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Build COO edge list (exclude self-loops: skip column 0)
        rows, cols = [], []
        for i in range(n):
            for j in indices[i, 1:]:          # skip self (index 0)
                rows.append(i)
                cols.append(int(j))
                # Also add reverse edge for undirected graph
                rows.append(int(j))
                cols.append(i)

        # Remove duplicate edges
        edge_set = set(zip(rows, cols))
        rows, cols = zip(*edge_set) if edge_set else ([], [])

        edge_index = torch.tensor([list(rows), list(cols)], dtype=torch.long)

        self.edge_index = edge_index
        self.num_nodes = n
        self.num_edges = edge_index.shape[1] // 2  # undirected count
        print(f"[GraphBuilder] Graph built: {self.num_nodes} nodes, {self.num_edges} undirected edges")
        return edge_index

    def build_inductive_graph(self, X_train: np.ndarray, X_eval: np.ndarray) -> torch.Tensor:
        """
        Build an inductive evaluation graph connecting evaluation nodes to their k-NN training nodes.
        Node indexing: [0 .. n_train-1] are training nodes, [n_train .. n_train+n_eval-1] are eval nodes.
        """
        n_train = X_train.shape[0]
        n_eval = X_eval.shape[0]

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
        nbrs.fit(X_train)
        _, indices = nbrs.kneighbors(X_eval)

        rows, cols = [], []
        for eval_i in range(n_eval):
            eval_node_id = n_train + eval_i
            for train_j in indices[eval_i]:
                rows.append(eval_node_id)
                cols.append(int(train_j))
                rows.append(int(train_j))
                cols.append(eval_node_id)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index

    def save(self):
        """Save edge_index tensor to artifacts directory."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / 'edge_index.pt'
        torch.save(self.edge_index, path)
        print(f"[GraphBuilder] edge_index saved to {path}")

    @classmethod
    def load(cls, k: int = GRAPH_K_NEIGHBORS):
        """Load a previously built graph."""
        path = ARTIFACTS_DIR / 'edge_index.pt'
        if not path.exists():
            raise FileNotFoundError(f"No saved graph at {path}. Run train_graphsage.py first.")
        obj = cls(k=k)
        obj.edge_index = torch.load(path)
        obj.num_nodes = int(obj.edge_index.max()) + 1
        obj.num_edges = obj.edge_index.shape[1] // 2
        return obj

    def get_statistics(self) -> dict:
        """Return graph statistics dictionary."""
        if self.edge_index is None:
            return {}
        degrees = torch.zeros(self.num_nodes, dtype=torch.long)
        for src in self.edge_index[0]:
            degrees[src] += 1
        avg_degree = float(degrees.float().mean())
        max_possible_edges = self.num_nodes * (self.num_nodes - 1) / 2
        density = self.num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'avg_degree': round(avg_degree, 2),
            'density': float(f'{density:.6f}'),
            'k_neighbors': self.k,
            'graph_type': 'k-NN Similarity Graph (undirected)',
            'similarity_metric': 'Euclidean distance in standardised feature space',
        }

    def get_sample_graph(self, n_nodes: int = 150, labels: np.ndarray = None) -> dict:
        """
        Return a small subgraph for frontend visualisation.

        Returns:
            dict with 'nodes' (list of {id, label, x, y}) and
            'edges' (list of {source, target}).
        """
        if self.edge_index is None:
            return {'nodes': [], 'edges': []}

        # Take first n_nodes nodes
        n_sample = min(n_nodes, self.num_nodes)
        node_set = set(range(n_sample))

        # Filter edges within subgraph
        mask = (self.edge_index[0] < n_sample) & (self.edge_index[1] < n_sample)
        sub_ei = self.edge_index[:, mask]

        # Build simple circular / random 2D layout
        np.random.seed(42)
        angles = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
        r_jitter = np.random.uniform(0.6, 1.0, n_sample)
        xs = (r_jitter * np.cos(angles) * 400 + 500).tolist()
        ys = (r_jitter * np.sin(angles) * 300 + 350).tolist()

        nodes = []
        for i in range(n_sample):
            lbl = int(labels[i]) if labels is not None and i < len(labels) else 0
            nodes.append({'id': i, 'label': lbl, 'x': round(xs[i], 1), 'y': round(ys[i], 1)})

        # Deduplicate edges
        edge_set = set()
        edges = []
        for s, t in zip(sub_ei[0].tolist(), sub_ei[1].tolist()):
            key = (min(s, t), max(s, t))
            if key not in edge_set:
                edge_set.add(key)
                edges.append({'source': s, 'target': t})

        return {'nodes': nodes, 'edges': edges}
