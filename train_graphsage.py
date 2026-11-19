#!/usr/bin/env python3
"""
train_graphsage.py
Construct patient similarity graph and train centralized GraphSAGE model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml.preprocessing.preprocessor import DataPreprocessor
from ml.graph.graph_builder import PatientGraphBuilder
from ml.graph.graphsage_model import GraphSAGETrainer
from config.config import ARTIFACTS_DIR, RESULTS_DIR, EXPECTED_COLUMNS, TARGET_COLUMN

if __name__ == "__main__":
    print("=" * 60)
    print("FedGraph-Diabetes: Training Centralized GraphSAGE")
    print("=" * 60)

    # 1. Load or run preprocessing
    try:
        preprocessor = DataPreprocessor.load_artifacts()
    except Exception:
        print("[1/4] Running preprocessor...")
        preprocessor = DataPreprocessor().fit()
        preprocessor.save_artifacts()

    split_data = preprocessor.get_split_data()
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_val = split_data['X_val']
    y_val = split_data['y_val']
    X_test = split_data['X_test']
    y_test = split_data['y_test']

    # 2. Build patient similarity graph
    print("\n[2/4] Building k-NN patient similarity graph...")
    builder = PatientGraphBuilder(k=10)
    edge_index_train = builder.build_graph(X_train)
    builder.save()

    # Build validation and test graphs
    val_builder = PatientGraphBuilder(k=10)
    edge_index_val = val_builder.build_graph(X_val)

    test_builder = PatientGraphBuilder(k=10)
    edge_index_test = test_builder.build_graph(X_test)

    # 3. Train GraphSAGE
    print("\n[3/4] Training GraphSAGE model with NeighborLoader...")
    in_channels = len([c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN])
    epochs = int(os.environ.get("GRAPHSAGE_EPOCHS", 30))
    trainer = GraphSAGETrainer(in_channels=in_channels, epochs=epochs)
    
    history = trainer.train(
        X_train=X_train, y_train=y_train, edge_index_train=edge_index_train,
        X_val=X_val, y_val=y_val, edge_index_val=edge_index_val
    )

    # 4. Evaluate on test set
    print("\n[4/4] Evaluating on test set (Inductive neighborhood aggregation)...")
    import numpy as np
    X_test_combined = np.vstack([X_train, X_test])
    edge_index_test = builder.build_inductive_graph(X_train, X_test)
    test_metrics = trainer._evaluate_inductive(X_test_combined, y_test, edge_index_test, n_train=len(X_train))
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    trainer.save_metrics(test_metrics, history)
    print("\n✅ Centralized GraphSAGE training complete.")
    print("=" * 60)
