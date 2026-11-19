#!/usr/bin/env python3
"""
evaluate.py
Evaluate all trained models, generate ROC comparisons, confusion matrices, and convergence plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import torch
import numpy as np
import pandas as pd
from ml.preprocessing.preprocessor import DataPreprocessor
from ml.graph.graph_builder import PatientGraphBuilder
from ml.graph.graphsage_model import GraphSAGEClassifier
from ml.evaluation.evaluator import ModelEvaluator
from config.config import ARTIFACTS_DIR, RESULTS_DIR, EXPECTED_COLUMNS, TARGET_COLUMN, BASE_PAPER_RESULTS

def main():
    print("=" * 60)
    print("FedGraph-Diabetes: Comprehensive Evaluation & Reporting")
    print("=" * 60)

    evaluator = ModelEvaluator()

    # Load preprocessed data
    try:
        preprocessor = DataPreprocessor.load_artifacts()
    except Exception:
        print("No preprocessing artifacts found. Please run train_baselines.py first.")
        return

    split_data = preprocessor.get_split_data()
    X_test = split_data['X_test']
    y_test = split_data['y_test']

    feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
    in_channels = len(feature_cols)

    # Collect predictions for ROC curve
    roc_dict = {}

    # Baselines
    for pkl_path in ARTIFACTS_DIR.glob('*.pkl'):
        if pkl_path.name in ['preprocessor.pkl', 'preprocessing.pkl']:
            continue
        model_name = pkl_path.stem
        try:
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_dict[model_name] = (y_test, y_prob)
        except Exception as e:
            print(f"Could not evaluate baseline {model_name}: {e}")

    # Build test graph for GNN evaluations
    test_builder = PatientGraphBuilder(k=10)
    edge_index_test = test_builder.build_graph(X_test)
    x_t = torch.tensor(X_test, dtype=torch.float32)

    # GNN models
    gnn_models = ['graphsage_centralized', 'fedavg_graphsage', 'fedprox_graphsage']
    for gnn_name in gnn_models:
        pt_path = ARTIFACTS_DIR / f"{gnn_name}.pt"
        if pt_path.exists():
            try:
                model = GraphSAGEClassifier(in_channels=in_channels)
                state = torch.load(pt_path, map_location=torch.device('cpu'))
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    logits = model(x_t, edge_index_test)
                    probs = torch.sigmoid(logits).cpu().numpy()
                roc_dict[gnn_name] = (y_test, probs)
            except Exception as e:
                print(f"Could not evaluate GNN {gnn_name}: {e}")

    # Generate plots
    if roc_dict:
        evaluator.plot_roc_curves(roc_dict)
    
    evaluator.generate_all_plots_and_summary()
    print("\n✅ Evaluation complete. All plots saved to results/plots/")
    print("=" * 60)

if __name__ == "__main__":
    main()
