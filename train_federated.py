#!/usr/bin/env python3
"""
train_federated.py
Simulate Federated Graph Learning (FedAvg and FedProx) across simulated hospital clients.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ml.preprocessing.preprocessor import DataPreprocessor
from ml.graph.graph_builder import PatientGraphBuilder
from ml.federated.federated_trainer import DataPartitioner, HospitalClient, FederatedServer
from config.config import (
    CLIENT_NAMES, NUM_CLIENTS, FEDERATED_ROUNDS, LOCAL_EPOCHS,
    FEDPROX_MU, EXPECTED_COLUMNS, TARGET_COLUMN
)

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation (FedAvg / FedProx)")
    parser.add_argument("--method", type=str, default="fedavg", choices=["fedavg", "fedprox"],
                        help="Federated aggregation algorithm")
    parser.add_argument("--rounds", type=int, default=FEDERATED_ROUNDS,
                        help="Number of federated communication rounds")
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS,
                        help="Number of simulated hospital clients")
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS,
                        help="Number of local training epochs per round")
    parser.add_argument("--mu", type=float, default=FEDPROX_MU,
                        help="FedProx proximal regularization coefficient")
    parser.add_argument("--non-iid", action="store_true", default=True,
                        help="Use Non-IID demographic partitioning")
    args = parser.parse_args()

    print("=" * 60)
    print(f"FedGraph-Diabetes: Federated Learning Simulation ({args.method.upper()})")
    print("=" * 60)

    # 1. Load Preprocessing
    try:
        preprocessor = DataPreprocessor.load_artifacts()
    except Exception:
        print("[1/4] Fitting preprocessor...")
        preprocessor = DataPreprocessor().fit()
        preprocessor.save_artifacts()

    split_data = preprocessor.get_split_data()
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']

    feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
    in_channels = len(feature_cols)

    # 2. Partition Data across simulated hospitals
    print(f"\n[2/4] Partitioning data across {args.clients} simulated hospitals...")
    partitioner = DataPartitioner(feature_columns=feature_cols)
    if args.non_iid:
        client_shards = partitioner.non_iid_partition(X_train, y_train, n_clients=args.clients)
    else:
        client_shards = partitioner.iid_partition(X_train, y_train, n_clients=args.clients)

    # 3. Create simulated hospital clients
    print(f"\n[3/4] Initializing hospital clients and building local graphs...")
    clients = []
    for i, (X_c, y_c) in enumerate(client_shards):
        c_name = CLIENT_NAMES[i] if i < len(CLIENT_NAMES) else f"Hospital_{chr(65+i)}"
        client = HospitalClient(client_id=i, name=c_name, X=X_c, y=y_c, k=8)
        clients.append(client)

    # Build test graph
    test_builder = PatientGraphBuilder(k=8)
    edge_index_test = test_builder.build_graph(X_test)

    # 4. Federated Training
    print(f"\n[4/4] Starting Federated Server ({args.method.upper()})...")
    server = FederatedServer(in_channels=in_channels, n_clients=args.clients)
    
    history = server.run_federated(
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        edge_index_test=edge_index_test,
        method=args.method,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        mu=args.mu
    )

    # Final evaluation & save
    final_metrics = server.evaluate_global(X_test, y_test, edge_index_test)
    server.save_model(method=args.method)
    server.save_history(history, method=args.method, test_metrics=final_metrics)

    print("\n" + "=" * 60)
    print(f"Federated {args.method.upper()} Training Complete!")
    print(f"Final Global Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Global Test F1-Score: {final_metrics['f1']:.4f}")
    print(f"Final Global Test ROC-AUC:  {final_metrics['roc_auc']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
