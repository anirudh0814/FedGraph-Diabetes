#!/usr/bin/env python3
"""
Quick demo script: generates synthetic DEMO data, trains baseline models
and GraphSAGE briefly, and starts the backend so the UI is immediately usable.
Useful for a quick viva/demo without the full dataset.

Usage: python demo.py
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("FedGraph-Diabetes — Quick Demo Setup")
    print("=" * 60)
    print("This will train models on DEMO (synthetic) data.")
    print("For real results, place data/diabetes.csv first.\n")

    # 1. Train baselines (fast)
    run([PYTHON, str(ROOT / "train_baselines.py")], cwd=ROOT)

    # 2. Train GraphSAGE with fewer epochs for quick demo
    env = os.environ.copy()
    env["GRAPHSAGE_EPOCHS"] = "10"
    run([PYTHON, str(ROOT / "train_graphsage.py")], cwd=ROOT, env=env)

    # 3. Train FedAvg with fewer rounds
    env["FEDERATED_ROUNDS"] = "5"
    run([PYTHON, str(ROOT / "train_federated.py"), "--method", "fedavg", "--rounds", "5"], cwd=ROOT, env=env)

    print("\n" + "=" * 60)
    print("Demo training complete!")
    print("\nTo start the application:")
    print("  Terminal 1: cd backend && ../.venv/bin/python -m uvicorn main:app --reload")
    print("  Terminal 2: cd frontend && npm run dev")
    print("=" * 60)
