#!/usr/bin/env python3
"""
FedGraph-Diabetes Startup Helper Script
Checks environment and provides setup guidance.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required. Found:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_venv():
    venv_path = ROOT / ".venv"
    if venv_path.exists():
        print(f"✅ Virtual environment found at .venv/")
        return True
    print("⚠️  No virtual environment found. Create one with: python3 -m venv .venv")
    return False


def check_packages():
    packages = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "torch": "PyTorch",
        "torch_geometric": "PyTorch Geometric",
        "sklearn": "scikit-learn",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "shap": "SHAP",
        "matplotlib": "Matplotlib",
    }
    all_ok = True
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_ok = False
    return all_ok


def check_dataset():
    dataset = ROOT / "data" / "diabetes.csv"
    if dataset.exists():
        size_mb = dataset.stat().st_size / 1024 / 1024
        print(f"✅ Dataset found: data/diabetes.csv ({size_mb:.1f} MB)")
        return True
    print("⚠️  Dataset not found at data/diabetes.csv")
    print("   → DEMO MODE will be activated automatically")
    print("   → Download from: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
    return False


def check_artifacts():
    artifacts = ROOT / "artifacts"
    found = list(artifacts.glob("*.pkl")) + list(artifacts.glob("*.pt")) if artifacts.exists() else []
    if found:
        print(f"✅ {len(found)} artifacts found:")
        for f in found:
            print(f"   - {f.name}")
    else:
        print("ℹ️  No trained models found. Run training scripts first.")
    return True


def main():
    print("=" * 50)
    print("FedGraph-Diabetes Environment Check")
    print("=" * 50)
    print()

    print("── Python Environment ──")
    check_python_version()
    check_venv()
    print()

    print("── Python Packages ──")
    check_packages()
    print()

    print("── Dataset ──")
    check_dataset()
    print()

    print("── Trained Models ──")
    check_artifacts()
    print()

    print("=" * 50)
    print("Quick Start:")
    print("  1. source .venv/bin/activate")
    print("  2. python train_baselines.py")
    print("  3. python train_graphsage.py")
    print("  4. python train_federated.py --method fedavg")
    print("  5. cd backend && uvicorn main:app --reload")
    print("  6. cd frontend && npm run dev")
    print("=" * 50)


if __name__ == "__main__":
    main()
