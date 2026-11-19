#!/usr/bin/env python3
"""
train_baselines.py
Train and evaluate traditional machine learning baseline models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ml.preprocessing.preprocessor import DataPreprocessor
from ml.baseline.train_baselines import BaselineTrainer
from config.config import RESULTS_DIR, ARTIFACTS_DIR

if __name__ == "__main__":
    print("=" * 60)
    print("FedGraph-Diabetes: Training Baseline Models")
    print("=" * 60)

    # 1. Preprocess data
    print("\n[1/3] Preprocessing dataset...")
    preprocessor = DataPreprocessor()
    preprocessor.fit()
    preprocessor.save_artifacts()

    # 2. Get data splits
    split_data = preprocessor.get_split_data()

    # 3. Train all baselines
    print("\n[2/3] Fitting baselines (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting)...")
    trainer = BaselineTrainer(split_data)
    trainer.run_all()

    print("\n[3/3] Training finished successfully!")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"Metrics saved to: {RESULTS_DIR / 'metrics.json'}")
    print(f"Comparison table: {RESULTS_DIR / 'model_comparison.csv'}")
    print("=" * 60)
