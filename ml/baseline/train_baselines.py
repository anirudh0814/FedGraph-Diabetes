"""
ml/baseline/train_baselines.py
Trains traditional ML baseline models and computes SHAP explanations.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score

from config.config import (
    ARTIFACTS_DIR, RESULTS_DIR, RANDOM_SEED,
    EXPECTED_COLUMNS, TARGET_COLUMN, BASE_PAPER_RESULTS
)

import shap


class BaselineTrainer:
    """
    Trains, evaluates and explains traditional ML baseline models.
    Results are computed from actual data — NOT hardcoded.
    """

    MODEL_DEFINITIONS = {
        'logistic_regression': LogisticRegression(
            max_iter=500, C=0.5, random_state=RANDOM_SEED, n_jobs=-1
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_SEED
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=60, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1
        ),
        'svm': SVC(
            probability=True, kernel='rbf', C=0.5,
            random_state=RANDOM_SEED, max_iter=1500
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=9, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED
        ),
    }

    def __init__(self, split_data: dict):
        """
        Args:
            split_data: dict with X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.X_train = split_data['X_train']
        self.X_val   = split_data['X_val']
        self.X_test  = split_data['X_test']
        self.y_train = split_data['y_train']
        self.y_val   = split_data['y_val']
        self.y_test  = split_data['y_test']
        self.feature_names = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
        self.results = {}
        self.trained_models = {}

    def run_all(self):
        """Train all baseline models and save results."""
        print("\n" + "="*60)
        print("Training Baseline Models")
        print("="*60)

        all_metrics = {}
        for name, model in self.MODEL_DEFINITIONS.items():
            print(f"\n[{name}] Training...")
            t0 = time.time()
            # Use combined train+val for final fit
            X_tr = np.vstack([self.X_train, self.X_val])
            y_tr = np.concatenate([self.y_train, self.y_val])
            model.fit(X_tr, y_tr)
            elapsed = time.time() - t0

            # Evaluate on test set
            metrics = self._evaluate(model, name, elapsed)
            all_metrics[name] = metrics
            self.trained_models[name] = model

            # Save model
            self._save_model(model, name)
            print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")

        self.results = all_metrics
        self._save_metrics(all_metrics)
        self._save_comparison_csv(all_metrics)
        self._compute_shap_global()
        print("\n✅ All baseline models trained and saved.")

    def _evaluate(self, model, name: str, elapsed: float) -> dict:
        """Compute full metrics on test set."""
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        cm = confusion_matrix(self.y_test, y_pred).tolist()
        return {
            'model': name,
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(self.y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(self.y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(self.y_test, y_proba)),
            'confusion_matrix': cm,
            'training_time_sec': round(elapsed, 2),
            'base_paper': BASE_PAPER_RESULTS.get(name, {}),
        }

    def _save_model(self, model, name: str):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def _save_metrics(self, all_metrics: dict):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = RESULTS_DIR / 'metrics.json'
        # Load existing metrics if any
        existing = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)
        existing['baselines'] = all_metrics
        with open(metrics_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")

    def _save_comparison_csv(self, all_metrics: dict):
        rows = []
        for name, m in all_metrics.items():
            rows.append({
                'model': name,
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'roc_auc': m['roc_auc'],
                'training_time_sec': m['training_time_sec'],
                'base_paper_accuracy': m['base_paper'].get('accuracy', None),
                'base_paper_recall': m['base_paper'].get('recall', None),
            })
        df = pd.DataFrame(rows)
        path = RESULTS_DIR / 'model_comparison.csv'
        df.to_csv(path, index=False)
        print(f"  Comparison CSV saved to {path}")

    def _compute_shap_global(self):
        """Compute global SHAP feature importances for best baseline."""
        print("\n[SHAP] Computing global feature importances...")
        # Use gradient boosting (tree-based, efficient with SHAP)
        model_name = 'gradient_boosting'
        if model_name not in self.trained_models:
            print("  Gradient boosting not trained; skipping SHAP.")
            return

        model = self.trained_models[model_name]
        # Use a background sample for efficiency
        background = self.X_test[:200]
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background)
            # shap_values shape: (n_samples, n_features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance = {
                self.feature_names[i]: float(mean_abs_shap[i])
                for i in range(len(self.feature_names))
            }
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            result = {
                'model': model_name,
                'method': 'SHAP TreeExplainer',
                'feature_importance': importance,
                'note': 'Global feature importances computed using SHAP TreeExplainer on gradient_boosting model.'
            }
            path = RESULTS_DIR / 'shap_baseline.json'
            with open(path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  SHAP importances saved to {path}")
        except Exception as e:
            print(f"  SHAP computation failed: {e}")

    def compute_shap_for_model(self, model_name: str):
        """Compute SHAP for a specific model (for API use)."""
        if model_name not in self.trained_models:
            return None
        model = self.trained_models[model_name]
        try:
            if model_name == 'logistic_regression':
                explainer = shap.LinearExplainer(model, self.X_train[:100])
            else:
                explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test[:100])
            mean_abs = np.abs(shap_values).mean(axis=0)
            return {self.feature_names[i]: float(mean_abs[i]) for i in range(len(self.feature_names))}
        except Exception:
            return None
