"""
ml/evaluation/evaluator.py
Evaluation and plotting utilities for comparing baselines, GraphSAGE, and Federated models.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

from config.config import PLOTS_DIR, RESULTS_DIR, ARTIFACTS_DIR, BASE_PAPER_RESULTS


class ModelEvaluator:
    """
    Computes comparative evaluations, generates publication-quality plots,
    and produces summary reports.
    """

    def __init__(self):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(self, roc_data_dict: dict, save_filename: str = 'roc_comparison.png'):
        """
        Plot overlay ROC curves for multiple models.
        roc_data_dict: { 'model_name': (y_true, y_proba), ... }
        """
        plt.figure(figsize=(9, 7))
        
        for name, (y_true, y_proba) in roc_data_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            display_name = name.replace('_', ' ').title()
            plt.plot(fpr, tpr, lw=2, label=f'{display_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = PLOTS_DIR / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Evaluator] Saved ROC plot to {save_path}")
        return str(save_path)

    def plot_model_comparison_bars(self, metrics_dict: dict, save_filename: str = 'model_comparison_bars.png'):
        """
        Bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC across models.
        metrics_dict: { 'model_name': {'accuracy': 0.8, 'precision': 0.75, ...}, ... }
        """
        models = list(metrics_dict.keys())
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        labels = [m.replace('_', ' ').title() for m in models]
        
        x = np.arange(len(models))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics_keys):
            values = [metrics_dict[m].get(metric, 0.0) for m in models]
            offset = (i - len(metrics_keys) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=metric.upper().replace('_', '-'))

        ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
        ax.set_title('Model Performance Comparison Across Key Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = PLOTS_DIR / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Evaluator] Saved Comparison Bar plot to {save_path}")
        return str(save_path)

    def plot_federated_rounds(self, history: list, method: str = 'fedavg', save_filename: str = None):
        """
        Plot training loss and global accuracy vs communication rounds.
        """
        if save_filename is None:
            save_filename = f'{method}_convergence.png'

        rounds = [item.get('round', item.get('epoch', idx + 1)) for idx, item in enumerate(history)]
        accuracies = [item.get('global_accuracy', item.get('val_accuracy', 0.0)) for item in history]
        f1s = [item.get('global_f1', item.get('val_f1', 0.0)) for item in history]
        losses = [item.get('avg_client_loss', item.get('train_loss', 0.0)) for item in history]

        fig, ax1 = plt.subplots(figsize=(9, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Communication Round / Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy / F1 Score', color=color, fontsize=12)
        line1 = ax1.plot(rounds, accuracies, color='tab:blue', marker='o', label='Global Accuracy', lw=2)
        line2 = ax1.plot(rounds, f1s, color='tab:green', marker='s', label='Global F1-Score', lw=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0.4, 1.0)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Loss', color=color, fontsize=12)
        line3 = ax2.plot(rounds, losses, color=color, marker='^', linestyle='--', label='Training Loss', lw=2)
        ax2.tick_params(axis='y', labelcolor=color)

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', fontsize=10)

        plt.title(f'Federated Learning Convergence ({method.upper()})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = PLOTS_DIR / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Evaluator] Saved FL convergence plot to {save_path}")
        return str(save_path)

    def generate_all_plots_and_summary(self):
        """
        Reads results/metrics.json and training_history.json to generate all figures.
        """
        metrics_file = RESULTS_DIR / 'metrics.json'
        history_file = RESULTS_DIR / 'training_history.json'

        if not metrics_file.exists():
            print("[Evaluator] No metrics.json found. Please train models first.")
            return

        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)

        flat_metrics = {}
        if 'baselines' in metrics_data:
            flat_metrics.update(metrics_data['baselines'])
        if 'graphsage_centralized' in metrics_data:
            flat_metrics['graphsage_centralized'] = metrics_data['graphsage_centralized']
        if 'federated_fedavg' in metrics_data:
            flat_metrics['fedavg_graphsage'] = metrics_data['federated_fedavg']
        if 'federated_fedprox' in metrics_data:
            flat_metrics['fedprox_graphsage'] = metrics_data['federated_fedprox']

        if flat_metrics:
            self.plot_model_comparison_bars(flat_metrics)

        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            for key, hist in history_data.items():
                if isinstance(hist, list) and len(hist) > 0:
                    self.plot_federated_rounds(hist, method=key)
