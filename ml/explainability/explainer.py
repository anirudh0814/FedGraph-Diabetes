"""
ml/explainability/explainer.py
Explainability methods for FedGraph-Diabetes:
- Baseline models: SHAP TreeExplainer & LinearExplainer
- GraphSAGE: Gradient-based saliency (clearly labeled as Gradient-based Saliency, NEVER SHAP)
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.config import EXPECTED_COLUMNS, TARGET_COLUMN, RESULTS_DIR


class ModelExplainer:
    """
    Provides explainability methods for both traditional tabular models and GraphSAGE.
    """

    def __init__(self, feature_names: list = None):
        if feature_names is None:
            self.feature_names = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
        else:
            self.feature_names = feature_names

    def compute_gradient_explanation(
        self,
        model: torch.nn.Module,
        x_patient: np.ndarray,
        edge_index: torch.Tensor = None,
    ) -> dict:
        """
        Compute gradient-based input saliency for GraphSAGE on a single patient sample.
        
        NOTE: This is Gradient-Based Input Saliency (Input * Gradient).
        It is explicitly NOT SHAP values.
        """
        model.eval()
        
        # Format input tensor
        if isinstance(x_patient, np.ndarray):
            if x_patient.ndim == 1:
                x_tensor = torch.tensor(x_patient, dtype=torch.float32).unsqueeze(0)
            else:
                x_tensor = torch.tensor(x_patient, dtype=torch.float32)
        else:
            x_tensor = x_patient.clone().detach().float()
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)

        x_tensor.requires_grad_(True)
        
        if edge_index is None:
            # Single node inference without edges
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Forward pass
        logits = model(x_tensor, edge_index)
        prob = torch.sigmoid(logits)
        
        # Target the prediction score for backward
        if logits.numel() == 1:
            target = logits
        else:
            target = logits[0]

        model.zero_grad()
        target.backward()

        # Input * Gradient attribution
        grads = x_tensor.grad.detach().cpu().numpy()
        inputs = x_tensor.detach().cpu().numpy()
        attribution = (grads * inputs)[0]

        # Map to feature names
        contributions = {}
        for i, col in enumerate(self.feature_names):
            if i < len(attribution):
                contributions[col] = float(attribution[i])

        # Sort by absolute impact
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        )

        return {
            "method": "Gradient-Based Input Saliency (Input * Gradient)",
            "feature_contributions": sorted_contributions,
            "probability": float(prob.detach().cpu().numpy().item() if prob.numel() == 1 else prob[0].item()),
            "note": "Attribution computed via backpropagating prediction gradients w.r.t input features for GraphSAGE."
        }

    def compute_shap_explanation(self, model, x_patient: np.ndarray, background_data: np.ndarray = None) -> dict:
        """
        Compute local SHAP explanation for baseline tabular models (Random Forest, Gradient Boosting, etc.)
        """
        import shap
        if x_patient.ndim == 1:
            x_sample = x_patient.reshape(1, -1)
        else:
            x_sample = x_patient

        try:
            # Check model type
            model_class_name = model.__class__.__name__
            if "Logistic" in model_class_name:
                if background_data is not None:
                    explainer = shap.LinearExplainer(model, background_data)
                else:
                    explainer = shap.LinearExplainer(model, x_sample)
                shap_values = explainer.shap_values(x_sample)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_sample)

            if isinstance(shap_values, list):
                # Binary classification, shap_values is list [class 0, class 1]
                vals = shap_values[1][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                vals = shap_values[0, :, 1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                vals = shap_values[0]
            else:
                vals = np.array(shap_values).flatten()

            contributions = {
                self.feature_names[i]: float(vals[i])
                for i in range(min(len(self.feature_names), len(vals)))
            }
            sorted_contributions = dict(
                sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
            )

            return {
                "method": f"SHAP ({explainer.__class__.__name__})",
                "feature_contributions": sorted_contributions,
                "note": f"SHAP value attributions for {model_class_name} model."
            }
        except Exception as e:
            # Fallback feature importance based on coefficients or tree importances if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                contributions = {
                    self.feature_names[i]: float(importances[i])
                    for i in range(min(len(self.feature_names), len(importances)))
                }
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0]
                contributions = {
                    self.feature_names[i]: float(coef[i] * x_sample[0, i])
                    for i in range(min(len(self.feature_names), len(coef)))
                }
            else:
                contributions = {f: 0.0 for f in self.feature_names}
            
            return {
                "method": "Fallback Feature Importance",
                "feature_contributions": dict(
                    sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
                ),
                "note": f"Fallback explanation: {str(e)}"
            }
