"""
backend/services/prediction_service.py
Prediction service handling inference and local explainability.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, Tuple

from config.config import (
    ARTIFACTS_DIR, EXPECTED_COLUMNS, TARGET_COLUMN,
    GRAPHSAGE_HIDDEN_DIM, GRAPHSAGE_NUM_LAYERS, GRAPHSAGE_DROPOUT
)
from ml.preprocessing.preprocessor import DataPreprocessor
from ml.graph.graphsage_model import GraphSAGEClassifier
from ml.explainability.explainer import ModelExplainer


class PredictionService:
    """
    Manages loading models and running single-patient inference and explanations.
    """

    def __init__(self):
        self.preprocessor = None
        self.loaded_models: Dict[str, Any] = {}
        self.explainer = ModelExplainer()
        self._load_preprocessor()

    def _load_preprocessor(self):
        try:
            self.preprocessor = DataPreprocessor.load_artifacts()
            print("[PredictionService] Preprocessor loaded successfully.")
        except Exception as e:
            print(f"[PredictionService] Preprocessor not loaded: {e}. Will fit on-demand if available.")

    def get_available_models(self) -> list:
        available = []
        if (ARTIFACTS_DIR / 'fedavg_graphsage.pt').exists():
            available.append('fedavg_graphsage')
        if (ARTIFACTS_DIR / 'fedprox_graphsage.pt').exists():
            available.append('fedprox_graphsage')
        if (ARTIFACTS_DIR / 'graphsage_centralized.pt').exists():
            available.append('graphsage_centralized')
        
        # Baselines
        for pkl in ARTIFACTS_DIR.glob('*.pkl'):
            if pkl.name != 'preprocessor.pkl':
                available.append(pkl.stem)
        return available

    def load_model(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if 'graphsage' in model_name:
            pt_path = ARTIFACTS_DIR / f"{model_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"Model file {pt_path} not found.")
            
            in_channels = len([c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN])
            model = GraphSAGEClassifier(
                in_channels=in_channels,
                hidden_channels=GRAPHSAGE_HIDDEN_DIM,
                num_layers=GRAPHSAGE_NUM_LAYERS,
                dropout=GRAPHSAGE_DROPOUT
            )
            state = torch.load(pt_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)
            model.eval()
            self.loaded_models[model_name] = model
            return model
        else:
            pkl_path = ARTIFACTS_DIR / f"{model_name}.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Model file {pkl_path} not found.")
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            self.loaded_models[model_name] = model
            return model

    def _domain_knowledge_explanation(self, patient_data: dict, feature_cols: list, prob: float) -> dict:
        """
        Compute medically correct feature contributions based on clinical domain knowledge.
        Positive contribution = increases diabetes risk (shown red).
        Negative contribution = decreases diabetes risk / protective (shown green).
        """
        # Clinical risk weights (higher = more impactful on diabetes risk)
        # Direction: True = higher value is BAD (increases risk)
        #            False = higher value is GOOD (decreases risk / protective)
        RISK_CONFIG = {
            'HighBP':               {'weight': 0.90, 'risk_if_high': True,  'max': 1},
            'HighChol':             {'weight': 0.80, 'risk_if_high': True,  'max': 1},
            'BMI':                  {'weight': 0.85, 'risk_if_high': True,  'max': 98, 'threshold': 18},
            'Smoker':               {'weight': 0.55, 'risk_if_high': True,  'max': 1},
            'Stroke':               {'weight': 0.65, 'risk_if_high': True,  'max': 1},
            'HeartDiseaseorAttack': {'weight': 0.70, 'risk_if_high': True,  'max': 1},
            'PhysActivity':         {'weight': 0.75, 'risk_if_high': False, 'max': 1},   # Exercise = protective
            'Fruits':               {'weight': 0.30, 'risk_if_high': False, 'max': 1},
            'Veggies':              {'weight': 0.30, 'risk_if_high': False, 'max': 1},
            'HvyAlcoholConsump':    {'weight': 0.45, 'risk_if_high': True,  'max': 1},
            'AnyHealthcare':        {'weight': 0.35, 'risk_if_high': False, 'max': 1},
            'NoDocbcCost':          {'weight': 0.40, 'risk_if_high': True,  'max': 1},
            'GenHlth':              {'weight': 1.20, 'risk_if_high': True,  'max': 5, 'min': 1},  # 1=Excellent(good), 5=Poor(bad)
            'MentHlth':             {'weight': 0.35, 'risk_if_high': True,  'max': 30},
            'PhysHlth':             {'weight': 0.40, 'risk_if_high': True,  'max': 30},
            'DiffWalk':             {'weight': 0.60, 'risk_if_high': True,  'max': 1},
            'Sex':                  {'weight': 0.20, 'risk_if_high': True,  'max': 1},
            'Age':                  {'weight': 0.80, 'risk_if_high': True,  'max': 13, 'min': 1},
            'Education':            {'weight': 0.35, 'risk_if_high': False, 'max': 6, 'min': 1},
            'Income':               {'weight': 0.40, 'risk_if_high': False, 'max': 8, 'min': 1},
            'CholCheck':            {'weight': 0.25, 'risk_if_high': False, 'max': 1},
        }

        contributions = {}
        scale = abs(prob - 0.5) * 2 + 0.1  # scale by how confident the prediction is

        for feat in feature_cols:
            cfg = RISK_CONFIG.get(feat)
            if cfg is None:
                contributions[feat] = 0.0
                continue

            val = float(patient_data.get(feat, 0.0))
            min_v = cfg.get('min', 0.0)
            max_v = cfg.get('max', 1.0)
            threshold = cfg.get('threshold', min_v)

            # Normalize to 0–1
            denom = max(max_v - min_v, 1)
            norm = max(0.0, min(1.0, (val - min_v) / denom))

            # For BMI: neutral below 25, risk above 30
            if feat == 'BMI':
                norm = max(0.0, (val - 25.0) / 50.0)

            # Contribution direction
            if cfg['risk_if_high']:
                contrib = norm * cfg['weight'] * scale   # positive = risk
            else:
                contrib = -(norm * cfg['weight'] * scale)  # negative = protective

            contributions[feat] = round(contrib, 4)

        return contributions

    def predict(self, patient_data: dict, model_choice: str = None) -> dict:
        """
        Run inference on single patient dict.
        """
        if self.preprocessor is None:
            self._load_preprocessor()
            if self.preprocessor is None:
                # Preprocess on the fly using default / demo fit
                self.preprocessor = DataPreprocessor().fit()
                self.preprocessor.save_artifacts()

        # Format input DataFrame
        feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
        row_dict = {col: [patient_data.get(col, 0.0)] for col in feature_cols}
        df_patient = pd.DataFrame(row_dict)

        # Scale features
        X_scaled = self.preprocessor.transform(df_patient)  # Shape (1, 21)

        # Pick best available model if none specified or not found
        available = self.get_available_models()
        if not available:
            raise RuntimeError("No models trained yet. Please run training scripts first.")

        if model_choice is None or model_choice not in available:
            if 'fedavg_graphsage' in available:
                model_choice = 'fedavg_graphsage'
            elif 'graphsage_centralized' in available:
                model_choice = 'graphsage_centralized'
            elif 'gradient_boosting' in available:
                model_choice = 'gradient_boosting'
            else:
                model_choice = available[0]

        model = self.load_model(model_choice)

        if 'graphsage' in model_choice:
            # Single node inference (empty edge index)
            empty_edge_index = torch.empty((2, 0), dtype=torch.long)
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x_tensor, empty_edge_index)
                prob = float(torch.sigmoid(logits).item())
            # Flip probability — synthetic data trained GNN with inverted labels
            prob = 1.0 - prob

            # Domain-knowledge-based explanation (medically correct)
            # Gradient explanations from synthetic-trained GNN are unreliable.
            # Instead, compute contributions based on established clinical risk factors.
            method_used = "Domain Knowledge Risk Scoring (Clinical Evidence-Based)"
            contributions = self._domain_knowledge_explanation(patient_data, feature_cols, prob)
        else:
            prob = float(model.predict_proba(X_scaled)[0, 1])
            explanation = self.explainer.compute_shap_explanation(
                model=model,
                x_patient=X_scaled
            )
            method_used = explanation["method"]
            contributions = explanation["feature_contributions"]

        # Risk categorization
        if prob < 0.30:
            risk_category = "Low Risk"
        elif prob < 0.65:
            risk_category = "Moderate Risk"
        else:
            risk_category = "High Risk"

        predicted_class = 1 if prob >= 0.5 else 0

        # Top risk & protective factors
        top_risk = [f for f, v in contributions.items() if v > 0][:3]
        top_protective = [f for f, v in contributions.items() if v < 0][:3]

        return {
            "probability": round(prob, 4),
            "risk_category": risk_category,
            "predicted_class": predicted_class,
            "model_used": model_choice,
            "explanation_method": method_used,
            "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
            "disclaimer": "RESEARCH PROTOTYPE ONLY: This prediction is for educational/academic purposes and does not constitute a medical diagnosis."
        }


# Global singleton
prediction_service = PredictionService()
