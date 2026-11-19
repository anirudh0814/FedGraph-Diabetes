"""
backend/api/routes.py
FastAPI API route endpoints for FedGraph-Diabetes.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from config.config import (
    DATASET_PATH, RESULTS_DIR, ARTIFACTS_DIR, EXPECTED_COLUMNS,
    TARGET_COLUMN, FEATURE_DESCRIPTIONS, BINARY_FEATURES,
    NUMERIC_FEATURES, ORDINAL_FEATURES, BASE_PAPER_RESULTS,
    NUM_CLIENTS, CLIENT_NAMES
)
from backend.schemas.schemas import (
    HealthResponse, DatasetSummary, FeatureInfo,
    ModelComparisonResponse, FederatedStatusResponse,
    GraphSampleResponse, GraphStatsResponse,
    GlobalExplanationResponse, PatientInput, PredictionResponse
)

# Lazy getter — heavy imports (torch, sklearn, etc.) happen only on first request
_prediction_service = None

def get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        from backend.services.prediction_service import PredictionService
        _prediction_service = PredictionService()
    return _prediction_service

router = APIRouter(prefix="/api", tags=["FedGraph-Diabetes"])



@router.get("/health", response_model=HealthResponse)
def get_health():
    import torch
    dataset_exists = DATASET_PATH.exists()
    svc = get_prediction_service()
    available_models = svc.get_available_models()
    models_trained = len(available_models) > 0
    device = "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU"

    demo_mode = False
    if not dataset_exists:
        demo_mode = True
    elif (ARTIFACTS_DIR / 'preprocessing.pkl').exists():
        try:
            from ml.preprocessing.preprocessor import DataPreprocessor
            p = DataPreprocessor.load_artifacts()
            demo_mode = p.demo_mode
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        demo_mode=demo_mode,
        dataset_loaded=dataset_exists,
        models_trained=models_trained,
        available_models=available_models,
        device=device
    )


@router.get("/dataset/summary", response_model=DatasetSummary)
def get_dataset_summary():
    summary_file = RESULTS_DIR / 'dataset_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            return DatasetSummary(**data)
    try:
        from ml.preprocessing.preprocessor import DataPreprocessor
        p = DataPreprocessor().fit()
        p.save_artifacts()
        return DatasetSummary(**p.get_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset summary: {str(e)}")


@router.get("/dataset/features", response_model=List[FeatureInfo])
def get_dataset_features():
    features = []
    feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
    for col in feature_cols:
        desc = FEATURE_DESCRIPTIONS.get(col, col)
        if col in BINARY_FEATURES:
            ftype = "binary"
            min_v, max_v, def_v = 0.0, 1.0, 0.0
            opts = [{"label": "No (0)", "value": 0.0}, {"label": "Yes (1)", "value": 1.0}]
        elif col in NUMERIC_FEATURES:
            ftype = "numeric"
            opts = None
            if col == 'BMI':
                min_v, max_v, def_v = 12.0, 98.0, 28.0
            elif col in ['MentHlth', 'PhysHlth']:
                min_v, max_v, def_v = 0.0, 30.0, 0.0
            else:
                min_v, max_v, def_v = 0.0, 100.0, 0.0
        else:
            ftype = "ordinal"
            opts = None
            if col == 'GenHlth':
                min_v, max_v, def_v = 1.0, 5.0, 2.0
                opts = [
                    {"label": "1 - Excellent", "value": 1.0},
                    {"label": "2 - Very Good", "value": 2.0},
                    {"label": "3 - Good", "value": 3.0},
                    {"label": "4 - Fair", "value": 4.0},
                    {"label": "5 - Poor", "value": 5.0}
                ]
            elif col == 'Age':
                min_v, max_v, def_v = 1.0, 13.0, 7.0
                opts = [
                    {"label": f"{i} - (Age group {i})", "value": float(i)}
                    for i in range(1, 14)
                ]
            elif col == 'Education':
                min_v, max_v, def_v = 1.0, 6.0, 5.0
            elif col == 'Income':
                min_v, max_v, def_v = 1.0, 8.0, 6.0
            else:
                min_v, max_v, def_v = 1.0, 10.0, 1.0
        features.append(FeatureInfo(
            name=col, description=desc, type=ftype,
            min_value=min_v, max_value=max_v, default_value=def_v, options=opts
        ))
    return features


@router.get("/dataset/statistics")
def get_dataset_statistics():
    summary_file = RESULTS_DIR / 'dataset_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            return data.get("feature_stats", {})
    return {}


@router.get("/models/comparison", response_model=ModelComparisonResponse)
def get_model_comparison():
    metrics_file = RESULTS_DIR / 'metrics.json'
    if not metrics_file.exists():
        return ModelComparisonResponse(
            status="not_trained", models={},
            base_paper_comparison=BASE_PAPER_RESULTS,
            note="Models have not been trained yet. Please execute training scripts."
        )
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    flat_models = {}
    if 'baselines' in metrics_data:
        flat_models.update(metrics_data['baselines'])
    if 'graphsage_centralized' in metrics_data:
        flat_models['graphsage_centralized'] = metrics_data['graphsage_centralized']
    if 'federated_fedavg' in metrics_data:
        flat_models['federated_fedavg'] = metrics_data['federated_fedavg']
    if 'federated_fedprox' in metrics_data:
        flat_models['federated_fedprox'] = metrics_data['federated_fedprox']
    return ModelComparisonResponse(
        status="ready", models=flat_models,
        base_paper_comparison=BASE_PAPER_RESULTS,
        note="Comparison of all evaluated models against base paper reported metrics."
    )


@router.get("/federated/status", response_model=FederatedStatusResponse)
def get_federated_status():
    history_file = RESULTS_DIR / 'training_history.json'
    metrics_file = RESULTS_DIR / 'metrics.json'
    history_data = {}
    if history_file.exists():
        with open(history_file, 'r') as f:
            history_data = json.load(f)
    latest_metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            m_data = json.load(f)
            if 'federated_fedavg' in m_data:
                latest_metrics['fedavg'] = m_data['federated_fedavg']
            if 'federated_fedprox' in m_data:
                latest_metrics['fedprox'] = m_data['federated_fedprox']
    available_methods = [k for k in ['fedavg', 'fedprox'] if k in history_data or f'federated_{k}' in latest_metrics]
    return FederatedStatusResponse(
        status="ready" if available_methods else "not_trained",
        available_methods=available_methods,
        num_clients=NUM_CLIENTS, client_names=CLIENT_NAMES,
        history=history_data, latest_metrics=latest_metrics
    )


@router.get("/federated/history")
def get_federated_history():
    history_file = RESULTS_DIR / 'training_history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return {}


@router.get("/graph/statistics", response_model=GraphStatsResponse)
def get_graph_statistics():
    from ml.preprocessing.preprocessor import DataPreprocessor
    from ml.graph.graph_builder import PatientGraphBuilder
    pt_path = ARTIFACTS_DIR / 'edge_index.pt'
    if not pt_path.exists():
        try:
            p = DataPreprocessor.load_artifacts()
        except Exception:
            p = DataPreprocessor().fit()
        split = p.get_split_data()
        builder = PatientGraphBuilder(k=10)
        builder.build_graph(split['X_train'][:500])
        return GraphStatsResponse(**builder.get_statistics())
    builder = PatientGraphBuilder.load(k=10)
    return GraphStatsResponse(**builder.get_statistics())


@router.get("/graph/sample", response_model=GraphSampleResponse)
def get_graph_sample(n_nodes: int = 120):
    from ml.preprocessing.preprocessor import DataPreprocessor
    from ml.graph.graph_builder import PatientGraphBuilder
    try:
        p = DataPreprocessor.load_artifacts()
    except Exception:
        p = DataPreprocessor().fit()
    split = p.get_split_data()
    X_train = split['X_train']
    y_train = split['y_train']
    builder = PatientGraphBuilder(k=8)
    sub_X = X_train[:n_nodes]
    sub_y = y_train[:n_nodes]
    builder.build_graph(sub_X)
    sample = builder.get_sample_graph(n_nodes=n_nodes, labels=sub_y)
    return GraphSampleResponse(
        nodes=sample['nodes'], edges=sample['edges'],
        total_graph_nodes=len(X_train),
        total_graph_edges=builder.num_edges, k_neighbors=8
    )


@router.get("/explainability/global", response_model=GlobalExplanationResponse)
def get_global_explanation():
    shap_file = RESULTS_DIR / 'shap_baseline.json'
    if shap_file.exists():
        with open(shap_file, 'r') as f:
            data = json.load(f)
            return GlobalExplanationResponse(
                status="ready",
                model=data.get('model', 'gradient_boosting'),
                method=data.get('method', 'SHAP TreeExplainer'),
                feature_importance=data.get('feature_importance', {}),
                note=data.get('note', '')
            )

    return GlobalExplanationResponse(
        status="not_available",
        model="gradient_boosting",
        method="SHAP",
        feature_importance={},
        note="Global explainability data not generated yet. Train models first."
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_patient(patient_input: PatientInput):
    data_dict = patient_input.dict()
    model_choice = data_dict.pop('model_choice', 'fedavg_graphsage')
    try:
        result = get_prediction_service().predict(data_dict, model_choice=model_choice)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
