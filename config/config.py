"""
FedGraph-Diabetes Configuration
Centralized configuration for all project components.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DATASET_PATH = DATA_DIR / "diabetes.csv"
PREPROCESSING_ARTIFACT = ARTIFACTS_DIR / "preprocessing.pkl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Diabetes_binary"
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # fraction of training set

EXPECTED_COLUMNS = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI",
    "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]

FEATURE_DESCRIPTIONS = {
    "HighBP": "High Blood Pressure (0=No, 1=Yes)",
    "HighChol": "High Cholesterol (0=No, 1=Yes)",
    "CholCheck": "Cholesterol Check in 5 Years (0=No, 1=Yes)",
    "BMI": "Body Mass Index",
    "Smoker": "Smoked ≥100 Cigarettes in Life (0=No, 1=Yes)",
    "Stroke": "History of Stroke (0=No, 1=Yes)",
    "HeartDiseaseorAttack": "Heart Disease or Attack (0=No, 1=Yes)",
    "PhysActivity": "Physical Activity in Past 30 Days (0=No, 1=Yes)",
    "Fruits": "Fruit Consumption ≥1/Day (0=No, 1=Yes)",
    "Veggies": "Vegetable Consumption ≥1/Day (0=No, 1=Yes)",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption (0=No, 1=Yes)",
    "AnyHealthcare": "Any Healthcare Coverage (0=No, 1=Yes)",
    "NoDocbcCost": "Unable to See Doctor Due to Cost (0=No, 1=Yes)",
    "GenHlth": "General Health (1=Excellent to 5=Poor)",
    "MentHlth": "Mental Health Days (0-30 days)",
    "PhysHlth": "Physical Health Days (0-30 days)",
    "DiffWalk": "Difficulty Walking (0=No, 1=Yes)",
    "Sex": "Sex (0=Female, 1=Male)",
    "Age": "Age Category (1-13, where 1=18-24, 13=80+)",
    "Education": "Education Level (1-6)",
    "Income": "Income Level (1-8)",
}

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]

# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_K_NEIGHBORS = 10          # default k for kNN graph
GRAPH_K_OPTIONS = [5, 10, 15]   # configurable options

# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────
GRAPHSAGE_HIDDEN_DIM = 64
GRAPHSAGE_NUM_LAYERS = 2
GRAPHSAGE_DROPOUT = 0.3
GRAPHSAGE_LR = 0.008
GRAPHSAGE_EPOCHS = 50
GRAPHSAGE_BATCH_SIZE = 512  # for neighbor sampling

# ─────────────────────────────────────────────────────────────────────────────
# Federated Learning
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLIENTS = 5
CLIENT_NAMES = ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"]
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 3
FEDPROX_MU = 0.01           # proximal term coefficient
PARTITIONING = "non_iid"    # "iid" or "non_iid"

# ─────────────────────────────────────────────────────────────────────────────
# Baseline Models
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "gradient_boosting",
]

# Base paper reported results (for academic comparison only)
BASE_PAPER_RESULTS = {
    "logistic_regression": {"accuracy": 0.726, "recall": 0.732},
    "decision_tree":       {"accuracy": 0.722, "recall": 0.723},
    "random_forest":       {"accuracy": 0.739, "recall": 0.741},
    "svm":                 {"accuracy": 0.741, "recall": 0.805},
    "knn":                 {"accuracy": 0.720, "recall": 0.718},
    "gradient_boosting":   {"accuracy": 0.749, "recall": 0.748},
}

# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_URL = "http://localhost:5173"
