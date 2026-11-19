# FedGraph-Diabetes

> **Privacy-Aware Explainable Federated Graph Learning for Diabetes Risk Prediction**

A final-year academic research project extending a 2025 base paper on explainable diabetes prediction ML by incorporating patient similarity graphs, GraphSAGE, and federated learning.

---

## Table of Contents

1. [Project Title](#1-project-title)
2. [Problem Statement](#2-problem-statement)
3. [Base Paper](#3-base-paper)
4. [Research Gap](#4-research-gap)
5. [Proposed Solution](#5-proposed-solution)
6. [Dataset](#6-dataset)
7. [Methodology](#7-methodology)
8. [System Architecture](#8-system-architecture)
9. [Installation](#9-installation)
10. [Dataset Setup](#10-dataset-setup)
11. [Training](#11-training)
12. [Running the Application](#12-running-the-application)
13. [API Documentation](#13-api-documentation)
14. [Experimental Results](#14-experimental-results)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)
17. [Medical Disclaimer](#17-medical-disclaimer)

---

## 1. Project Title

**FedGraph-Diabetes**: Privacy-Aware Explainable Federated Graph Learning for Diabetes Risk Prediction

---

## 2. Problem Statement

Diabetes is a significant global health burden. Early and accurate prediction enables timely intervention. Machine-learning-based prediction tools have shown promise, but existing approaches typically:

- Treat patients as independent tabular records
- Use centralized data aggregation (privacy risk)
- Ignore patient-to-patient relationships

This project addresses these limitations.

---

## 3. Base Paper

The 2025 base paper performs **centralized, explainable diabetes prediction** using:
- CDC BRFSS 2015-derived dataset (70,692 records, 21 features)
- Traditional ML: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, MARS
- SHAP for explainability
- Reported results: Gradient Boosting ~74.9% accuracy, SVM ~80.5% recall

---

## 4. Research Gap

The base paper:
1. Uses **centralized learning** — all data must be at one location
2. Treats patients as **independent tabular records** — ignores relationships
3. Does not model **patient-to-patient similarity**
4. Does not use **federated/decentralized healthcare learning**

---

## 5. Proposed Solution

FedGraph-Diabetes extends the base paper with:

| Component | Description |
|-----------|-------------|
| Patient Similarity Graph | Each patient = node, similar patients = edges |
| GraphSAGE | Inductive GNN that learns from graph neighborhoods |
| Federated Learning | Multiple hospital clients, no raw data sharing |
| FedAvg | Federated Averaging aggregation |
| FedProx | Proximal-term regularized federated optimization |
| Explainability | SHAP for baselines, gradient saliency for GraphSAGE |

---

## 6. Dataset

**Source:** CDC BRFSS 2015 (via Kaggle — Diabetes Health Indicators Dataset)

**Download:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

**File:** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

**Placement:** Rename and place as `data/diabetes.csv`

| Property | Value |
|----------|-------|
| Records | 70,692 |
| Features | 21 |
| Target | `Diabetes_binary` (0/1) |
| Balance | 50/50 |

---

## 7. Methodology

### Graph Construction
1. Preprocess patient features (scale numeric, keep binary/ordinal as-is)
2. Use `sklearn.NearestNeighbors` with `k=10` (configurable)
3. Create sparse kNN graph → edge_index for PyTorch Geometric
4. Patient similarity measured by Euclidean distance in feature space

### GraphSAGE
- 2 SAGEConv layers with ReLU + Dropout
- Hidden dimension: 64
- Binary cross-entropy loss, Adam optimizer
- NeighborLoader for scalable training (mini-batch)

### Federated Learning
- **5 simulated hospitals** (partitioned from the single dataset)
- **Non-IID partitioning** by Age, BMI, and Income distribution
- **FedAvg**: weighted averaging of model parameters by client sample count
- **FedProx**: adds proximal term μ‖w - w_global‖² to local loss

### Explainability
- **Baseline models**: SHAP TreeExplainer (tree models) / LinearExplainer (logistic regression)
- **GraphSAGE**: Gradient-based saliency (input gradients w.r.t. output probability)
- ⚠️ Gradient saliency ≠ SHAP — labeled correctly throughout

---

## 8. System Architecture

```
BRFSS DATASET
      ↓
Data Cleaning & Preprocessing
      ↓
Patient Similarity Graph (kNN, k=10)
      ↓
┌──────────────────────────────────────────┐
│ Hospital A │ Hospital B │ Hospital C │...│  ← Simulated FL Clients
└──────────────────────────────────────────┘
      │
   Local GraphSAGE Training
      │
      ↓
FEDERATED SERVER
  ┌──────┴──────┐
FedAvg      FedProx
  └──────┬──────┘
      ↓
GLOBAL GraphSAGE Model
      ↓
New Patient → Preprocessing → Inference → Prediction
      ↓
Explainability (Gradient Saliency)
      ↓
Web Application (FastAPI + React)
```

---

## 9. Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ (included in `.node/` if downloaded via setup script)
- ~4 GB disk space (for PyTorch)

### Setup

```bash
# Clone/navigate to project
cd fedgraph-diabetes

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version — faster setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install frontend dependencies (requires Node.js)
cd frontend
npm install
cd ..
```

---

## 10. Dataset Setup

```bash
# Download from Kaggle (requires Kaggle account)
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

# Place in data directory
cp ~/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2015.csv data/diabetes.csv
```

If no dataset is provided, the system automatically runs in **DEMO MODE** with synthetic data clearly labelled.

---

## 11. Training

Activate virtual environment first:
```bash
source .venv/bin/activate
```

### Step 1: Train Baseline Models
```bash
python train_baselines.py
```
Outputs: `artifacts/*.pkl`, `results/metrics.json`, `results/model_comparison.csv`

### Step 2: Train Centralized GraphSAGE
```bash
python train_graphsage.py
```
Outputs: `artifacts/graphsage_centralized.pt`, updates `results/metrics.json`

### Step 3: Train Federated Models
```bash
# FedAvg
python train_federated.py --method fedavg --rounds 20 --clients 5

# FedProx
python train_federated.py --method fedprox --rounds 20 --clients 5 --mu 0.01
```
Outputs: `artifacts/fedavg_graphsage.pt`, `artifacts/fedprox_graphsage.pt`, `results/training_history.json`

### Step 4: Evaluate All Models
```bash
python evaluate.py
```
Outputs: plots in `results/plots/`, updated `results/model_comparison.csv`

---

## 12. Running the Application

### Backend (FastAPI)
```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API available at: http://localhost:8000
API docs: http://localhost:8000/docs

### Frontend (React)
```bash
cd frontend
npm run dev
```
App available at: http://localhost:5173

---

## 13. API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health status |
| GET | `/api/dataset/summary` | Dataset overview |
| GET | `/api/dataset/features` | Feature descriptions |
| GET | `/api/dataset/statistics` | Feature statistics |
| GET | `/api/models/comparison` | All model metrics |
| GET | `/api/federated/status` | FL training status |
| GET | `/api/federated/history` | Per-round FL history |
| GET | `/api/graph/statistics` | Graph stats |
| GET | `/api/graph/sample` | Sample graph nodes/edges |
| GET | `/api/explainability/global` | Global feature importance |
| POST | `/api/predict` | Patient risk prediction |
| POST | `/api/train` | Trigger background training |

Full interactive docs: http://localhost:8000/docs

---

## 14. Experimental Results

Results are computed from actual training runs and saved to `results/metrics.json`.

⚠️ **Note:** Our experimental results may differ from the base paper's reported values because:
1. Results depend on the train/test split and random seeds
2. The base paper may use different evaluation protocols
3. Our implementation focuses on educational reproducibility, not optimized hypertuning

The dashboard clearly distinguishes "Base Paper Reported" vs "Our Experimental Results."

---

## 15. Limitations

- **Simulated hospitals**: The 5 hospitals are artificially partitioned from one dataset, not real institutions
- **Single dataset**: Real FL would use data from truly separate institutions
- **Privacy**: This is a privacy-preserving simulation; it does not provide cryptographic privacy guarantees
- **Scale**: Graph training on 70k nodes requires significant memory; we use neighbor sampling
- **Graph sensitivity**: Results depend on choice of k (number of neighbors)
- **Explainability**: GraphSAGE explanations use gradient saliency, not true SHAP values
- **No clinical validation**: This is a research prototype only

---

## 16. Future Work

- [ ] Real multi-hospital data integration with actual institutional data
- [ ] Differential privacy integration (DP-SGD)
- [ ] Secure aggregation protocols
- [ ] GNNExplainer or PGExplainer for better GNN explainability
- [ ] Hyperparameter optimization
- [ ] Temporal modeling (longitudinal patient data)
- [ ] Node feature enrichment with clinical biomarkers

---

## 17. Medical Disclaimer

> ⚠️ **This system is a research and educational prototype and is NOT a medical diagnostic tool. Predictions should not be used for diagnosis or treatment decisions. Please consult a qualified healthcare professional for any medical concerns.**

The models in this project are trained on survey data and have not been clinically validated. They do not represent FDA-approved or CE-marked medical devices.

---

## Project Structure

```
fedgraph-diabetes/
├── backend/          # FastAPI backend
├── ml/               # ML pipeline
│   ├── preprocessing/
│   ├── baseline/
│   ├── graph/
│   ├── federated/
│   ├── explainability/
│   └── evaluation/
├── frontend/         # React frontend
├── data/             # Dataset (place diabetes.csv here)
├── artifacts/        # Saved models and preprocessing artifacts
├── results/          # Metrics, plots, comparison CSV
├── experiments/      # Experiment notebooks
├── config/           # Configuration
├── train_baselines.py
├── train_graphsage.py
├── train_federated.py
├── evaluate.py
└── requirements.txt
```
