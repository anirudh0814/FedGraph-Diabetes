"""
backend/schemas/schemas.py
Pydantic response and request models for FedGraph-Diabetes API.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    demo_mode: bool
    dataset_loaded: bool
    models_trained: bool
    available_models: List[str]
    device: str


class FeatureStat(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    null_count: int
    type: str
    description: str


class DatasetSummary(BaseModel):
    n_records: int
    n_features: int
    target_column: str
    class_distribution: Dict[str, int]
    demo_mode: bool
    dataset_path: str
    missing_values: int
    feature_stats: Dict[str, FeatureStat]


class FeatureInfo(BaseModel):
    name: str
    description: str
    type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = None


class ModelMetricItem(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: Optional[List[List[int]]] = None
    training_time_sec: Optional[float] = None
    base_paper: Optional[Dict[str, Any]] = None


class ModelComparisonResponse(BaseModel):
    status: str
    models: Dict[str, Any]
    base_paper_comparison: Dict[str, Any]
    note: str


class ClientLoss(BaseModel):
    client: str
    loss: float
    n_samples: int


class FederatedRoundHistory(BaseModel):
    round: int
    method: str
    global_accuracy: float
    global_f1: float
    global_roc_auc: float
    avg_client_loss: float
    client_losses: Optional[List[ClientLoss]] = None


class FederatedStatusResponse(BaseModel):
    status: str
    available_methods: List[str]
    num_clients: int
    client_names: List[str]
    history: Dict[str, List[Dict[str, Any]]]
    latest_metrics: Dict[str, Any]


class GraphNode(BaseModel):
    id: int
    label: int
    x: float
    y: float


class GraphEdge(BaseModel):
    source: int
    target: int


class GraphSampleResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_graph_nodes: int
    total_graph_edges: int
    k_neighbors: int


class GraphStatsResponse(BaseModel):
    num_nodes: int
    num_edges: int
    avg_degree: float
    density: float
    k_neighbors: int
    graph_type: str
    similarity_metric: str


class GlobalExplanationResponse(BaseModel):
    status: str
    model: str
    method: str
    feature_importance: Dict[str, float]
    note: str


class PatientInput(BaseModel):
    HighBP: float = Field(0.0, description="High Blood Pressure (0=No, 1=Yes)")
    HighChol: float = Field(0.0, description="High Cholesterol (0=No, 1=Yes)")
    CholCheck: float = Field(1.0, description="Cholesterol check in 5 years (0=No, 1=Yes)")
    BMI: float = Field(28.0, description="Body Mass Index (e.g. 12-98)")
    Smoker: float = Field(0.0, description="Smoked at least 100 cigarettes (0=No, 1=Yes)")
    Stroke: float = Field(0.0, description="Ever had a stroke (0=No, 1=Yes)")
    HeartDiseaseorAttack: float = Field(0.0, description="Coronary heart disease/MI (0=No, 1=Yes)")
    PhysActivity: float = Field(1.0, description="Physical activity in past 30 days (0=No, 1=Yes)")
    Fruits: float = Field(1.0, description="Consume Fruit 1+ times per day (0=No, 1=Yes)")
    Veggies: float = Field(1.0, description="Consume Vegetables 1+ times per day (0=No, 1=Yes)")
    HvyAlcoholConsump: float = Field(0.0, description="Heavy alcohol consumption (0=No, 1=Yes)")
    AnyHealthcare: float = Field(1.0, description="Have any healthcare coverage (0=No, 1=Yes)")
    NoDocbcCost: float = Field(0.0, description="Could not see doctor due to cost (0=No, 1=Yes)")
    GenHlth: float = Field(2.0, description="General health 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor")
    MentHlth: float = Field(0.0, description="Days of poor mental health (0-30)")
    PhysHlth: float = Field(0.0, description="Days of physical illness/injury (0-30)")
    DiffWalk: float = Field(0.0, description="Serious difficulty walking/climbing stairs (0=No, 1=Yes)")
    Sex: float = Field(0.0, description="0=Female, 1=Male")
    Age: float = Field(7.0, description="Age category 1=18-24, 7=50-54, 13=80+")
    Education: float = Field(5.0, description="Education level (1-6)")
    Income: float = Field(6.0, description="Income level (1-8)")
    model_choice: Optional[str] = Field("fedavg_graphsage", description="Model to use for inference")


class PredictionResponse(BaseModel):
    probability: float
    risk_category: str
    predicted_class: int
    model_used: str
    explanation_method: str
    feature_contributions: Dict[str, float]
    top_risk_factors: List[str]
    top_protective_factors: List[str]
    disclaimer: str
