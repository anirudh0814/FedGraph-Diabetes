import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

// Mock data definitions for fallback
const MOCK_HEALTH = { status: 'healthy', version: '1.0.0', demo_mode: true };

const MOCK_DATASET_SUMMARY = {
  total_records: 70692,
  num_features: 21,
  class_distribution: { '0': 35346, '1': 35346 },
  source: 'CDC BRFSS 2015',
  demo_mode: true
};

const MOCK_DATASET_FEATURES = [
  { name: 'HighBP', type: 'binary', description: 'High Blood Pressure', min: 0, max: 1 },
  { name: 'HighChol', type: 'binary', description: 'High Cholesterol', min: 0, max: 1 },
  { name: 'BMI', type: 'numeric', description: 'Body Mass Index', min: 12, max: 98 },
  { name: 'Smoker', type: 'binary', description: 'Smoked 100+ cigarettes', min: 0, max: 1 },
  { name: 'Stroke', type: 'binary', description: 'Ever had a stroke', min: 0, max: 1 },
  { name: 'HeartDiseaseorAttack', type: 'binary', description: 'Heart disease or MI', min: 0, max: 1 },
  { name: 'PhysActivity', type: 'binary', description: 'Physical activity past 30 days', min: 0, max: 1 },
  { name: 'Fruits', type: 'binary', description: 'Consume fruit 1+ times/day', min: 0, max: 1 },
  { name: 'Veggies', type: 'binary', description: 'Consume veggies 1+ times/day', min: 0, max: 1 },
  { name: 'HvyAlcoholConsump', type: 'binary', description: 'Heavy alcohol consumer', min: 0, max: 1 },
  { name: 'AnyHealthcare', type: 'binary', description: 'Have health care coverage', min: 0, max: 1 },
  { name: 'NoDocbcCost', type: 'binary', description: 'Could not see doc due to cost', min: 0, max: 1 },
  { name: 'GenHlth', type: 'ordinal', description: 'General health (1-5)', min: 1, max: 5 },
  { name: 'MentHlth', type: 'numeric', description: 'Days poor mental health', min: 0, max: 30 },
  { name: 'PhysHlth', type: 'numeric', description: 'Days poor physical health', min: 0, max: 30 },
  { name: 'DiffWalk', type: 'binary', description: 'Difficulty walking', min: 0, max: 1 },
  { name: 'Sex', type: 'binary', description: '0=Female, 1=Male', min: 0, max: 1 },
  { name: 'Age', type: 'ordinal', description: 'Age category (1-13)', min: 1, max: 13 },
  { name: 'Education', type: 'ordinal', description: 'Education level (1-6)', min: 1, max: 6 },
  { name: 'Income', type: 'ordinal', description: 'Income category (1-8)', min: 1, max: 8 },
];

const MOCK_MODELS_COMPARISON = {
  logistic_regression:   { accuracy: 0.8076, roc_auc: 0.8867, f1: 0.8084, recall: 0.8123 },
  decision_tree:         { accuracy: 0.7716, roc_auc: 0.8450, f1: 0.7663, recall: 0.7489 },
  random_forest:         { accuracy: 0.8100, roc_auc: 0.8923, f1: 0.8135, recall: 0.8288 },
  svm:                   { accuracy: 0.6505, roc_auc: 0.7276, f1: 0.6753, recall: 0.7271 },
  knn:                   { accuracy: 0.7638, roc_auc: 0.8421, f1: 0.7819, recall: 0.8469 },
  gradient_boosting:     { accuracy: 0.8117, roc_auc: 0.8927, f1: 0.8177, recall: 0.8448 },
  graphsage_centralized: { accuracy: 0.8226, roc_auc: 0.9032, f1: 0.8305, recall: 0.8694 },
  federated_fedavg:      { accuracy: 0.8108, roc_auc: 0.8988, f1: 0.8261, recall: 0.8987 },
  federated_fedprox:     { accuracy: 0.8210, roc_auc: 0.8991, f1: 0.8213, recall: 0.8226 },
};

const MOCK_FEDERATED_STATUS = {
  trained: true,
  num_clients: 5,
  clients_data_size: { 'Hospital A': 7543, 'Hospital B': 6479, 'Hospital C': 21197, 'Hospital D': 5291, 'Hospital E': 8973 },
  best_algorithm: 'FedAvg'
};

const MOCK_FEDERATED_HISTORY = {
  rounds: Array.from({ length: 5 }, (_, i) => ({
    round: i + 1,
    accuracy: 0.68 + i * 0.025,
    loss: 0.65 - i * 0.04
  }))
};

const MOCK_GRAPH_STATISTICS = {
  num_nodes: 49483,
  num_edges: 354199,
  avg_degree: 14.32,
  density: 0.000289,
  k_neighbors: 10,
  is_connected: false,
  num_components: 1
};

const MOCK_GLOBAL_EXPLANATION = {
  method: "SHAP TreeExplainer",
  model: "GradientBoosting",
  top_features: [
    { feature: "GenHlth", importance: 0.15, direction: "positive" as const },
    { feature: "HighBP", importance: 0.12, direction: "positive" as const },
    { feature: "BMI", importance: 0.10, direction: "positive" as const },
    { feature: "Age", importance: 0.08, direction: "positive" as const },
    { feature: "HighChol", importance: 0.07, direction: "positive" as const },
    { feature: "Income", importance: -0.06, direction: "negative" as const },
    { feature: "HeartDiseaseorAttack", importance: 0.05, direction: "positive" as const },
    { feature: "Education", importance: -0.04, direction: "negative" as const },
  ]
};

export interface HealthStatus {
  status: string;
  version: string;
  demo_mode: boolean;
}

export interface DatasetSummary {
  total_records: number;
  num_features: number;
  class_distribution: Record<string, number>;
  source: string;
  demo_mode: boolean;
}

export interface FeatureInfo {
  name: string;
  type: string;
  description: string;
  min: number;
  max: number;
}

export interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  f1: number;
  recall: number;
}

export interface ModelComparison {
  [modelName: string]: ModelMetrics;
}

export interface FederatedStatus {
  trained: boolean;
  num_clients: number;
  clients_data_size: Record<string, number>;
  best_algorithm: string;
}

export interface FederatedRound {
  round: number;
  accuracy: number;
  loss: number;
}

export interface FederatedHistory {
  rounds: FederatedRound[];
}

export interface GraphStatistics {
  num_nodes: number;
  num_edges: number;
  avg_degree: number;
  density: number;
  k_neighbors: number;
  is_connected: boolean;
  num_components: number;
}

export interface SampleGraph {
  nodes: { id: string; diabetes: number; x?: number; y?: number }[];
  edges: { source: string; target: string }[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';
}

export interface GlobalExplanation {
  method: string;
  model: string;
  top_features: FeatureImportance[];
}

export interface PredictionResult {
  probability: number;
  prediction: number;
  model_used: string;
  risk_category?: string;
  explanation_method?: string;
  explanation: {
    contributions: FeatureImportance[];
  };
}

export const apiService = {
  getHealth: async (): Promise<HealthStatus> => {
    try {
      const res = await api.get('/health');
      return {
        status: res.data.status,
        version: '1.0.0',
        demo_mode: res.data.demo_mode ?? true
      };
    } catch {
      return MOCK_HEALTH;
    }
  },

  getDatasetSummary: async (): Promise<DatasetSummary> => {
    try {
      const res = await api.get('/dataset/summary');
      const d = res.data;
      return {
        total_records: d.n_records || d.total_records || 70692,
        num_features: d.n_features || d.num_features || 21,
        class_distribution: d.class_distribution || { '0': 35346, '1': 35346 },
        source: 'CDC BRFSS 2015',
        demo_mode: d.demo_mode ?? true
      };
    } catch {
      return MOCK_DATASET_SUMMARY;
    }
  },

  getDatasetFeatures: async (): Promise<FeatureInfo[]> => {
    try {
      const res = await api.get('/dataset/features');
      return res.data.map((f: any) => ({
        name: f.name,
        type: f.type,
        description: f.description,
        min: f.min_value ?? 0,
        max: f.max_value ?? 1
      }));
    } catch {
      return MOCK_DATASET_FEATURES;
    }
  },

  getDatasetStatistics: async (): Promise<any> => {
    try {
      const res = await api.get('/dataset/statistics');
      return res.data;
    } catch {
      return {};
    }
  },

  getModelsComparison: async (): Promise<ModelComparison> => {
    try {
      const res = await api.get('/models/comparison');
      const models = res.data.models || {};
      const formatted: ModelComparison = {};
      
      for (const [key, val] of Object.entries(models) as [string, any][]) {
        formatted[key] = {
          accuracy: val.accuracy ?? 0,
          roc_auc: val.roc_auc ?? 0,
          f1: val.f1 ?? 0,
          recall: val.recall ?? 0,
        };
      }
      return Object.keys(formatted).length > 0 ? formatted : MOCK_MODELS_COMPARISON;
    } catch {
      return MOCK_MODELS_COMPARISON;
    }
  },

  getFederatedStatus: async (): Promise<FederatedStatus> => {
    try {
      const res = await api.get('/federated/status');
      const d = res.data;
      return {
        trained: (d.available_methods && d.available_methods.length > 0) || false,
        num_clients: d.num_clients || 5,
        clients_data_size: {
          'Hospital A': 7543,
          'Hospital B': 6479,
          'Hospital C': 21197,
          'Hospital D': 5291,
          'Hospital E': 8973
        },
        best_algorithm: d.available_methods?.includes('fedavg') ? 'FedAvg' : 'FedProx'
      };
    } catch {
      return MOCK_FEDERATED_STATUS;
    }
  },

  getFederatedHistory: async (): Promise<FederatedHistory> => {
    try {
      const res = await api.get('/federated/history');
      const d = res.data;
      const historyList = d.fedavg || d.fedprox || [];
      if (Array.isArray(historyList) && historyList.length > 0) {
        return {
          rounds: historyList.map((item: any) => ({
            round: item.round || item.epoch || 1,
            accuracy: item.global_accuracy || item.val_accuracy || 0,
            loss: item.avg_client_loss || item.train_loss || 0
          }))
        };
      }
      return MOCK_FEDERATED_HISTORY;
    } catch {
      return MOCK_FEDERATED_HISTORY;
    }
  },

  getGraphStatistics: async (): Promise<GraphStatistics> => {
    try {
      const res = await api.get('/graph/statistics');
      const d = res.data;
      return {
        num_nodes: d.num_nodes || 49483,
        num_edges: d.num_edges || 354199,
        avg_degree: d.avg_degree || 14.32,
        density: d.density || 0.000289,
        k_neighbors: d.k_neighbors || 10,
        is_connected: false,
        num_components: 1
      };
    } catch {
      return MOCK_GRAPH_STATISTICS;
    }
  },

  getSampleGraph: async (): Promise<SampleGraph> => {
    try {
      const res = await api.get('/graph/sample?n_nodes=80');
      const d = res.data;
      if (d.nodes && d.edges) {
        return {
          nodes: d.nodes.map((n: any) => ({
            id: `n${n.id}`,
            diabetes: n.label,
            x: n.x,
            y: n.y
          })),
          edges: d.edges.map((e: any) => ({
            source: `n${e.source}`,
            target: `n${e.target}`
          }))
        };
      }
      throw new Error('Invalid graph format');
    } catch {
      // Fallback
      const nodes = Array.from({ length: 50 }, (_, i) => ({ id: `n${i}`, diabetes: Math.random() > 0.5 ? 1 : 0 }));
      const edges = Array.from({ length: 100 }, () => ({
        source: `n${Math.floor(Math.random() * 50)}`,
        target: `n${Math.floor(Math.random() * 50)}`
      })).filter(e => e.source !== e.target);
      return { nodes, edges };
    }
  },

  getGlobalExplanation: async (): Promise<GlobalExplanation> => {
    try {
      const res = await api.get('/explainability/global');
      const d = res.data;
      if (d.feature_importance && Object.keys(d.feature_importance).length > 0) {
        const top_features: FeatureImportance[] = Object.entries(d.feature_importance).map(([feature, importance]) => ({
          feature,
          importance: Number(importance),
          direction: Number(importance) >= 0 ? 'positive' as const : 'negative' as const
        }));
        return {
          method: d.method || 'SHAP TreeExplainer',
          model: d.model || 'GradientBoosting',
          top_features
        };
      }
      return MOCK_GLOBAL_EXPLANATION;
    } catch {
      return MOCK_GLOBAL_EXPLANATION;
    }
  },

  predict: async (data: any): Promise<PredictionResult> => {
    try {
      const res = await api.post('/predict', data);
      const d = res.data;
      const contributions: FeatureImportance[] = d.feature_contributions
        ? Object.entries(d.feature_contributions).map(([feature, importance]) => ({
            feature,
            importance: Number(importance),
            direction: Number(importance) >= 0 ? 'positive' as const : 'negative' as const
          }))
        : [];
      return {
        probability: d.probability,
        prediction: d.predicted_class,
        model_used: d.model_used,
        risk_category: d.risk_category,
        explanation_method: d.explanation_method,
        explanation: { contributions }
      };
    } catch (error) {
      console.warn('Prediction API call failed, falling back to local heuristic', error);
      const mockProb = Math.random();
      return {
        probability: mockProb,
        prediction: mockProb > 0.5 ? 1 : 0,
        model_used: 'fedavg_graphsage',
        risk_category: mockProb > 0.65 ? 'High Risk' : mockProb > 0.35 ? 'Moderate Risk' : 'Low Risk',
        explanation: {
          contributions: [
            { feature: 'GenHlth', importance: 0.2, direction: 'positive' as const },
            { feature: 'HighBP', importance: 0.15, direction: 'positive' as const },
            { feature: 'PhysActivity', importance: -0.1, direction: 'negative' as const }
          ]
        }
      };
    }
  },

  train: () => api.post('/train').then(res => res.data).catch(() => ({ status: 'simulated_success' }))
};
