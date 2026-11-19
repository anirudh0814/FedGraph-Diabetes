"""
ml/preprocessing/preprocessor.py
DataPreprocessor for FedGraph-Diabetes
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config.config import (
    DATASET_PATH, PREPROCESSING_ARTIFACT, ARTIFACTS_DIR, RESULTS_DIR,
    TARGET_COLUMN, EXPECTED_COLUMNS, BINARY_FEATURES, NUMERIC_FEATURES,
    ORDINAL_FEATURES, RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    FEATURE_DESCRIPTIONS
)


class DataPreprocessor:
    """
    Handles loading, validation, cleaning, splitting and
    preprocessing of the CDC BRFSS 2015 diabetes dataset.

    If the real dataset is not found, automatically generates
    clearly-labelled DEMO synthetic data for UI testing.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        self.demo_mode = False
        self._df = None
        self._split_data = {}
        self._summary = {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def fit(self):
        """Load data, validate, preprocess and split."""
        df = self._load_data()
        self._df = df
        self._validate(df)
        df = self._handle_missing(df)
        X, y = self._separate_features_target(df)
        self.feature_columns = list(X.columns)
        X_scaled = self._scale(X, fit=True)
        self._do_split(X_scaled, y)
        self._compute_summary(df, y)
        self.is_fitted = True
        print(f"[Preprocessor] Fit complete. Demo mode: {self.demo_mode}")
        return self

    def transform(self, df_new: pd.DataFrame) -> np.ndarray:
        """
        Transform a new patient DataFrame using fitted scaler.
        df_new must contain the same feature columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load_artifacts() first.")
        # Ensure column order matches
        df_new = df_new[self.feature_columns].copy()
        X = df_new.values.astype(np.float32)
        # Apply scaler only to numeric feature columns
        numeric_idx = [
            self.feature_columns.index(c)
            for c in NUMERIC_FEATURES
            if c in self.feature_columns
        ]
        X[:, numeric_idx] = self.scaler.transform(X[:, numeric_idx])
        return X

    def save_artifacts(self):
        """Persist scaler + metadata for later inference."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'demo_mode': self.demo_mode,
            'summary': self._summary,
        }
        with open(PREPROCESSING_ARTIFACT, 'wb') as f:
            pickle.dump(artifact, f)
        print(f"[Preprocessor] Artifacts saved to {PREPROCESSING_ARTIFACT}")

    @classmethod
    def load_artifacts(cls):
        """Load a previously fitted preprocessor from disk."""
        if not PREPROCESSING_ARTIFACT.exists():
            raise FileNotFoundError(
                f"Preprocessing artifact not found at {PREPROCESSING_ARTIFACT}. "
                "Run train_baselines.py first."
            )
        with open(PREPROCESSING_ARTIFACT, 'rb') as f:
            artifact = pickle.load(f)
        obj = cls()
        obj.scaler = artifact['scaler']
        obj.feature_columns = artifact['feature_columns']
        obj.demo_mode = artifact['demo_mode']
        obj._summary = artifact.get('summary', {})
        obj.is_fitted = True
        return obj

    def get_split_data(self) -> dict:
        """Return X_train, X_val, X_test, y_train, y_val, y_test."""
        if not self._split_data:
            self.fit()
        return self._split_data

    def get_summary(self) -> dict:
        """Return dataset statistics."""
        return self._summary

    # ─────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        if DATASET_PATH.exists():
            print(f"[Preprocessor] Loading real dataset from {DATASET_PATH}")
            df = pd.read_csv(DATASET_PATH)
            self.demo_mode = False
        else:
            print(f"[Preprocessor] Dataset not found. Generating DEMO synthetic data.")
            print(f"  ⚠  DEMO MODE: Results are NOT real experimental results.")
            df = self._generate_demo_data()
            self.demo_mode = True
        return df

    def _validate(self, df: pd.DataFrame):
        """Check required columns exist."""
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Dataset is missing required columns: {missing_cols}. "
                f"Please check data/diabetes.csv matches the expected format."
            )
        print(f"[Preprocessor] Validation passed. Shape: {df.shape}")

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values: median for numeric, mode for categorical."""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            print(f"[Preprocessor] Found {total_nulls} missing values. Filling...")
            for col in NUMERIC_FEATURES:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
            for col in BINARY_FEATURES + ORDINAL_FEATURES + [TARGET_COLUMN]:
                if col in df.columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def _separate_features_target(self, df: pd.DataFrame):
        feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
        X = df[feature_cols].astype(np.float32)
        y = df[TARGET_COLUMN].astype(np.int64)
        return X, y

    def _scale(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale only NUMERIC_FEATURES; leave binary/ordinal as-is."""
        X_arr = X.values.astype(np.float32)
        numeric_idx = [
            list(X.columns).index(c)
            for c in NUMERIC_FEATURES
            if c in X.columns
        ]
        if fit:
            self.scaler.fit(X_arr[:, numeric_idx])
        X_arr[:, numeric_idx] = self.scaler.transform(X_arr[:, numeric_idx])
        return X_arr

    def _do_split(self, X: np.ndarray, y: pd.Series):
        """Stratified train/val/test split with no data leakage."""
        y_arr = y.values
        # First split off test set
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y_arr, test_size=TEST_SIZE,
            stratify=y_arr, random_state=RANDOM_SEED
        )
        # Then split val from train
        val_fraction = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction,
            stratify=y_trainval,
            random_state=RANDOM_SEED
        )
        self._split_data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        }
        print(f"[Preprocessor] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    def _compute_summary(self, df: pd.DataFrame, y: pd.Series):
        """Build summary statistics dictionary."""
        class_counts = y.value_counts().to_dict()
        feature_stats = {}
        feature_cols = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
        for col in feature_cols:
            if col in df.columns:
                feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'null_count': int(df[col].isnull().sum()),
                    'type': ('numeric' if col in NUMERIC_FEATURES
                             else 'binary' if col in BINARY_FEATURES
                             else 'ordinal'),
                    'description': FEATURE_DESCRIPTIONS.get(col, col)
                }
        self._summary = {
            'n_records': int(len(df)),
            'n_features': len(feature_cols),
            'target_column': TARGET_COLUMN,
            'class_distribution': {
                'no_diabetes': int(class_counts.get(0, 0)),
                'diabetes': int(class_counts.get(1, 0)),
            },
            'demo_mode': self.demo_mode,
            'dataset_path': str(DATASET_PATH),
            'missing_values': int(df.isnull().sum().sum()),
            'feature_stats': feature_stats,
        }
        # Save summary to results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / 'dataset_summary.json', 'w') as f:
            json.dump(self._summary, f, indent=2)

    def _generate_demo_data(self) -> pd.DataFrame:
        """
        Generate realistic-looking DEMO synthetic data.
        Based on approximate CDC BRFSS 2015 marginal distributions.
        NOT real patient data. Clearly marked as DEMO.
        """
        np.random.seed(RANDOM_SEED)
        n = 70692
        # Target: balanced 50/50
        target = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
        np.random.shuffle(target)

        df = pd.DataFrame()
        df[TARGET_COLUMN] = target.astype(int)

        # Base latent risk factor for each patient (cluster-based)
        latent_clusters = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.25, 0.25, 0.20, 0.15, 0.15])
        # Diabetics have cluster distribution shifted towards high-risk phenotype clusters (3 & 4)
        is_diab = (df[TARGET_COLUMN] == 1).values
        
        # Cluster-conditioned indicator probabilities
        # Cluster 0: Healthy young
        # Cluster 1: Average adult
        # Cluster 2: Older mild risk
        # Cluster 3: Metabolic risk (High BP + High BMI + High Chol)
        # Cluster 4: Chronic multi-condition
        for feat in BINARY_FEATURES:
            base_p = 0.35 if feat in ['HighBP', 'HighChol'] else 0.10 if feat in ['Stroke', 'HeartDiseaseorAttack'] else 0.50
            if feat in ['HighBP', 'HighChol']:
                p_c = np.array([0.20, 0.35, 0.50, 0.75, 0.85])
            elif feat in ['HeartDiseaseorAttack', 'Stroke', 'DiffWalk']:
                p_c = np.array([0.02, 0.05, 0.10, 0.25, 0.40])
            elif feat in ['PhysActivity', 'Fruits', 'Veggies']:
                p_c = np.array([0.85, 0.78, 0.70, 0.55, 0.45])
            else:
                p_c = np.array([0.30, 0.40, 0.50, 0.60, 0.70])

            # Sample based on cluster + diabetes status
            prob_vec = p_c[latent_clusters] * (1.15 if feat not in ['PhysActivity', 'Fruits', 'Veggies'] else 0.9)
            prob_vec = np.clip(np.where(is_diab, prob_vec * 1.25, prob_vec * 0.85), 0.02, 0.98)
            df[feat] = np.random.binomial(1, prob_vec, n).astype(float)

        # BMI conditioned on cluster
        cluster_bmi_means = np.array([23.5, 26.5, 28.5, 33.5, 36.0])
        df['BMI'] = np.clip(np.random.normal(cluster_bmi_means[latent_clusters], 4.8, n), 12, 98)

        # MentHlth, PhysHlth conditioned on cluster
        cluster_hlth_exp = np.array([1.5, 2.5, 4.0, 7.5, 12.0])
        df['MentHlth'] = np.clip(np.random.exponential(cluster_hlth_exp[latent_clusters], n), 0, 30).astype(float)
        df['PhysHlth'] = np.clip(np.random.exponential(cluster_hlth_exp[latent_clusters] * 1.2, n), 0, 30).astype(float)

        # GenHlth: 1-5 conditioned on cluster
        cluster_gen_means = np.array([1.5, 2.2, 2.9, 3.8, 4.4])
        df['GenHlth'] = np.clip(np.random.normal(cluster_gen_means[latent_clusters], 0.8, n).round(), 1, 5).astype(float)

        # Age: 1-13 conditioned on cluster
        cluster_age_means = np.array([3.5, 6.0, 8.5, 10.0, 11.2])
        df['Age'] = np.clip(np.random.normal(cluster_age_means[latent_clusters], 2.2, n).round(), 1, 13).astype(float)

        # Education: 1-6
        df['Education'] = np.clip(np.random.randint(1, 7, n), 1, 6).astype(float)
        # Income: 1-8
        df['Income'] = np.where(
            is_diab,
            np.clip(np.random.normal(5.0, 2.0, n), 1, 8).round(),
            np.clip(np.random.normal(6.2, 1.8, n), 1, 8).round()
        ).astype(float)

        # Ensure all expected columns exist
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        return df[EXPECTED_COLUMNS]
