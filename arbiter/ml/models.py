"""ML models — runtime prediction and failure classification."""

import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score

import joblib

from arbiter.models.task import Task
from arbiter.models.worker import Worker


RESOURCE_ENCODING = {"cpu": 0, "gpu": 1, "memory": 2}


FEATURE_COLUMNS = [
    "compute_cost", "estimated_duration", "priority", "failure_probability",
    "dependency_count", "resource_type_encoded",
    "worker_speed", "worker_capacity", "worker_load_ratio",
]


def _extract_features(data: list[dict]) -> np.ndarray:
    """Convert list of feature dicts to numpy array."""
    return np.array([[d[col] for col in FEATURE_COLUMNS] for d in data])


def _extract_target(data: list[dict], target_column: str) -> np.ndarray:
    """Extract a single target column from feature dicts."""
    return np.array([d[target_column] for d in data])


class RuntimePredictor:
    """Predicts actual task runtime using regression."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.metrics: dict = {}

    def train(self, data: list[dict], test_size: float = 0.2) -> dict:
        """Train the model and return evaluation metrics."""
        X = _extract_features(data)
        y = _extract_target(data, "actual_runtime")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if self.model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        self.is_trained = True
        return self.metrics

    def predict(self, task: Task, worker: Worker) -> float:
        """Predict runtime for a task on a worker. Falls back to estimated_duration."""
        if not self.is_trained or self.model is None:
            return task.estimated_duration

        features = np.array([[
            task.compute_cost, task.estimated_duration, task.priority,
            task.failure_probability, len(task.dependencies),
            RESOURCE_ENCODING.get(task.resource_type, 0),
            worker.speed_multiplier, worker.cpu_capacity,
            task.compute_cost / worker.cpu_capacity,
        ]])

        prediction = self.model.predict(features)[0]
        return max(0.1, float(prediction))

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({"model": self.model, "model_type": self.model_type}, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.is_trained = True


class FailureClassifier:
    """Predicts task failure probability using classification."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.metrics: dict = {}

    def train(self, data: list[dict], test_size: float = 0.2) -> dict:
        """Train the classifier and return evaluation metrics."""
        X = _extract_features(data)
        y = _extract_target(data, "did_fail")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Handle case where test set might have only one class
        unique_classes = np.unique(y_test)
        if len(unique_classes) > 1:
            self.metrics = {
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "failure_rate_train": float(y_train.mean()),
            }
        else:
            self.metrics = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "train_samples": len(X_train), "test_samples": len(X_test),
                "failure_rate_train": float(y_train.mean()),
                "note": "Only one class in test set",
            }

        self.is_trained = True
        return self.metrics

    def predict_proba(self, task: Task, worker: Worker) -> float:
        """Predict failure probability [0, 1]. Falls back to task.failure_probability."""
        if not self.is_trained or self.model is None:
            return task.failure_probability

        features = np.array([[
            task.compute_cost, task.estimated_duration, task.priority,
            task.failure_probability, len(task.dependencies),
            RESOURCE_ENCODING.get(task.resource_type, 0),
            worker.speed_multiplier, worker.cpu_capacity,
            task.compute_cost / worker.cpu_capacity,
        ]])

        proba = self.model.predict_proba(features)
        # Column 1 = probability of failure (class 1)
        if proba.shape[1] > 1:
            return float(proba[0][1])
        return task.failure_probability

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({"model": self.model}, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.is_trained = True
