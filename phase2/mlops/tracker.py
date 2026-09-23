"""
UPI-Shield · Phase 2
MLOps Pipeline
- MLflow experiment tracking & model registry
- Evidently-based data/model drift detection
- Automated retraining triggers
- Shadow-mode deployment support
- Model versioning with approval workflow
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[MLOps] MLflow not available, using mock logging.")


# ─────────────────────────────────────────────
# Model Lifecycle States
# ─────────────────────────────────────────────

class ModelStage(str, Enum):
    DEVELOPMENT  = "development"
    STAGING      = "staging"
    SHADOW       = "shadow"       # live traffic but not serving decisions
    PRODUCTION   = "production"
    DEPRECATED   = "deprecated"


@dataclass
class ModelVersion:
    model_name: str
    version: str
    stage: ModelStage
    run_id: str
    metrics: dict[str, float]
    artifact_path: str
    created_at: str = ""
    approved_by: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


# ─────────────────────────────────────────────
# Experiment Tracker (MLflow wrapper)
# ─────────────────────────────────────────────

class ExperimentTracker:
    """
    Wraps MLflow for experiment tracking.
    Falls back to a local JSON log when MLflow server is unavailable.
    """

    def __init__(self, experiment_name: str = "upi-shield-fraud-detection", tracking_uri: str = "http://localhost:5000") -> None:
        self.experiment_name = experiment_name
        self._local_log: list[dict] = []

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._use_mlflow = True
                print(f"[MLOps] Connected to MLflow at {tracking_uri}")
            except Exception:
                self._use_mlflow = False
        else:
            self._use_mlflow = False

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        run_id = str(uuid.uuid4())[:8]
        if self._use_mlflow:
            mlflow.start_run(run_name=run_name, tags=tags or {})
            run_id = mlflow.active_run().info.run_id
        self._current_run = {"run_id": run_id, "name": run_name, "metrics": {}, "params": {}}
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._use_mlflow:
            mlflow.log_params(params)
        if hasattr(self, "_current_run"):
            self._current_run["params"].update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        if hasattr(self, "_current_run"):
            self._current_run["metrics"].update(metrics)

    def log_model(self, model: Any, artifact_path: str) -> None:
        if self._use_mlflow:
            try:
                mlflow.pytorch.log_model(model, artifact_path)
            except Exception:
                pass

    def end_run(self) -> dict:
        if self._use_mlflow:
            mlflow.end_run()
        run = getattr(self, "_current_run", {})
        self._local_log.append(run)
        return run

    def get_best_run(self, metric: str = "f1", higher_is_better: bool = True) -> dict | None:
        if not self._local_log:
            return None
        return sorted(
            self._local_log,
            key=lambda r: r["metrics"].get(metric, 0.0),
            reverse=higher_is_better,
        )[0]


# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────

class ModelRegistry:
    """
    Tracks model versions, stages, and approvals.
    Persists to a JSON file; in production this is backed by MLflow Model Registry
    or a database table.
    """

    def __init__(self, registry_path: str = "/tmp/upi_shield/model_registry.json") -> None:
        self.path = registry_path
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        self._versions: list[ModelVersion] = self._load()

    def _load(self) -> list[ModelVersion]:
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            return [ModelVersion(**d) for d in data]
        return []

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump([asdict(v) for v in self._versions], f, indent=2)

    def register(self, version: ModelVersion) -> None:
        self._versions.append(version)
        self._save()
        print(f"[Registry] Registered {version.model_name} v{version.version} [{version.stage.value}]")

    def transition(self, model_name: str, version: str, new_stage: ModelStage, approver: str = "system") -> None:
        for v in self._versions:
            if v.model_name == model_name and v.version == version:
                v.stage = new_stage
                v.approved_by = approver
                self._save()
                print(f"[Registry] {model_name} v{version} → {new_stage.value} (by {approver})")
                return
        raise ValueError(f"Model {model_name} v{version} not found in registry.")

    def get_production(self, model_name: str) -> ModelVersion | None:
        prod = [v for v in self._versions if v.model_name == model_name and v.stage == ModelStage.PRODUCTION]
        return prod[-1] if prod else None

    def get_shadow(self, model_name: str) -> ModelVersion | None:
        shadow = [v for v in self._versions if v.model_name == model_name and v.stage == ModelStage.SHADOW]
        return shadow[-1] if shadow else None

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        return [v for v in self._versions if v.model_name == model_name]


# ─────────────────────────────────────────────
# Drift Detector
# ─────────────────────────────────────────────

@dataclass
class DriftReport:
    timestamp: str
    feature_drift: dict[str, float]      # feature_name -> drift_score (0-1)
    prediction_drift: float              # KL divergence on score distribution
    data_quality_issues: list[str]
    retraining_recommended: bool

    @property
    def has_critical_drift(self) -> bool:
        return self.prediction_drift > 0.3 or any(v > 0.5 for v in self.feature_drift.values())


class DriftDetector:
    """
    Detects data drift using Population Stability Index (PSI) for features
    and KL divergence for prediction score distributions.
    Reference window = training data distribution.
    Production window = recent N transactions.
    """

    PSI_THRESHOLD   = 0.2   # moderate drift
    SCORE_KL_THRESHOLD = 0.15

    def __init__(self, reference_df: pd.DataFrame | None = None) -> None:
        self._reference: pd.DataFrame | None = reference_df
        self._production_buffer: list[dict] = []
        self._score_buffer: list[float] = []
        self._ref_scores: list[float] = []

    def set_reference(self, df: pd.DataFrame, scores: list[float]) -> None:
        self._reference = df
        self._ref_scores = scores

    def add_production_sample(self, features: dict[str, float], score: float) -> None:
        self._production_buffer.append(features)
        self._score_buffer.append(score)

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index."""
        exp_hist, bin_edges = np.histogram(expected, bins=bins)
        act_hist, _ = np.histogram(actual, bins=bin_edges)
        exp_pct = (exp_hist + 1e-6) / len(expected)
        act_pct = (act_hist + 1e-6) / len(actual)
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        p /= p.sum(); q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    def compute_report(self, window_size: int = 1000) -> DriftReport:
        if self._reference is None or len(self._production_buffer) < 100:
            return DriftReport(
                timestamp=datetime.utcnow().isoformat(),
                feature_drift={},
                prediction_drift=0.0,
                data_quality_issues=["Insufficient production data for drift analysis"],
                retraining_recommended=False,
            )

        prod_df = pd.DataFrame(self._production_buffer[-window_size:])
        feature_drift: dict[str, float] = {}
        issues: list[str] = []

        for col in self._reference.columns:
            if col not in prod_df.columns:
                issues.append(f"Missing feature in production: {col}")
                continue
            if pd.api.types.is_numeric_dtype(self._reference[col]):
                psi = self._psi(
                    self._reference[col].dropna().values,
                    prod_df[col].dropna().values,
                )
                feature_drift[col] = round(psi, 4)
                if psi > self.PSI_THRESHOLD:
                    issues.append(f"Feature drift detected: {col} (PSI={psi:.3f})")

        # Score distribution drift
        pred_drift = 0.0
        if self._ref_scores and self._score_buffer:
            ref_hist, bins = np.histogram(self._ref_scores, bins=20, range=(0, 1))
            prod_hist, _   = np.histogram(self._score_buffer[-window_size:], bins=bins)
            pred_drift = self._kl_divergence(ref_hist.astype(float), prod_hist.astype(float))

        return DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            feature_drift=feature_drift,
            prediction_drift=round(pred_drift, 4),
            data_quality_issues=issues,
            retraining_recommended=(pred_drift > self.SCORE_KL_THRESHOLD or
                                    any(v > self.PSI_THRESHOLD for v in feature_drift.values())),
        )


# ─────────────────────────────────────────────
# Shadow Mode Evaluator
# ─────────────────────────────────────────────

class ShadowEvaluator:
    """
    Runs a challenger model in shadow mode alongside the production model.
    Collects scores from both; compares after N samples.
    """

    def __init__(self, min_samples: int = 500) -> None:
        self.min_samples = min_samples
        self._prod_scores: list[float] = []
        self._shadow_scores: list[float] = []
        self._labels: list[int] = []

    def record(self, prod_score: float, shadow_score: float, label: int | None = None) -> None:
        self._prod_scores.append(prod_score)
        self._shadow_scores.append(shadow_score)
        if label is not None:
            self._labels.append(label)

    def ready_to_compare(self) -> bool:
        return len(self._prod_scores) >= self.min_samples

    def compare(self) -> dict[str, Any]:
        if not self.ready_to_compare():
            return {"status": "insufficient_data", "n": len(self._prod_scores)}

        prod = np.array(self._prod_scores)
        shadow = np.array(self._shadow_scores)

        report: dict[str, Any] = {
            "n_samples": len(prod),
            "prod_mean_score": round(float(prod.mean()), 4),
            "shadow_mean_score": round(float(shadow.mean()), 4),
            "score_correlation": round(float(np.corrcoef(prod, shadow)[0, 1]), 4),
        }

        if self._labels:
            labels = np.array(self._labels)
            for name, scores in [("production", prod), ("shadow", shadow)]:
                threshold = 0.5
                preds = (scores >= threshold).astype(int)
                tp = ((preds == 1) & (labels == 1)).sum()
                fp = ((preds == 1) & (labels == 0)).sum()
                fn = ((preds == 0) & (labels == 1)).sum()
                prec = tp / (tp + fp + 1e-8)
                rec  = tp / (tp + fn + 1e-8)
                f1   = 2 * prec * rec / (prec + rec + 1e-8)
                report[f"{name}_f1"]        = round(float(f1), 4)
                report[f"{name}_precision"] = round(float(prec), 4)
                report[f"{name}_recall"]    = round(float(rec), 4)

        report["promote_shadow"] = (
            self._labels and
            report.get("shadow_f1", 0) > report.get("production_f1", 0) * 1.02  # 2% improvement threshold
        )
        return report
