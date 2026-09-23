"""
UPI-Shield · Main Orchestrator
Wires all 4 phases into a unified inference pipeline.
Entry point for the production scoring service.
"""

from __future__ import annotations
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

# Phase 1
from phase1.feature_engineering.pipeline import FeaturePipeline, Transaction
from phase1.feature_store.store import FeatureStore
from phase1.security.hardening import SecurityLayer

# Phase 2
from phase2.mlops.tracker import (
    ExperimentTracker, ModelRegistry, DriftDetector, ShadowEvaluator, ModelStage
)

# Phase 3
from phase3.kafka.streaming import (
    TransactionStreamProcessor, TransactionEvent, ScoredEvent, create_producer, TOPICS
)

# Phase 4
from phase4.rl_engine.ppo_agent import FraudEnvironment, PPOTrainer, PPOConfig, RLState
from phase4.autoscaling.k8s_and_chaos import CircuitBreaker, ChaosMonkey


# ─────────────────────────────────────────────
# UPI-Shield Engine
# ─────────────────────────────────────────────

class UPIShieldEngine:
    """
    Unified inference orchestrator.
    Phase 1 → feature extraction
    Phase 2 → Transformer + GNN scoring + MLOps monitoring
    Phase 3 → Kafka publish + case creation
    Phase 4 → RL threshold adjustment + circuit breakers
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        kafka_bootstrap: str = "localhost:9092",
        mlflow_uri: str = "http://localhost:5000",
    ) -> None:
        print("[UPI-Shield] Initialising engine...")

        # Phase 1
        self.feature_pipeline = FeaturePipeline()
        self.feature_store    = FeatureStore(redis_host=redis_host)
        self.security         = SecurityLayer()

        # Phase 2
        self.tracker  = ExperimentTracker(tracking_uri=mlflow_uri)
        self.registry = ModelRegistry()
        self.drift    = DriftDetector()
        self.shadow   = ShadowEvaluator()

        # Phase 4
        self.env        = FraudEnvironment()
        self.rl_trainer = PPOTrainer(self.env, PPOConfig())
        self.breaker    = CircuitBreaker(name="model-service", failure_threshold=0.4)
        self.chaos      = ChaosMonkey(enabled=False)  # enable during chaos testing

        # Producer (Phase 3)
        self.producer = create_producer(kafka_bootstrap)

        # Current thresholds (will be updated by RL agent)
        self.block_threshold  = 0.70
        self.review_threshold = 0.35

        print("[UPI-Shield] Engine ready.")

    # ── Phase 1: Feature Extraction ──────────

    def extract_features(self, txn: Transaction) -> dict[str, Any]:
        fv = self.feature_pipeline.transform(txn)
        features = {k: v for k, v in fv.__dict__.items() if k not in ("user_embed", "merchant_embed")}
        self.feature_store.ingest_transaction(
            txn.user_id, txn.txn_id, txn.amount, txn.timestamp, features
        )
        return features

    # ── Phase 2: Model Scoring ────────────────

    def _mock_model_score(self, features: dict[str, Any]) -> dict[str, float]:
        """
        Mock scoring — replace with actual Transformer + GNN inference.
        In production loads ONNX models via ONNXRuntime.
        """
        import random
        base = min(1.0, features.get("amount", 0) / 100_000 * 0.4)
        base += features.get("is_new_merchant", 0) * 0.15
        base += min(0.3, features.get("amount_zscore", 0) * 0.05)
        t_score = min(1.0, max(0.0, base + random.gauss(0, 0.08)))
        g_score = min(1.0, max(0.0, base + random.gauss(0, 0.06)))
        return {"transformer": t_score, "gnn": g_score}

    def score(self, features: dict[str, Any]) -> dict[str, float]:
        scores = self._mock_model_score(features)
        ensemble = 0.5 * scores["transformer"] + 0.5 * scores["gnn"]

        # Phase 4: RL threshold adjustment
        rl_thresholds = self.rl_trainer.get_policy_thresholds()
        self.block_threshold  = rl_thresholds["block_threshold"]
        self.review_threshold = rl_thresholds["review_threshold"]

        return {
            "transformer":    round(scores["transformer"], 4),
            "gnn":            round(scores["gnn"], 4),
            "ensemble":       round(ensemble, 4),
            "block_threshold":  self.block_threshold,
            "review_threshold": self.review_threshold,
        }

    def decide(self, ensemble_score: float) -> str:
        if ensemble_score >= self.block_threshold:   return "BLOCK"
        if ensemble_score >= self.review_threshold:  return "REVIEW"
        return "ALLOW"

    # ── Full Inference Pipeline ───────────────

    def process(self, txn: Transaction) -> dict[str, Any]:
        t0 = time.perf_counter()

        # Security check
        sec = self.security.check_request("api", txn.user_id, txn.amount, txn.timestamp)
        if not sec["rate_limit_passed"]:
            return {"txn_id": txn.txn_id, "decision": "RATE_LIMITED", "fraud_score": 0.0}

        # Feature extraction (Phase 1)
        features = self.extract_features(txn)

        # Scoring (Phase 2)
        scores = self.score(features)
        decision = self.decide(scores["ensemble"])

        # Drift monitoring (Phase 2)
        self.drift.add_production_sample(features, scores["ensemble"])

        # Shadow evaluation (Phase 2)
        self.shadow.record(
            prod_score=scores["transformer"],
            shadow_score=scores["gnn"],
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = {
            "txn_id":         txn.txn_id,
            "fraud_score":    scores["ensemble"],
            "transformer":    scores["transformer"],
            "gnn":            scores["gnn"],
            "decision":       decision,
            "latency_ms":     latency_ms,
            "block_threshold":  self.block_threshold,
            "review_threshold": self.review_threshold,
        }

        # Publish to Kafka (Phase 3)
        self._publish_score(txn, result)

        return result

    def _publish_score(self, txn: Transaction, result: dict) -> None:
        import json
        scored = ScoredEvent(
            txn_id=txn.txn_id,
            fraud_score=result["fraud_score"],
            decision=result["decision"],
            transformer_score=result["transformer"],
            gnn_score=result["gnn"],
            rl_score=result["fraud_score"],
            latency_ms=result["latency_ms"],
            model_version="v2.0",
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
        self.producer.send(TOPICS["scores"], value=scored.to_json())

    # ── RL Training (Phase 4) ─────────────────

    def train_rl_agent(self, n_iterations: int = 50) -> list[dict]:
        print("[UPI-Shield] Training RL threshold agent...")
        return self.rl_trainer.train(n_iterations)

    # ── Drift Report ──────────────────────────

    def drift_report(self) -> dict:
        report = self.drift.compute_report()
        return {
            "timestamp": report.timestamp,
            "prediction_drift": report.prediction_drift,
            "issues": report.data_quality_issues,
            "retraining_recommended": report.retraining_recommended,
        }

    # ── Shadow Report ─────────────────────────

    def shadow_report(self) -> dict:
        return self.shadow.compare()

    # ── Circuit Breaker Status ────────────────

    def health(self) -> dict:
        return {
            "status": "healthy",
            "circuit_breaker": self.breaker.status(),
            "thresholds": {
                "block":  self.block_threshold,
                "review": self.review_threshold,
            },
        }


# ─────────────────────────────────────────────
# Demo Run
# ─────────────────────────────────────────────

def run_demo() -> None:
    import random
    engine = UPIShieldEngine()

    print("\n" + "=" * 60)
    print("UPI-Shield — Live Inference Demo (10 transactions)")
    print("=" * 60)

    for i in range(10):
        txn = Transaction(
            txn_id=f"TXN_{i:04}",
            user_id=f"USR_{random.randint(1, 100):04}",
            merchant_id=f"MER_{random.randint(1, 50):03}",
            device_id=f"DEV_{random.randint(1, 20):03}",
            amount=random.choice([500, 1500, 12000, 75000, 150000]),
            timestamp=datetime.now(timezone.utc),
            lat=12.9716 + random.uniform(-0.5, 0.5),
            lon=77.5946 + random.uniform(-0.5, 0.5),
            upi_app=random.choice(["gpay", "phonepe", "paytm"]),
            txn_type=random.choice(["P2M", "P2P", "RECHARGE"]),
            is_new_merchant=random.random() > 0.8,
            is_new_device=random.random() > 0.9,
        )
        result = engine.process(txn)
        flag = "🚨" if result["decision"] == "BLOCK" else "⚠️" if result["decision"] == "REVIEW" else "✅"
        print(f"{flag} {txn.txn_id} | ₹{txn.amount:>8,.0f} | "
              f"Score={result['fraud_score']:.3f} | {result['decision']:8} | {result['latency_ms']}ms")

    print("\n--- Health ---")
    import json
    print(json.dumps(engine.health(), indent=2))

    print("\n--- Training RL Agent (5 iters) ---")
    engine.train_rl_agent(n_iterations=5)

    print("\n--- Drift Report ---")
    print(json.dumps(engine.drift_report(), indent=2))


if __name__ == "__main__":
    run_demo()