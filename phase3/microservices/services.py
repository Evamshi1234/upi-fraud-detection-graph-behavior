"""
UPI-Shield · Phase 3
Microservices Architecture — FastAPI
Services:
  /score   — Real-time fraud scoring
  /cases   — Case management (fraud alerts)
  /explain — SHAP-based explanation
  /health  — Liveness + readiness probes
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


# ─────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────

SCORE_REQUESTS   = Counter("upi_score_requests_total", "Total scoring requests", ["status"])
SCORE_LATENCY    = Histogram("upi_score_latency_seconds", "Scoring latency", buckets=[.005,.01,.025,.05,.1,.25,.5,1,2.5])
FRAUD_DETECTIONS = Counter("upi_fraud_detections_total", "Fraud detections", ["model", "decision"])
CASE_COUNTER     = Counter("upi_cases_total", "Cases created", ["severity"])


# ─────────────────────────────────────────────
# Domain Models
# ─────────────────────────────────────────────

class Decision(str, Enum):
    ALLOW   = "ALLOW"
    REVIEW  = "REVIEW"
    BLOCK   = "BLOCK"


class CaseSeverity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRITICAL = "CRITICAL"


class CaseStatus(str, Enum):
    OPEN       = "OPEN"
    ASSIGNED   = "ASSIGNED"
    RESOLVED   = "RESOLVED"
    ESCALATED  = "ESCALATED"


@dataclass
class FraudCase:
    case_id: str
    txn_id: str
    user_id: str
    amount: float
    fraud_score: float
    decision: Decision
    severity: CaseSeverity
    model_scores: dict[str, float]
    explanation: dict[str, float]
    timestamp: str
    status: CaseStatus = CaseStatus.OPEN
    assigned_to: str = ""
    notes: str = ""


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class ScoreRequest(BaseModel):
    txn_id: str = Field(..., example="TXN_20240615_001")
    user_id: str = Field(..., example="tok_abc123")
    merchant_id: str = Field(..., example="MER_XYZ")
    device_id: str = Field(..., example="DEV_001")
    amount: float = Field(..., gt=0, example=1500.0)
    timestamp: str = Field(..., example="2024-06-15T14:30:00Z")
    lat: float = Field(0.0, example=12.9716)
    lon: float = Field(0.0, example=77.5946)
    upi_app: str = Field("gpay", example="gpay")
    txn_type: str = Field("P2M", example="P2M")


class ScoreResponse(BaseModel):
    txn_id: str
    fraud_score: float                       # 0-1
    transformer_score: float
    gnn_score: float
    rl_adjusted_score: float
    decision: Decision
    latency_ms: float
    model_version: str
    explanation: dict[str, float]            # feature importances


class CaseResponse(BaseModel):
    case_id: str
    txn_id: str
    severity: str
    status: str
    fraud_score: float
    decision: str
    timestamp: str


class UpdateCaseRequest(BaseModel):
    status: CaseStatus
    assigned_to: str = ""
    notes: str = ""


# ─────────────────────────────────────────────
# In-Memory Stores (replaced by DB in production)
# ─────────────────────────────────────────────

_cases: dict[str, FraudCase] = {}


# ─────────────────────────────────────────────
# Mock Model Runner (replaced by real models in production)
# ─────────────────────────────────────────────

class ModelRunner:
    """
    Wraps Transformer + GNN + RL ensemble.
    In production loads ONNX models via ONNXRuntime for <10ms inference.
    """

    def __init__(self) -> None:
        import random
        self._rng = random.Random(42)
        self.version = "v2.1.0"

    def score(self, req: ScoreRequest) -> dict[str, Any]:
        # Simulate model scores (replace with actual model inference)
        import math, random

        base = 0.05
        if req.amount > 50_000:    base += 0.3
        if req.amount > 100_000:   base += 0.2
        if req.txn_type == "P2P" and req.amount > 10_000: base += 0.15

        t_score = min(1.0, base + random.uniform(-0.05, 0.15))
        g_score = min(1.0, base + random.uniform(-0.05, 0.12))
        ensemble = 0.5 * t_score + 0.5 * g_score
        rl_adj   = min(1.0, ensemble * random.uniform(0.9, 1.1))

        explanation = {
            "amount":            round(min(1.0, req.amount / 100_000), 3),
            "velocity_1h":       round(random.uniform(0, 0.5), 3),
            "geo_velocity":      round(random.uniform(0, 0.4), 3),
            "is_new_merchant":   0.2 if self._rng.random() > 0.7 else 0.0,
            "amount_zscore":     round(random.uniform(0, 0.6), 3),
            "device_novelty":    round(random.uniform(0, 0.3), 3),
        }

        return {
            "transformer_score":  round(t_score, 4),
            "gnn_score":          round(g_score, 4),
            "ensemble_score":     round(ensemble, 4),
            "rl_adjusted_score":  round(rl_adj, 4),
            "explanation":        explanation,
        }


_model_runner = ModelRunner()


def _make_decision(score: float) -> Decision:
    if score < 0.35:   return Decision.ALLOW
    if score < 0.70:   return Decision.REVIEW
    return Decision.BLOCK


def _severity(score: float) -> CaseSeverity:
    if score < 0.5:    return CaseSeverity.LOW
    if score < 0.7:    return CaseSeverity.MEDIUM
    if score < 0.85:   return CaseSeverity.HIGH
    return CaseSeverity.CRITICAL


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="UPI-Shield Fraud Detection API",
    description="Real-time AI-powered UPI transaction fraud scoring",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ──────────────────────────────────

@app.get("/health/live")
async def liveness():
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready")
async def readiness():
    # In production: check Redis, DB, model service connectivity
    return {"status": "ready", "model_version": _model_runner.version}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─── Score ───────────────────────────────────

@app.post("/v1/score", response_model=ScoreResponse)
async def score_transaction(req: ScoreRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    try:
        scores = _model_runner.score(req)
        fraud_score = scores["rl_adjusted_score"]
        decision    = _make_decision(fraud_score)
        latency_ms  = round((time.perf_counter() - t0) * 1000, 2)

        SCORE_REQUESTS.labels(status="success").inc()
        SCORE_LATENCY.observe(latency_ms / 1000)
        FRAUD_DETECTIONS.labels(model="ensemble", decision=decision.value).inc()

        # Create case for REVIEW or BLOCK decisions
        if decision in (Decision.REVIEW, Decision.BLOCK):
            background_tasks.add_task(
                _create_case, req, fraud_score, decision,
                scores, scores["explanation"]
            )

        return ScoreResponse(
            txn_id=req.txn_id,
            fraud_score=fraud_score,
            transformer_score=scores["transformer_score"],
            gnn_score=scores["gnn_score"],
            rl_adjusted_score=scores["rl_adjusted_score"],
            decision=decision,
            latency_ms=latency_ms,
            model_version=_model_runner.version,
            explanation=scores["explanation"],
        )
    except Exception as exc:
        SCORE_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc))


async def _create_case(
    req: ScoreRequest,
    fraud_score: float,
    decision: Decision,
    scores: dict,
    explanation: dict,
) -> None:
    case_id = f"CASE_{uuid.uuid4().hex[:10].upper()}"
    sev = _severity(fraud_score)
    case = FraudCase(
        case_id=case_id,
        txn_id=req.txn_id,
        user_id=req.user_id,
        amount=req.amount,
        fraud_score=fraud_score,
        decision=decision,
        severity=sev,
        model_scores={
            "transformer": scores["transformer_score"],
            "gnn":         scores["gnn_score"],
            "rl_adjusted": scores["rl_adjusted_score"],
        },
        explanation=explanation,
        timestamp=datetime.utcnow().isoformat(),
    )
    _cases[case_id] = case
    CASE_COUNTER.labels(severity=sev.value).inc()


# ─── Cases ───────────────────────────────────

@app.get("/v1/cases", response_model=list[CaseResponse])
async def list_cases(status: str | None = None, severity: str | None = None, limit: int = 50):
    cases = list(_cases.values())
    if status:
        cases = [c for c in cases if c.status.value == status.upper()]
    if severity:
        cases = [c for c in cases if c.severity.value == severity.upper()]
    cases = sorted(cases, key=lambda c: c.fraud_score, reverse=True)[:limit]
    return [
        CaseResponse(
            case_id=c.case_id, txn_id=c.txn_id, severity=c.severity.value,
            status=c.status.value, fraud_score=round(c.fraud_score, 4),
            decision=c.decision.value, timestamp=c.timestamp,
        )
        for c in cases
    ]


@app.get("/v1/cases/{case_id}")
async def get_case(case_id: str):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    return asdict(case)


@app.patch("/v1/cases/{case_id}")
async def update_case(case_id: str, update: UpdateCaseRequest):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    case.status = update.status
    if update.assigned_to:
        case.assigned_to = update.assigned_to
    if update.notes:
        case.notes = update.notes
    return {"message": "Updated", "case_id": case_id, "status": case.status.value}


# ─── Batch Score ─────────────────────────────

@app.post("/v1/score/batch")
async def score_batch(requests: list[ScoreRequest]):
    if len(requests) > 100:
        raise HTTPException(400, "Batch size exceeds maximum (100)")
    results = []
    for req in requests:
        scores = _model_runner.score(req)
        fraud_score = scores["rl_adjusted_score"]
        results.append({
            "txn_id": req.txn_id,
            "fraud_score": fraud_score,
            "decision": _make_decision(fraud_score).value,
        })
    return {"results": results, "count": len(results)}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("services:app", host="0.0.0.0", port=8000, reload=False, workers=4)
