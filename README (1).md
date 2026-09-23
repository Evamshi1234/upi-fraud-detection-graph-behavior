⬡ UPI-Shield — AI-Powered UPI Fraud Detection Platform

Real-time fraud detection for UPI transactions using Transformer models, Graph Neural Networks, Reinforcement Learning, and Kafka streaming. Built across 4 phases from feature engineering to production-grade microservices.


📋 Table of Contents

Project Overview
Architecture
Project Structure
Prerequisites
Installation
How to Run — Step by Step

Run the Main Engine
Run the FastAPI Scoring API
Run MLflow Dashboard
Run the Tests
Open the Website


Input Format
Output Format
API Endpoints
Phase Breakdown
Fallback Behaviour
CSV Test Files
Key Design Decisions
KPIs & Performance Targets
Known Warnings
Tech Stack


Project Overview
UPI-Shield is an end-to-end fraud detection system purpose-built for India's UPI (Unified Payments Interface) ecosystem. It processes transactions in real time, extracts 30+ risk features, scores them using an ensemble of ML models, and makes a three-tier decision: ALLOW, REVIEW, or BLOCK.
The system is designed to handle 10M+ transactions per day with sub-2ms inference latency and a false positive rate capped at 0.5%.
What makes it different from a simple ML model
LayerWhat it doesFeature Engineering6 specialised feature extractors including velocity windows, geo-velocity, and amount z-scoresTransformer modelBERT-style encoder over the user's last 50 transactions for contextual anomaly detectionGraph Neural NetworkDetects fraud rings — clusters of accounts coordinating to defraudRL Agent (PPO)Dynamically adjusts BLOCK/REVIEW thresholds based on real-time performanceMLflow MLOpsDrift detection, model registry, shadow evaluation, auto-retraining triggersKafka streamingAsynchronous event pipeline for high-throughput processingFastAPIREST API with Prometheus metrics, CORS, and background case management

Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        UPI TRANSACTION                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Security Layer  │  Rate limiting · PII tokenisation
                    │  (Phase 1)       │  JWT RBAC · Adversarial detection
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Feature Pipeline         │
              │     (Phase 1)                │
              │  VelocityEngineer            │  1h / 24h / 7d windows
              │  AmountStatEngineer          │  Z-score vs user history
              │  TemporalEngineer            │  Cyclical sin/cos encoding
              │  GeoEngineer                 │  Haversine distance
              │  ChannelEngineer             │  UPI app + transaction type
              │  FrequencyEngineer           │  Per-user / merchant / device
              └──────┬───────────┬──────────┘
                     │           │
          ┌──────────▼──┐   ┌────▼──────────┐
          │  Redis Store │   │ Parquet Store  │
          │  (online)    │   │ (offline/S3)   │
          │  TTL = 7d    │   │ Point-in-time  │
          └──────┬───────┘   └───────────────┘
                 │
     ┌───────────▼────────────┐
     │     Model Ensemble      │
     │     (Phase 2)           │
     │  Transformer Encoder    │  BERT-style · 4 layers · CLS token
     │  GNN (GraphSAGE + GAT)  │  Fraud ring detection
     │  Ensemble Score         │  0.5×Transformer + 0.5×GNN
     └───────────┬─────────────┘
                 │
     ┌───────────▼────────────┐
     │   RL Threshold Engine   │
     │   (Phase 4 — PPO)       │
     │  BLOCK  ≥ dynamic       │  Default 0.70
     │  REVIEW ≥ dynamic       │  Default 0.35
     │  ALLOW  < review        │
     └───────────┬─────────────┘
                 │
     ┌───────────▼─────────────────────────────┐
     │             Decision Output              │
     │  ALLOW → published to Kafka              │
     │  REVIEW → case created for analyst       │
     │  BLOCK  → transaction blocked + alert    │
     └──────────────────────────────────────────┘
                 │
     ┌───────────▼────────────┐
     │   MLOps Monitoring      │
     │   (Phase 2)             │
     │  MLflow experiment log  │
     │  PSI drift detection    │
     │  Shadow evaluator       │
     │  Auto-retraining check  │
     └────────────────────────┘

Project Structure
upi-shield/
│
├── main.py                          ← Entry point. Runs the full demo pipeline.
├── requirements.txt                 ← All Python dependencies
├── setup_structure.py               ← One-time setup script (creates __init__.py files)
├── log_runs.py                      ← Logs 5 sample experiment runs to MLflow
├── upi-shield-website.html          ← Standalone web dashboard (double-click to open)
│
├── phase1/                          ← Data pipeline & security
│   ├── feature_engineering/
│   │   └── pipeline.py              ← 6 feature extractors + FeaturePipeline orchestrator
│   ├── feature_store/
│   │   └── store.py                 ← Redis online store + Parquet offline store
│   ├── security/
│   │   └── hardening.py             ← PII tokeniser, AES-128 encryption, JWT RBAC, rate limiter
│   └── tests/
│       └── test_phase1.py           ← 40 pytest unit tests
│
├── phase2/                          ← AI models & MLOps
│   ├── transformer/
│   │   └── model.py                 ← BERT-style encoder, ONNX export
│   ├── gnn/
│   │   └── model.py                 ← GraphSAGE + GAT, fraud ring detection
│   └── mlops/
│       └── tracker.py               ← MLflow wrapper, ModelRegistry, DriftDetector, ShadowEvaluator
│
├── phase3/                          ← Streaming & API
│   ├── microservices/
│   │   └── services.py              ← FastAPI: /score, /score/batch, /cases, /health, /metrics
│   └── kafka/
│       └── streaming.py             ← TransactionStreamProcessor, FeedbackConsumer, mock fallback
│
└── phase4/                          ← RL & Infrastructure
    ├── rl_engine/
    │   └── ppo_agent.py             ← FraudEnvironment (MDP), PPOTrainer (GAE), threshold policy
    └── autoscaling/
        └── k8s_and_chaos.py         ← K8s HPA+KEDA manifest generators, CircuitBreaker, ChaosMonkey

Prerequisites
RequirementVersionNotesPython3.10+3.12 recommendedpiplatestpython -m pip install --upgrade pipRedisoptionalFalls back to in-memory dict if unavailableKafkaoptionalFalls back to mock producer if unavailableGitoptionalOnly needed to silence an MLflow warning

Windows users: All fallbacks are implemented. You do NOT need Redis or Kafka installed to run the project. Everything works out of the box.


## Quick Start

```bash
pip install -r requirements.txt

# Run the full demo
python main.py

# Run tests
python -m pytest phase1/tests/ -v --cov=phase1

# Start API server
cd phase3\microservices; python -m uvicorn services:app --reload --port 8000

# MLflow UI
python -m mlflow ui --port 5000
```

## Installation
Step 1 — Clone / download the project
Place the upi-shield folder anywhere on your machine.
Recommended: C:\Users\<you>\OneDrive\Desktop\upi-shield\
Step 2 — Open PowerShell in the project folder
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield
Step 3 — Install dependencies
powershellpip install -r requirements.txt
This installs everything: PyTorch, FastAPI, MLflow, Kafka client, Redis client, scikit-learn, and more. Takes 3–5 minutes on first install.
Step 4 — Run the one-time setup script (Windows only)
powershellpython setup_structure.py
This creates all __init__.py files so Python can find the modules across phases. Only needs to be run once.

How to Run
1. Run the Main Engine
powershellpython main.py
What it does:

Initialises all 4 phases
Runs a live demo of 10 transactions
Prints health check (circuit breaker status + thresholds)
Trains the RL agent for 5 iterations
Prints a drift report

Expected output:
[MLOps] MLflow not available, using mock logging.
[UPI-Shield] Initialising engine...
Kafka unavailable (...), using mock producer.
[UPI-Shield] Engine ready.

============================================================
UPI-Shield — Live Inference Demo (10 transactions)
============================================================
✅ TXN_0000 | ₹  75,000 | Score=0.383 | ALLOW    | 0.96ms
⚠️ TXN_0001 | ₹  75,000 | Score=0.434 | REVIEW   | 0.60ms
✅ TXN_0002 | ₹  12,000 | Score=0.000 | ALLOW    | 0.34ms
⚠️ TXN_0005 | ₹ 150,000 | Score=0.633 | REVIEW   | 0.25ms
✅ TXN_0006 | ₹     500 | Score=0.029 | ALLOW    | 0.20ms

--- Health ---
{
  "status": "healthy",
  "circuit_breaker": {
    "name": "model-service",
    "state": "closed",
    "call_count": 0,
    "failure_count": 0,
    "failure_rate": 0.0
  },
  "thresholds": {
    "block": 0.75,
    "review": 0.3
  }
}

--- Training RL Agent (5 iters) ---
[UPI-Shield] Training RL threshold agent...
[RL] Iter 000 | Loss=0.0000 | MeanReward=0.000 | Steps=2048

--- Drift Report ---
{
  "timestamp": "2026-03-09T...",
  "prediction_drift": 0.0,
  "issues": ["Insufficient production data for drift analysis"],
  "retraining_recommended": false
}
Output meaning:
SymbolDecisionScore RangeMeaning✅ALLOW0.00 – 0.34Transaction is safe, proceed normally⚠️REVIEW0.35 – 0.69Suspicious, flagged for human analyst review🚨BLOCK0.70 – 1.00High-confidence fraud, transaction blocked

2. Run the FastAPI Scoring API
Open a new PowerShell window (keep main.py terminal open if you want):
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield
python -m uvicorn phase3.microservices.services:app --port 8000 --reload
Expected output:
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
Test it — score a single transaction:
Open your browser and go to:
http://127.0.0.1:8000/docs
This opens the interactive Swagger UI where you can test every endpoint with a form.
Or test from PowerShell:
powershellInvoke-RestMethod -Uri "http://127.0.0.1:8000/v1/health/live" -Method GET
Score a transaction via API:
powershell$body = @{
    txn_id     = "TXN_TEST_001"
    user_id    = "USR_042"
    merchant_id = "MER_099"
    device_id  = "DEV_007"
    amount     = 85000
    upi_app    = "gpay"
    txn_type   = "P2P"
    lat        = 12.97
    lon        = 77.59
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/score" -Method POST -Body $body -ContentType "application/json"
Expected API response:
json{
  "txn_id": "TXN_TEST_001",
  "fraud_score": 0.4821,
  "decision": "REVIEW",
  "transformer_score": 0.51,
  "gnn_score": 0.45,
  "explanation": {
    "amount_zscore": 0.31,
    "velocity_1h": 0.12,
    "is_new_merchant": 0.15
  },
  "case_id": "CASE-abc123",
  "latency_ms": 1.84,
  "model_version": "v2.0"
}

3. Run MLflow Dashboard
Open a new PowerShell window:
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield\phase3\microservices
python -m mlflow ui --port 5000
Then open your browser:
http://127.0.0.1:5000
To populate it with experiment data:
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield
python log_runs.py
What you'll see in MLflow:

Experiment called "king"
5 runs: upi_shield_v1 through upi_shield_v5
Each run logged with:

Params: model_type, threshold_block, threshold_review
Metrics: fraud_recall, false_pos_rate, f1_score, latency_ms


Click Compare to see side-by-side metric charts across all runs


4. Run the Tests
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield
python -m pytest phase1/tests/ -v --tb=short
Expected output:
collected 40 items

phase1/tests/test_phase1.py::test_velocity_feature_basic         PASSED
phase1/tests/test_phase1.py::test_amount_zscore                  PASSED
phase1/tests/test_phase1.py::test_temporal_cyclical_encoding     PASSED
phase1/tests/test_phase1.py::test_geo_velocity                   PASSED
phase1/tests/test_phase1.py::test_security_rate_limiter          PASSED
...
40 passed in 3.2s
To run with coverage report:
powershellpython -m pytest phase1/tests/ -v --cov=phase1 --cov-report=term-missing

5. Open the Website
No server needed. Just double-click:
upi-shield-website.html
Opens in your browser. Features:

Live transaction feed — streams real-time simulated transactions scored by the actual risk engine
CSV upload — drag and drop any of the 3 provided CSV files
Manual scorer — fill in any transaction details and get an instant score
Analytics dashboard — score distribution histogram, top risk transactions, decision breakdown
Export — download all scored results as CSV


Input Format
Python API (Transaction object)
pythonfrom phase1.feature_engineering.pipeline import Transaction
from datetime import datetime, timezone

txn = Transaction(
    txn_id      = "TXN_0001",          # Unique transaction ID (string)
    user_id     = "USR_0042",          # UPI registered user ID (string)
    merchant_id = "MER_099",           # Merchant identifier (string)
    device_id   = "DEV_007",           # Device fingerprint (string)
    amount      = 85000.0,             # Transaction amount in INR (float)
    timestamp   = datetime.now(timezone.utc),  # UTC datetime
    lat         = 12.9716,             # Latitude of transaction (float)
    lon         = 77.5946,             # Longitude of transaction (float)
    upi_app     = "gpay",              # One of: gpay, phonepe, paytm, bhim, amazon_pay
    txn_type    = "P2P",               # One of: P2M, P2P, RECHARGE, BILL
    is_new_merchant = False,           # bool — first time transacting with this merchant
    is_new_device   = False,           # bool — unrecognised device
)
CSV Upload (website)
Minimum required columns:
txn_id, user_id, merchant_id, amount
Full supported columns:
txn_id, user_id, merchant_id, merchant_name, device_id, amount,
upi_app, txn_type, hour, day_of_week, is_weekend, is_night,
is_new_device, is_new_merchant, velocity_1h, lat, lon, city, timestamp
Example row:
csvTXN_001,USR_042,MER_099,Amazon,DEV_007,85000,gpay,P2P,2,1,0,1,1,0,6,12.97,77.59,Bangalore,2025-03-10 02:15:00
REST API (POST /v1/score)
json{
  "txn_id":      "TXN_TEST_001",
  "user_id":     "USR_042",
  "merchant_id": "MER_099",
  "device_id":   "DEV_007",
  "amount":      85000,
  "upi_app":     "gpay",
  "txn_type":    "P2P",
  "lat":         12.97,
  "lon":         77.59
}

Output Format
Python engine (engine.process(txn))
python{
    "txn_id":           "TXN_0001",
    "fraud_score":      0.4821,        # Ensemble score (0.0 = safe, 1.0 = definite fraud)
    "transformer":      0.5100,        # Transformer model individual score
    "gnn":              0.4542,        # GNN model individual score
    "decision":         "REVIEW",      # ALLOW | REVIEW | BLOCK
    "latency_ms":       1.84,          # End-to-end inference time in milliseconds
    "block_threshold":  0.70,          # Current RL-tuned BLOCK threshold
    "review_threshold": 0.35,          # Current RL-tuned REVIEW threshold
}
REST API response
json{
  "txn_id":          "TXN_TEST_001",
  "fraud_score":     0.4821,
  "decision":        "REVIEW",
  "transformer_score": 0.51,
  "gnn_score":       0.45,
  "explanation": {
    "amount_zscore":    0.31,
    "velocity_1h":      0.12,
    "is_new_merchant":  0.15
  },
  "case_id":        "CASE-abc123",
  "latency_ms":     1.84,
  "model_version":  "v2.0"
}
Decision thresholds
DecisionScore RangeAction takenALLOW0.00 – 0.34Transaction proceeds normally, published to KafkaREVIEW0.35 – 0.69Fraud case created, assigned to analyst queueBLOCK0.70 – 1.00Transaction blocked, CRITICAL alert raised, case created

Thresholds are dynamic — the PPO RL agent adjusts them based on live false positive / recall tradeoffs.


API Endpoints
Base URL: http://127.0.0.1:8000
MethodEndpointDescriptionPOST/v1/scoreScore a single transactionPOST/v1/score/batchScore up to 100 transactions at onceGET/v1/casesList all open fraud casesPATCH/v1/cases/{case_id}Update case status (OPEN → RESOLVED etc.)GET/health/liveLiveness probe — returns 200 if service is upGET/health/readyReadiness probe — returns 200 if model is loadedGET/metricsPrometheus metrics scrape endpointGET/docsInteractive Swagger UI
Batch scoring example
jsonPOST /v1/score/batch
{
  "transactions": [
    {"txn_id": "T001", "user_id": "U001", "amount": 500, ...},
    {"txn_id": "T002", "user_id": "U002", "amount": 99000, ...}
  ]
}
Response:
json{
  "results": [...],
  "total": 2,
  "blocked": 1,
  "review": 0,
  "allowed": 1,
  "batch_latency_ms": 3.2
}

Phase Breakdown
Phase 1 — Feature Engineering & Security
File: phase1/feature_engineering/pipeline.py
Six feature extractors run in sequence on every transaction:
ExtractorFeatures producedHow it worksVelocityFeatureEngineervelocity_1h, velocity_24h, velocity_7dCounts transactions per user in rolling time windows using RedisAmountStatFeatureEngineeramount_zscore, amount_percentileZ-score of current amount vs user's historical mean/stdTemporalFeatureEngineerhour_sin, hour_cos, is_weekend, is_nightCyclical encoding so hour 23 and hour 0 are close togetherGeoFeatureEngineerdistance_from_home, geo_velocityHaversine distance; flags impossible travel speedChannelFeatureEngineerapp_risk_score, txn_type_riskRisk weights per UPI app and transaction typeFrequencyFeatureEngineeruser_txn_freq, merchant_txn_freqHow often this user/merchant appears in the system
File: phase1/security/hardening.py

PII Tokenisation — HMAC-SHA256 replaces raw user IDs before feature store ingestion
AES-128 Encryption (Fernet) — encrypts sensitive field values at rest
JWT RBAC — role-based access control for the API (analyst / admin / read-only)
Rate Limiter — token bucket algorithm: 100 requests/minute per user
Adversarial Detector — flags transactions that probe the decision boundary (e.g. repeated amounts just below the BLOCK threshold)


Phase 2 — Transformer + GNN + MLOps
File: phase2/transformer/model.py

BERT-style encoder with 4 attention layers and a CLS token
Encodes the user's last 50 transactions as a sequence
Outputs: fraud_probability (classification head) + reconstruction_error (autoencoder head)
Final score: 0.4 × fraud_prob + 0.6 × recon_error
Exports to ONNX for <10ms inference

File: phase2/gnn/model.py

Alternating SAGEConv and GATConv layers
Builds a graph: users, merchants, and devices as nodes; transactions as edges
Fraud ring detection via cosine similarity community detection
Flags accounts with high similarity scores to known fraudsters

File: phase2/mlops/tracker.py

ExperimentTracker — MLflow wrapper with JSON fallback
ModelRegistry — stage transitions: development → staging → shadow → production → deprecated
DriftDetector — PSI (Population Stability Index) detects when score distributions shift
ShadowEvaluator — runs challenger model in parallel; auto-promotes if F1 improves by >2%


Phase 3 — Microservices + Kafka
File: phase3/microservices/services.py
FastAPI service with Prometheus metrics, CORS middleware, and background task processing.
File: phase3/kafka/streaming.py
Kafka topics:
TopicPurposeupi.transactions.rawRaw incoming transactionsupi.transactions.featuresFeature-extracted transactionsupi.transactions.scoresScored resultsupi.fraud.alertsBLOCK/REVIEW decisionsupi.audit.eventsFull audit trailupi.model.feedbackConfirmed fraud labels for retraining

Phase 4 — RL Threshold Engine + Kubernetes
File: phase4/rl_engine/ppo_agent.py

State space (11 dimensions): current thresholds, recent TPR, FPR, precision, score distribution stats
Action space (9 discrete actions): nudge block/review thresholds up or down by 0.05
Reward function: 10 × TPR − 20 × FPR − 50 × (FPR > 0.005)
Hard constraint: FPR is capped at 0.5% regardless of reward — the agent cannot violate this
Training: PPO with Generalised Advantage Estimation (γ=0.99, λ=0.95, clipping ε=0.2)

File: phase4/autoscaling/k8s_and_chaos.py

Generates Kubernetes HPA manifests (CPU + memory autoscaling)
Generates KEDA manifests (Kafka consumer lag autoscaling)
CircuitBreaker — trips OPEN if failure rate exceeds 40%, recovers via HALF_OPEN state
ChaosMonkey — injects latency, errors, and timeouts for chaos engineering tests


Fallback Behaviour
The system is designed to work fully even without any external services installed:
ServiceIf unavailableFallback behaviourRedisNot installed / not runningIn-memory Python dict with same APIKafkaNot installed / not runningMock producer that logs to consoleMLflowNot runningJSON file logging in ./mlruns/PyTorchImport errorRandom policy for RL agent
You will see these messages on startup — they are not errors:
[FeatureStore] Redis unavailable (...), using in-memory fallback.
Kafka unavailable (...), using mock producer.
[MLOps] MLflow not available, using mock logging.

CSV Test Files
Three CSV files are provided for testing the website's upload feature:
upi_normal_transactions.csv — 60 rows

Everyday legitimate transactions
Small amounts: ₹199 – ₹19,999
Daytime hours (08:00 – 22:00)
Low velocity (1–3 transactions/hour)
Expected results: mostly ✅ ALLOW, very few ⚠️ REVIEW

upi_mixed_transactions.csv — 75 rows

Mix of legitimate and fraudulent transactions (~15% fraud)
Has a label column (FRAUD / LEGIT) to verify model accuracy
Fraud transactions: large amounts at night, new devices, high velocity
Expected results: mix of ✅ ALLOW, ⚠️ REVIEW, and some 🚨 BLOCK

upi_highrisk_transactions.csv — 80 rows

High fraud concentration (~25%)
Contains 2 fraud rings baked in:

Ring 1: USR_0901 – USR_0905 all transacting to MER_666 at 1–4am
Ring 2: USR_0911 – USR_0915 all transacting to MER_777


Expected results: many 🚨 BLOCK decisions, clear risk clusters visible

Upload order for best demo impact: normal → mixed → highrisk

Key Design Decisions
Why ONNX export?
ONNX removes the PyTorch runtime overhead. Inference goes from ~50ms (PyTorch) to <10ms (ONNX) in production. The model trains in PyTorch and exports once.
Why Fernet (AES-128) + PBKDF2?
Fernet provides authenticated encryption — you cannot tamper with the ciphertext without detection. PBKDF2 derives the key from a password with 100,000 iterations, making brute-force attacks impractical.
Why HMAC-SHA256 for PII tokenisation (not encryption)?
PII fields like user_id need to be consistent across requests for feature aggregation. Encryption is non-deterministic. HMAC gives the same token for the same input every time, while being irreversible.
Why focal loss?
Fraud is ~2% of all transactions (severe class imbalance). Focal loss down-weights easy negative examples so the model focuses on the hard-to-detect fraud cases.
Why a constrained MDP for RL?
An unconstrained RL agent optimising for fraud recall will eventually block everything and achieve 100% recall with 100% false positives. The hard FPR cap (0.5%) in the reward function makes it impossible for the agent to trade-off customer experience for recall.
Why point-in-time joins in the offline store?
Without point-in-time correctness, feature aggregations (e.g. "user's 7-day average amount") would accidentally use future data during training, causing label leakage and inflated offline metrics.

KPIs & Performance Targets
MetricTargetHow it's enforcedP99 Scoring Latency< 2msONNX export + Redis feature cacheFraud Recall> 95%PPO reward function shapes for recallFalse Positive Rate< 0.5%Hard constraint in RL MDPAPI Uptime99.9%Circuit breaker + K8s HPADrift Response Time< 24hPSI detector + auto-retraining triggerTransactions/day10M+KEDA Kafka-lag autoscaling

Known Warnings
These appear on every run and are safe to ignore:
[FeatureStore] Redis unavailable, using in-memory fallback.
→ Redis not installed. System uses RAM instead. Zero impact.
Kafka unavailable, using mock producer.
→ Kafka not running. Events are logged locally instead. Zero impact.
Failed to import Git (the Git executable is not on your PATH)
→ MLflow tried to record your git commit hash. Git not installed. To silence permanently:
powershell[System.Environment]::SetEnvironmentVariable("GIT_PYTHON_REFRESH", "quiet", "User")
MLflow job execution requirements not met (Windows system)
→ MLflow's job scheduling doesn't support Windows. You're not using that feature. Ignore.
UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
→ Fixed in the latest version of ppo_agent.py. If you still see it, replace with the provided fixed file.
DeprecationWarning: datetime.datetime.utcnow() is deprecated
→ Fixed in the latest version of main.py. If you still see it, replace with the provided fixed file.

Tech Stack
CategoryTechnologyLanguagePython 3.12Deep LearningPyTorch 2.1, ONNX, ONNXRuntimeGraph MLPyTorch Geometric (GraphSAGE, GAT)NLP / TransformersHuggingFace TransformersRLStable Baselines 3, GymnasiumFeature StoreRedis (online), Apache Parquet (offline)MLOpsMLflow 2.8StreamingApache Kafka, kafka-pythonAPIFastAPI, UvicornObservabilityPrometheus, OpenTelemetrySecurityCryptography (Fernet/AES-128), python-jose (JWT)Testingpytest, pytest-cov, hypothesisDataNumPy, Pandas, scikit-learn, SciPyInfrastructureKubernetes (HPA + KEDA), DockerWebsiteVanilla HTML/CSS/JS (zero dependencies)

Running Everything at Once
For the full demo stack, open 3 PowerShell windows side by side:
Window 1 — Main engine:
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield
python main.py
Window 2 — FastAPI:
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield
uvicorn phase3.microservices.services:app --port 8000 --reload
Window 3 — MLflow:
powershellcd C:\Users\latur\OneDrive\Desktop\upi-shield\phase3\microservices
python -m mlflow ui --port 5000
Then open in browser:

Website: double-click upi-shield-website.html
API docs: http://127.0.0.1:8000/docs
MLflow: http://127.0.0.1:5000


Built by [Your Name] · UPI-Shield v2.1.0 · All 4 Phases Implemented