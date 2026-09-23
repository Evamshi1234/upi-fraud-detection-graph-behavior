"""
UPI-Shield · Phase 4
Auto-Scaling & Chaos Engineering
- Kubernetes HPA/KEDA configuration generator
- Chaos monkey for resilience testing
- Circuit breaker implementation
"""

from __future__ import annotations

import asyncio
import random
import time
import yaml
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable


# ─────────────────────────────────────────────
# Kubernetes Manifest Generators
# ─────────────────────────────────────────────

def generate_hpa(
    deployment_name: str,
    min_replicas: int = 2,
    max_replicas: int = 20,
    cpu_threshold: int = 70,
    memory_threshold: int = 80,
) -> dict:
    """Generate Kubernetes HorizontalPodAutoscaler manifest."""
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{deployment_name}-hpa", "namespace": "upi-shield"},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name,
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": cpu_threshold},
                    },
                },
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {"type": "Utilization", "averageUtilization": memory_threshold},
                    },
                },
            ],
            "behavior": {
                "scaleUp": {
                    "stabilizationWindowSeconds": 30,
                    "policies": [{"type": "Pods", "value": 4, "periodSeconds": 60}],
                },
                "scaleDown": {
                    "stabilizationWindowSeconds": 300,
                    "policies": [{"type": "Pods", "value": 2, "periodSeconds": 120}],
                },
            },
        },
    }


def generate_keda_scaler(
    deployment_name: str,
    kafka_topic: str,
    consumer_group: str,
    lag_threshold: int = 1000,
    min_replicas: int = 1,
    max_replicas: int = 30,
) -> dict:
    """Generate KEDA ScaledObject for Kafka consumer lag-based autoscaling."""
    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": f"{deployment_name}-keda", "namespace": "upi-shield"},
        "spec": {
            "scaleTargetRef": {"name": deployment_name},
            "minReplicaCount": min_replicas,
            "maxReplicaCount": max_replicas,
            "pollingInterval": 15,
            "cooldownPeriod": 60,
            "triggers": [
                {
                    "type": "kafka",
                    "metadata": {
                        "bootstrapServers": "kafka:9092",
                        "consumerGroup": consumer_group,
                        "topic": kafka_topic,
                        "lagThreshold": str(lag_threshold),
                        "offsetResetPolicy": "latest",
                    },
                }
            ],
        },
    }


def generate_deployment(
    name: str,
    image: str,
    port: int = 8000,
    cpu_request: str = "200m",
    cpu_limit: str = "1000m",
    mem_request: str = "256Mi",
    mem_limit: str = "1Gi",
    replicas: int = 2,
    env_vars: dict[str, str] | None = None,
) -> dict:
    """Generate Kubernetes Deployment manifest with resource limits."""
    env = [{"name": k, "value": v} for k, v in (env_vars or {}).items()]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": "upi-shield", "labels": {"app": name}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "containers": [
                        {
                            "name": name,
                            "image": image,
                            "ports": [{"containerPort": port}],
                            "env": env,
                            "resources": {
                                "requests": {"cpu": cpu_request, "memory": mem_request},
                                "limits":   {"cpu": cpu_limit,   "memory": mem_limit},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health/live", "port": port},
                                "initialDelaySeconds": 30, "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health/ready", "port": port},
                                "initialDelaySeconds": 10, "periodSeconds": 5,
                            },
                        }
                    ],
                    "affinity": {
                        "podAntiAffinity": {
                            "preferredDuringSchedulingIgnoredDuringExecution": [
                                {
                                    "weight": 100,
                                    "podAffinityTerm": {
                                        "labelSelector": {"matchLabels": {"app": name}},
                                        "topologyKey": "kubernetes.io/hostname",
                                    },
                                }
                            ]
                        }
                    },
                },
            },
        },
    }


def write_k8s_manifests(output_dir: str = "/tmp/upi_shield/k8s") -> None:
    """Write all K8s manifests to YAML files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    services = [
        ("scorer",    "upi-shield/scorer:latest",    8000),
        ("explainer", "upi-shield/explainer:latest", 8001),
        ("cases",     "upi-shield/cases:latest",     8002),
        ("notifier",  "upi-shield/notifier:latest",  8003),
    ]

    for name, image, port in services:
        # Deployment
        dep = generate_deployment(name, image, port)
        with open(f"{output_dir}/{name}-deployment.yaml", "w") as f:
            yaml.dump(dep, f)

        # HPA
        hpa = generate_hpa(name)
        with open(f"{output_dir}/{name}-hpa.yaml", "w") as f:
            yaml.dump(hpa, f)

    # KEDA for scorer (Kafka-driven)
    keda = generate_keda_scaler(
        "scorer", "upi.transactions.raw", "upi-shield-scorer", lag_threshold=500
    )
    with open(f"{output_dir}/scorer-keda.yaml", "w") as f:
        yaml.dump(keda, f)

    print(f"[K8s] Manifests written to {output_dir}")


# ─────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED   = "closed"     # normal operation
    OPEN     = "open"       # failing, reject calls
    HALF_OPEN = "half_open" # testing recovery


@dataclass
class CircuitBreaker:
    """
    Resilience4j-style circuit breaker for inter-service calls.
    Transitions: CLOSED → OPEN (on failure rate) → HALF_OPEN → CLOSED/OPEN.
    """
    name: str
    failure_threshold: float = 0.5    # 50% failure rate opens circuit
    min_calls: int           = 10     # min calls before evaluating
    wait_duration_s: float   = 30.0   # time in OPEN before trying HALF_OPEN
    half_open_calls: int     = 5      # test calls in HALF_OPEN

    _state: CircuitState = CircuitState.CLOSED
    _call_count: int = 0
    _failure_count: int = 0
    _last_opened: float = 0.0
    _half_open_successes: int = 0
    _half_open_failures: int = 0

    def __post_init__(self):
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_opened >= self.wait_duration_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                self._half_open_failures = 0
        return self._state

    def allow_request(self) -> bool:
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.HALF_OPEN:
            return (self._half_open_successes + self._half_open_failures) < self.half_open_calls
        return False   # OPEN

    def record_success(self) -> None:
        self._call_count += 1
        s = self.state
        if s == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_calls // 2:
                self._state = CircuitState.CLOSED
                self._call_count = 0
                self._failure_count = 0

    def record_failure(self) -> None:
        self._call_count += 1
        self._failure_count += 1
        s = self.state
        if s == CircuitState.HALF_OPEN:
            self._half_open_failures += 1
            self._state = CircuitState.OPEN
            self._last_opened = time.monotonic()
        elif s == CircuitState.CLOSED and self._call_count >= self.min_calls:
            if self._failure_count / self._call_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._last_opened = time.monotonic()

    async def call(self, func: Callable, *args, fallback: Any = None, **kwargs) -> Any:
        if not self.allow_request():
            print(f"[CircuitBreaker:{self.name}] OPEN — returning fallback")
            return fallback
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "call_count": self._call_count,
            "failure_count": self._failure_count,
            "failure_rate": round(self._failure_count / max(self._call_count, 1), 3),
        }


# ─────────────────────────────────────────────
# Chaos Engineering
# ─────────────────────────────────────────────

class ChaosMonkey:
    """
    Injects controlled failures into the system for resilience testing.
    Scenarios:
      - latency spike
      - random service error
      - Kafka partition failure
      - Redis eviction
      - model inference timeout
    """

    def __init__(self, enabled: bool = False, failure_prob: float = 0.05) -> None:
        self.enabled = enabled
        self.failure_prob = failure_prob
        self._rng = random.Random()
        self._active_experiments: list[str] = []
        self._event_log: list[dict] = []

    def maybe_inject(self, service: str) -> None:
        """Call before each service request; may raise or sleep."""
        if not self.enabled or self._rng.random() > self.failure_prob:
            return
        scenario = self._rng.choice([
            "latency_spike",
            "service_error",
            "timeout",
        ])
        self._log_event(service, scenario)
        if scenario == "latency_spike":
            delay = self._rng.uniform(0.1, 2.0)
            time.sleep(delay)
        elif scenario == "service_error":
            raise RuntimeError(f"[ChaosMonkey] Injected error in {service}")
        elif scenario == "timeout":
            time.sleep(5.0)
            raise TimeoutError(f"[ChaosMonkey] Injected timeout in {service}")

    def run_experiment(
        self,
        name: str,
        target_service: str,
        scenario: str,
        duration_s: float = 60.0,
    ) -> dict:
        """
        Run a named chaos experiment.
        In production, this triggers a GameDay runbook.
        """
        start = datetime.utcnow()
        self._active_experiments.append(name)
        result = {
            "experiment": name,
            "target": target_service,
            "scenario": scenario,
            "start": start.isoformat(),
            "duration_s": duration_s,
            "status": "completed",
        }
        self._log_event(target_service, scenario, experiment=name)
        self._active_experiments.remove(name)
        return result

    def _log_event(self, service: str, scenario: str, **extra) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "service": service,
            "scenario": scenario,
            **extra,
        }
        self._event_log.append(entry)

    @property
    def event_log(self) -> list[dict]:
        return list(self._event_log)


# ─────────────────────────────────────────────
# Load Test Helper (Locust-compatible)
# ─────────────────────────────────────────────

SAMPLE_TRANSACTIONS = [
    {"txn_id": f"LOAD_{i:06}", "user_id": f"USR_{i % 1000:04}", "merchant_id": f"MER_{i % 200:03}",
     "device_id": f"DEV_{i % 500:03}", "amount": round(random.uniform(100, 50000), 2),
     "timestamp": "2024-06-15T14:30:00Z", "lat": 12.9716, "lon": 77.5946,
     "upi_app": random.choice(["gpay","phonepe","paytm"]), "txn_type": "P2M"}
    for i in range(1000)
]
