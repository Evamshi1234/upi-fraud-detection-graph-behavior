"""
UPI-Shield · Phase 3
Kafka Streaming Pipeline
Real-time event ingestion → feature extraction → scoring → alert publishing.
Uses kafka-python for consumer/producer and asyncio for non-blocking processing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Awaitable

logger = logging.getLogger("upi-shield.kafka")

# ─────────────────────────────────────────────
# Topic Definitions
# ─────────────────────────────────────────────

TOPICS = {
    "transactions":     "upi.transactions.raw",
    "features":         "upi.transactions.features",
    "scores":           "upi.transactions.scores",
    "fraud_alerts":     "upi.fraud.alerts",
    "audit":            "upi.audit.events",
    "model_feedback":   "upi.model.feedback",     # confirmed labels for online learning
}


# ─────────────────────────────────────────────
# Message Schemas
# ─────────────────────────────────────────────

@dataclass
class TransactionEvent:
    txn_id: str
    user_id: str
    merchant_id: str
    device_id: str
    amount: float
    timestamp: str
    lat: float
    lon: float
    upi_app: str
    txn_type: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "TransactionEvent":
        data = json.loads(raw)
        return cls(**data)


@dataclass
class ScoredEvent:
    txn_id: str
    fraud_score: float
    decision: str
    transformer_score: float
    gnn_score: float
    rl_score: float
    latency_ms: float
    model_version: str
    scored_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class FraudAlert:
    alert_id: str
    txn_id: str
    user_id: str
    amount: float
    fraud_score: float
    decision: str
    severity: str
    channels: list[str]   # email | sms | push | webhook
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ─────────────────────────────────────────────
# Mock Kafka Producer/Consumer
# Falls back gracefully when Kafka is unavailable.
# ─────────────────────────────────────────────

class MockProducer:
    def __init__(self) -> None:
        self._sent: list[tuple[str, str]] = []

    def send(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        msg = value.decode() if isinstance(value, bytes) else value
        self._sent.append((topic, msg))
        logger.debug(f"[MockProducer] → {topic}: {msg[:80]}...")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def sent_messages(self) -> list[tuple[str, str]]:
        return self._sent


class MockConsumer:
    def __init__(self, topics: list[str], messages: list[str] | None = None) -> None:
        self.topics = topics
        self._messages = messages or []
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._messages):
            raise StopIteration
        msg = self._messages[self._idx]
        self._idx += 1

        class _Msg:
            def __init__(self, value):
                self.value = value.encode() if isinstance(value, str) else value
        return _Msg(msg)

    def close(self) -> None:
        pass


def create_producer(bootstrap_servers: str = "localhost:9092") -> Any:
    try:
        from kafka import KafkaProducer
        return KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
            compression_type="gzip",
            acks="all",
            retries=5,
        )
    except Exception as e:
        logger.warning(f"Kafka unavailable ({e}), using mock producer.")
        return MockProducer()


def create_consumer(topics: list[str], group_id: str, bootstrap_servers: str = "localhost:9092") -> Any:
    try:
        from kafka import KafkaConsumer
        return KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: v,
        )
    except Exception as e:
        logger.warning(f"Kafka unavailable ({e}), using mock consumer.")
        return MockConsumer(topics)


# ─────────────────────────────────────────────
# Stream Processor
# ─────────────────────────────────────────────

ScoreHandler = Callable[[TransactionEvent], Awaitable[ScoredEvent]]


class TransactionStreamProcessor:
    """
    Reads from upi.transactions.raw,
    invokes the scoring pipeline (async),
    publishes to upi.transactions.scores and upi.fraud.alerts.

    Designed to run as a Kubernetes Deployment with KEDA autoscaling
    based on Kafka consumer lag.
    """

    def __init__(
        self,
        score_handler: ScoreHandler,
        bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "upi-shield-scorer",
        alert_threshold: float = 0.5,
    ) -> None:
        self.score_handler = score_handler
        self.alert_threshold = alert_threshold
        self.consumer = create_consumer(
            [TOPICS["transactions"]], consumer_group, bootstrap_servers
        )
        self.producer = create_producer(bootstrap_servers)
        self._processed = 0
        self._errors = 0

    async def _process_message(self, raw: bytes) -> None:
        t0 = time.perf_counter()
        try:
            event = TransactionEvent.from_json(raw)
            scored = await self.score_handler(event)

            # Publish score
            self.producer.send(
                TOPICS["scores"],
                value=scored.to_json(),
                key=event.txn_id.encode(),
            )

            # Publish alert if above threshold
            if scored.fraud_score >= self.alert_threshold:
                import uuid
                severity = (
                    "CRITICAL" if scored.fraud_score >= 0.85 else
                    "HIGH"     if scored.fraud_score >= 0.70 else
                    "MEDIUM"
                )
                alert = FraudAlert(
                    alert_id=f"ALT_{uuid.uuid4().hex[:8].upper()}",
                    txn_id=event.txn_id,
                    user_id=event.user_id,
                    amount=event.amount,
                    fraud_score=scored.fraud_score,
                    decision=scored.decision,
                    severity=severity,
                    channels=["push", "email"] if severity == "CRITICAL" else ["push"],
                    created_at=datetime.utcnow().isoformat(),
                )
                self.producer.send(TOPICS["fraud_alerts"], value=alert.to_json())

            self._processed += 1
            latency = (time.perf_counter() - t0) * 1000
            if self._processed % 1000 == 0:
                logger.info(f"[Processor] Processed {self._processed} | Last latency: {latency:.1f}ms")

        except Exception as exc:
            self._errors += 1
            logger.error(f"[Processor] Error: {exc}")

    async def run(self, max_messages: int | None = None) -> None:
        """Main event loop. Run indefinitely (max_messages=None) or for testing."""
        logger.info("[Processor] Starting stream processor...")
        count = 0
        for msg in self.consumer:
            await self._process_message(msg.value)
            count += 1
            if max_messages and count >= max_messages:
                break
        self.producer.flush()
        self.consumer.close()
        logger.info(f"[Processor] Done. Processed={self._processed} Errors={self._errors}")

    @property
    def stats(self) -> dict[str, int]:
        return {"processed": self._processed, "errors": self._errors}


# ─────────────────────────────────────────────
# Feedback Consumer (for RL online learning)
# ─────────────────────────────────────────────

@dataclass
class FeedbackEvent:
    txn_id: str
    confirmed_fraud: bool
    feedback_source: str   # "manual_review" | "chargeback" | "user_report"
    received_at: str

    @classmethod
    def from_json(cls, raw: str | bytes) -> "FeedbackEvent":
        data = json.loads(raw)
        return cls(**data)


class FeedbackConsumer:
    """
    Consumes confirmed fraud labels from the feedback topic.
    Feeds into the RL agent's experience replay buffer.
    """

    def __init__(
        self,
        on_feedback: Callable[[FeedbackEvent], None],
        bootstrap_servers: str = "localhost:9092",
    ) -> None:
        self.on_feedback = on_feedback
        self.consumer = create_consumer(
            [TOPICS["model_feedback"]], "upi-shield-feedback", bootstrap_servers
        )

    def run(self, max_messages: int | None = None) -> None:
        count = 0
        for msg in self.consumer:
            try:
                event = FeedbackEvent.from_json(msg.value)
                self.on_feedback(event)
                count += 1
                if max_messages and count >= max_messages:
                    break
            except Exception as e:
                logger.error(f"[FeedbackConsumer] {e}")
        self.consumer.close()
