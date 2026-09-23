"""
UPI-Shield · Phase 1
Feature Store — Redis-backed online store + Parquet offline store.
Wraps Feast-style API for online/offline feature retrieval.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# ─────────────────────────────────────────────
# Online Store (Redis)
# ─────────────────────────────────────────────

class OnlineFeatureStore:
    """
    Stores and retrieves pre-computed feature vectors in Redis.
    TTL mirrors the longest feature window (7 days).
    Falls back to an in-memory dict when Redis is unavailable (testing).
    """

    TTL_SECONDS = 7 * 24 * 3600   # 7 days

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        self._redis: redis.Redis | None = None
        self._fallback: dict[str, str] = {}
        if REDIS_AVAILABLE:
            try:
                self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                self._redis.ping()
                print(f"[FeatureStore] Connected to Redis at {host}:{port}")
            except Exception as exc:
                print(f"[FeatureStore] Redis unavailable ({exc}), using in-memory fallback.")
                self._redis = None

    # ── write ────────────────────────────────
    def put(self, entity_key: str, features: dict[str, Any]) -> None:
        payload = json.dumps(features, default=str)
        if self._redis:
            self._redis.setex(f"fs:{entity_key}", self.TTL_SECONDS, payload)
        else:
            self._fallback[f"fs:{entity_key}"] = payload

    def put_user_window(self, user_id: str, window_tag: str, data: dict[str, Any]) -> None:
        """Store velocity window data separately for partial updates."""
        key = f"fs:velocity:{user_id}:{window_tag}"
        payload = json.dumps({"ts": time.time(), "data": data})
        if self._redis:
            self._redis.setex(key, self.TTL_SECONDS, payload)
        else:
            self._fallback[key] = payload

    # ── read ─────────────────────────────────
    def get(self, entity_key: str) -> dict[str, Any] | None:
        raw = (
            self._redis.get(f"fs:{entity_key}")
            if self._redis
            else self._fallback.get(f"fs:{entity_key}")
        )
        return json.loads(raw) if raw else None

    def get_batch(self, keys: list[str]) -> dict[str, dict[str, Any] | None]:
        return {k: self.get(k) for k in keys}

    # ── user profile ─────────────────────────
    def update_user_profile(self, user_id: str, amount: float, timestamp: datetime) -> None:
        """
        Append a transaction to the user's rolling profile stored in Redis list.
        Each entry: JSON {"ts": epoch, "amt": amount}
        """
        key = f"fs:profile:{user_id}"
        entry = json.dumps({"ts": timestamp.timestamp(), "amt": amount})
        if self._redis:
            self._redis.lpush(key, entry)
            self._redis.ltrim(key, 0, 999)   # keep last 1000 txns
            self._redis.expire(key, self.TTL_SECONDS)
        else:
            lst = json.loads(self._fallback.get(key, "[]"))
            lst.insert(0, json.loads(entry))
            self._fallback[key] = json.dumps(lst[:1000])

    def get_user_profile(self, user_id: str) -> list[dict[str, Any]]:
        key = f"fs:profile:{user_id}"
        if self._redis:
            raw_list = self._redis.lrange(key, 0, -1)
            return [json.loads(x) for x in raw_list]
        raw = self._fallback.get(key, "[]")
        return json.loads(raw)


# ─────────────────────────────────────────────
# Offline Store (Parquet / S3)
# ─────────────────────────────────────────────

class OfflineFeatureStore:
    """
    Writes time-stamped feature vectors to partitioned Parquet files.
    In production these are written to S3/GCS and queried via Spark or BigQuery.
    """

    def __init__(self, base_path: str = "/tmp/upi_shield/features") -> None:
        import os
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def write(self, df: pd.DataFrame, partition_date: str | None = None) -> str:
        date_str = partition_date or datetime.utcnow().strftime("%Y-%m-%d")
        path = f"{self.base_path}/date={date_str}/features.parquet"
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[OfflineStore] Wrote {len(df)} rows → {path}")
        return path

    def read_range(self, start: str, end: str) -> pd.DataFrame:
        """Read all parquet partitions between two date strings (YYYY-MM-DD)."""
        import glob
        files = glob.glob(f"{self.base_path}/date=*/features.parquet")
        filtered = [
            f for f in files
            if start <= f.split("date=")[1].split("/")[0] <= end
        ]
        if not filtered:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in filtered], ignore_index=True)

    def point_in_time_join(
        self,
        entity_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Correct historical join: for each entity row, fetch the latest
        feature vector whose timestamp ≤ entity timestamp.
        Prevents label leakage in training data.
        """
        entity_df = entity_df.sort_values(timestamp_col)
        feature_df = feature_df.sort_values(timestamp_col)
        return pd.merge_asof(
            entity_df,
            feature_df,
            on=timestamp_col,
            by="user_id",
            direction="backward",
            suffixes=("", "_feat"),
        )


# ─────────────────────────────────────────────
# Unified Feature Store Facade
# ─────────────────────────────────────────────

class FeatureStore:
    """Single entry-point used by inference and training pipelines."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        offline_path: str = "/tmp/upi_shield/features",
    ) -> None:
        self.online = OnlineFeatureStore(redis_host, redis_port)
        self.offline = OfflineFeatureStore(offline_path)

    def ingest_transaction(
        self, user_id: str, txn_id: str, amount: float, timestamp: datetime, features: dict[str, Any]
    ) -> None:
        """Write features to online store and update user profile."""
        self.online.put(txn_id, features)
        self.online.update_user_profile(user_id, amount, timestamp)

    def get_online_features(self, txn_id: str) -> dict[str, Any] | None:
        return self.online.get(txn_id)

    def get_training_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.offline.read_range(start_date, end_date)

    def flush_to_offline(self, df: pd.DataFrame) -> None:
        self.offline.write(df)
