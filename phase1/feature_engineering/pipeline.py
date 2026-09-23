"""
UPI-Shield · Phase 1
Advanced Feature Engineering Pipeline
Computes velocity, behavioural, geo, and graph-embedding features
from raw UPI transaction events.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────

@dataclass
class Transaction:
    txn_id: str
    user_id: str
    merchant_id: str
    device_id: str
    amount: float
    timestamp: datetime
    lat: float
    lon: float
    upi_app: str
    txn_type: str                    # P2P | P2M | RECHARGE
    is_new_merchant: bool = False
    is_new_device: bool = False


@dataclass
class FeatureVector:
    txn_id: str
    # velocity
    amt_1h: float = 0.0
    amt_24h: float = 0.0
    amt_7d: float = 0.0
    count_1h: int = 0
    count_24h: int = 0
    count_7d: int = 0
    # amount stats
    amount_zscore: float = 0.0
    amount_log: float = 0.0
    rolling_mean_7d: float = 0.0
    rolling_std_7d: float = 0.0
    # behavioural
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    day_sin: float = 0.0
    day_cos: float = 0.0
    is_weekend: int = 0
    is_night: int = 0
    merchant_freq_7d: int = 0
    device_freq_7d: int = 0
    # geo
    geo_velocity_kmh: float = 0.0
    distance_from_home_km: float = 0.0
    is_new_geo_region: int = 0
    # device/channel
    is_new_device: int = 0
    is_new_merchant: int = 0
    app_encoded: int = 0
    txn_type_encoded: int = 0
    # graph embeddings (filled by GNNFeatureExtractor in Phase 2)
    user_embed: list[float] = field(default_factory=lambda: [0.0] * 16)
    merchant_embed: list[float] = field(default_factory=lambda: [0.0] * 16)
    # raw (kept for model input)
    amount: float = 0.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_UPI_APPS = ["gpay", "phonepe", "paytm", "bhim", "amazon_pay", "other"]
_TXN_TYPES = ["P2P", "P2M", "RECHARGE", "BILL", "OTHER"]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _geo_token(lat: float, lon: float, precision: int = 2) -> str:
    return f"{round(lat, precision)}:{round(lon, precision)}"


# ─────────────────────────────────────────────
# Feature Engineers
# ─────────────────────────────────────────────

class VelocityFeatureEngineer:
    """
    Maintains a rolling in-memory window of transactions per user.
    In production this is replaced by a Redis-backed window (Phase 1 Feature Store).
    """

    def __init__(self) -> None:
        # user_id -> list[(timestamp, amount)]
        self._history: dict[str, list[tuple[datetime, float]]] = {}

    def update_and_compute(self, txn: Transaction) -> dict[str, Any]:
        uid = txn.user_id
        now = txn.timestamp
        hist = self._history.setdefault(uid, [])
        hist.append((now, txn.amount))

        windows = {
            "1h": now - timedelta(hours=1),
            "24h": now - timedelta(hours=24),
            "7d": now - timedelta(days=7),
        }
        feats: dict[str, Any] = {}
        for tag, cutoff in windows.items():
            window_txns = [(t, a) for t, a in hist if t >= cutoff]
            feats[f"amt_{tag}"] = sum(a for _, a in window_txns)
            feats[f"count_{tag}"] = len(window_txns)

        # prune older than 7d
        self._history[uid] = [(t, a) for t, a in hist if t >= windows["7d"]]
        return feats


class AmountStatFeatureEngineer:
    """Z-score, log-transform, rolling mean/std per user."""

    def __init__(self) -> None:
        self._user_amounts: dict[str, list[float]] = {}

    def compute(self, txn: Transaction) -> dict[str, float]:
        uid = txn.user_id
        hist = self._user_amounts.setdefault(uid, [])

        rolling = hist[-49:] if len(hist) > 49 else hist  # last 50
        mean = float(np.mean(rolling)) if rolling else txn.amount
        std = float(np.std(rolling)) if len(rolling) > 1 else 1.0
        zscore = (txn.amount - mean) / max(std, 1e-6)

        hist.append(txn.amount)
        return {
            "amount_zscore": round(zscore, 4),
            "amount_log": math.log1p(txn.amount),
            "rolling_mean_7d": round(mean, 4),
            "rolling_std_7d": round(std, 4),
        }


class TemporalFeatureEngineer:
    """Cyclical time encoding + weekend/night flags."""

    @staticmethod
    def compute(txn: Transaction) -> dict[str, Any]:
        h = txn.timestamp.hour
        dow = txn.timestamp.weekday()          # 0=Mon
        return {
            "hour_sin": math.sin(2 * math.pi * h / 24),
            "hour_cos": math.cos(2 * math.pi * h / 24),
            "day_sin": math.sin(2 * math.pi * dow / 7),
            "day_cos": math.cos(2 * math.pi * dow / 7),
            "is_weekend": int(dow >= 5),
            "is_night": int(h < 6 or h >= 23),
        }


class GeoFeatureEngineer:
    """
    Tracks last known location per user to compute geo-velocity and
    distance from the user's most frequent ('home') region.
    """

    def __init__(self) -> None:
        self._last_loc: dict[str, tuple[float, float, datetime]] = {}
        self._home_region: dict[str, str] = {}
        self._region_counts: dict[str, dict[str, int]] = {}

    def compute(self, txn: Transaction) -> dict[str, Any]:
        uid = txn.user_id
        feats: dict[str, Any] = {"geo_velocity_kmh": 0.0, "distance_from_home_km": 0.0, "is_new_geo_region": 0}

        token = _geo_token(txn.lat, txn.lon)
        region_hist = self._region_counts.setdefault(uid, {})
        region_hist[token] = region_hist.get(token, 0) + 1

        # determine home region (most visited)
        home = max(region_hist, key=lambda k: region_hist[k])
        self._home_region[uid] = home

        # distance from home
        h_lat, h_lon = (float(x) for x in home.split(":"))
        feats["distance_from_home_km"] = round(_haversine_km(txn.lat, txn.lon, h_lat, h_lon), 2)

        # geo velocity
        if uid in self._last_loc:
            last_lat, last_lon, last_ts = self._last_loc[uid]
            dt_h = max((txn.timestamp - last_ts).total_seconds() / 3600, 1e-6)
            dist = _haversine_km(txn.lat, txn.lon, last_lat, last_lon)
            feats["geo_velocity_kmh"] = round(dist / dt_h, 2)

        feats["is_new_geo_region"] = int(region_hist[token] == 1)
        self._last_loc[uid] = (txn.lat, txn.lon, txn.timestamp)
        return feats


class ChannelFeatureEngineer:
    """Encodes UPI app, txn type, device & merchant novelty."""

    @staticmethod
    def compute(txn: Transaction) -> dict[str, Any]:
        app_enc = _UPI_APPS.index(txn.upi_app) if txn.upi_app in _UPI_APPS else len(_UPI_APPS) - 1
        type_enc = _TXN_TYPES.index(txn.txn_type) if txn.txn_type in _TXN_TYPES else len(_TXN_TYPES) - 1
        return {
            "app_encoded": app_enc,
            "txn_type_encoded": type_enc,
            "is_new_device": int(txn.is_new_device),
            "is_new_merchant": int(txn.is_new_merchant),
        }


class FrequencyFeatureEngineer:
    """How often user visited a merchant/device in past 7 days."""

    def __init__(self) -> None:
        self._merchant_hist: dict[str, list[datetime]] = {}
        self._device_hist: dict[str, list[datetime]] = {}

    def compute(self, txn: Transaction) -> dict[str, int]:
        now = txn.timestamp
        cutoff = now - timedelta(days=7)

        mk = f"{txn.user_id}:{txn.merchant_id}"
        dk = f"{txn.user_id}:{txn.device_id}"

        m_hist = self._merchant_hist.setdefault(mk, [])
        d_hist = self._device_hist.setdefault(dk, [])

        m_hist.append(now)
        d_hist.append(now)

        self._merchant_hist[mk] = [t for t in m_hist if t >= cutoff]
        self._device_hist[dk] = [t for t in d_hist if t >= cutoff]

        return {
            "merchant_freq_7d": len(self._merchant_hist[mk]),
            "device_freq_7d": len(self._device_hist[dk]),
        }


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class FeaturePipeline:
    """
    Combines all sub-engineers into a single call.
    In production, stateful engineers (velocity, geo) are backed by Redis
    via the Feature Store adapter (see feature_store/store.py).
    """

    def __init__(self) -> None:
        self.velocity = VelocityFeatureEngineer()
        self.amount_stat = AmountStatFeatureEngineer()
        self.temporal = TemporalFeatureEngineer()
        self.geo = GeoFeatureEngineer()
        self.channel = ChannelFeatureEngineer()
        self.frequency = FrequencyFeatureEngineer()

    def transform(self, txn: Transaction) -> FeatureVector:
        fv = FeatureVector(txn_id=txn.txn_id, amount=txn.amount)
        fv.__dict__.update(self.velocity.update_and_compute(txn))
        fv.__dict__.update(self.amount_stat.compute(txn))
        fv.__dict__.update(self.temporal.compute(txn))
        fv.__dict__.update(self.geo.compute(txn))
        fv.__dict__.update(self.channel.compute(txn))
        fv.__dict__.update(self.frequency.compute(txn))
        return fv

    def transform_batch(self, txns: list[Transaction]) -> list[FeatureVector]:
        return [self.transform(t) for t in txns]

    def to_dataframe(self, txns: list[Transaction]) -> pd.DataFrame:
        vecs = self.transform_batch(txns)
        rows = []
        for v in vecs:
            row = {k: val for k, val in v.__dict__.items() if k not in ("user_embed", "merchant_embed")}
            for i, emb in enumerate(v.user_embed):
                row[f"user_emb_{i}"] = emb
            for i, emb in enumerate(v.merchant_embed):
                row[f"merchant_emb_{i}"] = emb
            rows.append(row)
        return pd.DataFrame(rows)
