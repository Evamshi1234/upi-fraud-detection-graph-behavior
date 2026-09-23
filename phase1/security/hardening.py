"""
UPI-Shield · Phase 1
Security Hardening Layer
- PII tokenization & encryption
- JWT auth with RBAC
- Adversarial input detection
- Rate limiting
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


# ─────────────────────────────────────────────
# PII Tokenization
# ─────────────────────────────────────────────

class PIITokenizer:
    """
    Deterministically tokenizes PII fields (phone, UPI ID, account number)
    using HMAC-SHA256 so the same input always maps to the same token
    (needed for graph joins) while being non-reversible without the secret key.
    """

    def __init__(self, secret_key: str | None = None) -> None:
        self._key = (secret_key or os.environ.get("PII_SECRET_KEY", "change-me-in-production")).encode()

    def tokenize(self, value: str) -> str:
        h = hmac.new(self._key, value.encode(), hashlib.sha256)
        return "tok_" + h.hexdigest()[:24]

    def tokenize_batch(self, values: list[str]) -> list[str]:
        return [self.tokenize(v) for v in values]

    # regex-based scrubber for log lines
    _PHONE_RE = re.compile(r"\b(\+91|0)?[6-9]\d{9}\b")
    _UPI_RE   = re.compile(r"\b[\w.\-]+@[a-zA-Z]{3,}\b")
    _ACCT_RE  = re.compile(r"\b\d{9,18}\b")

    def scrub_string(self, text: str) -> str:
        text = self._PHONE_RE.sub("[PHONE]", text)
        text = self._UPI_RE.sub("[UPI_ID]", text)
        text = self._ACCT_RE.sub("[ACCT]", text)
        return text


# ─────────────────────────────────────────────
# Encryption at Rest
# ─────────────────────────────────────────────

class FieldEncryptor:
    """
    AES-128 (via Fernet) for encrypting sensitive fields before
    writing to the feature store or database.
    """

    def __init__(self, passphrase: str | None = None) -> None:
        raw = (passphrase or os.environ.get("ENCRYPT_PASSPHRASE", "dev-passphrase-change-me")).encode()
        salt = b"upi_shield_salt_"  # in prod: store in Vault, rotate periodically
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
        key = base64.urlsafe_b64encode(kdf.derive(raw))
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self._fernet.decrypt(token.encode()).decode()

    def encrypt_dict_fields(self, data: dict, fields: list[str]) -> dict:
        out = dict(data)
        for f in fields:
            if f in out and out[f] is not None:
                out[f] = self.encrypt(str(out[f]))
        return out


# ─────────────────────────────────────────────
# JWT Authentication & RBAC
# ─────────────────────────────────────────────

_SECRET_KEY = os.environ.get("JWT_SECRET", "super-secret-jwt-key-change-me")
_ALGORITHM  = "HS256"

ROLES = {
    "analyst":      ["read:scores", "read:dashboard"],
    "ops":          ["read:scores", "read:dashboard", "write:rules", "read:cases"],
    "model_admin":  ["read:scores", "write:models", "read:cases", "read:audit"],
    "admin":        ["*"],
}


def create_access_token(subject: str, role: str, expires_minutes: int = 60) -> str:
    if not JWT_AVAILABLE:
        return "jwt-unavailable"
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "role": role, "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    if not JWT_AVAILABLE:
        return {"sub": "unknown", "role": "analyst"}
    return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])


def has_permission(role: str, permission: str) -> bool:
    perms = ROLES.get(role, [])
    return "*" in perms or permission in perms


def require_permission(permission: str):
    """FastAPI-compatible dependency decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, token: str = "", **kwargs):
            try:
                claims = decode_token(token)
                if not has_permission(claims.get("role", ""), permission):
                    raise PermissionError(f"Role '{claims.get('role')}' lacks '{permission}'")
            except Exception as e:
                raise PermissionError(str(e))
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# Rate Limiter (Token Bucket)
# ─────────────────────────────────────────────

@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """
    Token-bucket rate limiter keyed by (client_id, endpoint).
    In production this state lives in Redis with atomic Lua scripts.
    """

    def __init__(self, rate_per_sec: float = 100, burst: float = 200) -> None:
        self.rate = rate_per_sec
        self.burst = burst
        self._buckets: dict[str, _Bucket] = {}

    def _refill(self, bucket: _Bucket) -> _Bucket:
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
        bucket.last_refill = now
        return bucket

    def allow(self, client_id: str, cost: float = 1.0) -> bool:
        key = client_id
        if key not in self._buckets:
            self._buckets[key] = _Bucket(tokens=self.burst, last_refill=time.monotonic())
        bucket = self._refill(self._buckets[key])
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True
        return False


# ─────────────────────────────────────────────
# Adversarial Input Detection
# ─────────────────────────────────────────────

class AdversarialDetector:
    """
    Detects suspicious patterns in transaction requests that may indicate
    model probing or adversarial manipulation.
    Rules are intentionally lightweight for Phase 1; Phase 4 adds RL-based detection.
    """

    # amounts that probe decision boundaries
    _BOUNDARY_AMOUNTS = {999.0, 999.99, 9999.0, 9999.99, 99999.0, 99999.99}
    _ROUND_THRESHOLD = 0.01   # suspiciously round

    def __init__(self) -> None:
        self._probe_counts: dict[str, list[float]] = defaultdict(list)

    def inspect(self, user_id: str, amount: float, timestamp: datetime) -> dict[str, Any]:
        flags: list[str] = []

        # boundary probing
        if amount in self._BOUNDARY_AMOUNTS:
            flags.append("boundary_amount")

        # sequential identical amounts (>3 in 10 min)
        hist = self._probe_counts[user_id]
        hist.append(amount)
        if len(hist) > 20:
            self._probe_counts[user_id] = hist[-20:]
        if hist.count(amount) >= 3:
            flags.append("repeated_amount_probe")

        # extreme round numbers above threshold
        if amount > 10_000 and amount % 1000 < self._ROUND_THRESHOLD:
            flags.append("suspicious_round_amount")

        return {
            "adversarial_flags": flags,
            "adversarial_score": len(flags) / 3.0,   # normalised 0-1
            "is_suspicious": len(flags) > 0,
        }


# ─────────────────────────────────────────────
# Security Facade
# ─────────────────────────────────────────────

class SecurityLayer:
    """Single entry-point for all security operations."""

    def __init__(self) -> None:
        self.tokenizer   = PIITokenizer()
        self.encryptor   = FieldEncryptor()
        self.rate_limiter = RateLimiter()
        self.adversarial = AdversarialDetector()

    def sanitize_transaction(self, txn_dict: dict[str, Any]) -> dict[str, Any]:
        """Tokenize PII and encrypt sensitive fields before storage."""
        pii_fields = ["user_id", "phone", "upi_id", "account_number"]
        for f in pii_fields:
            if f in txn_dict:
                txn_dict[f] = self.tokenizer.tokenize(str(txn_dict[f]))
        return self.encryptor.encrypt_dict_fields(txn_dict, ["device_id"])

    def check_request(self, client_id: str, user_id: str, amount: float, timestamp: datetime) -> dict:
        allowed = self.rate_limiter.allow(client_id)
        adversarial = self.adversarial.inspect(user_id, amount, timestamp)
        return {"rate_limit_passed": allowed, **adversarial}
