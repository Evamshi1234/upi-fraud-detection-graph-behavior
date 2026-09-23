"""
UPI-Shield · Phase 1
Comprehensive Test Suite
Tests for feature engineering, feature store, security, and data validation.
Run: pytest phase1/tests/ -v --cov=phase1 --cov-report=term-missing
"""

from __future__ import annotations

import math
import sys
import os
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from phase1.feature_engineering.pipeline import (
    FeaturePipeline,
    Transaction,
    VelocityFeatureEngineer,
    AmountStatFeatureEngineer,
    TemporalFeatureEngineer,
    GeoFeatureEngineer,
    ChannelFeatureEngineer,
)
from phase1.security.hardening import (
    PIITokenizer,
    FieldEncryptor,
    RateLimiter,
    AdversarialDetector,
    SecurityLayer,
    has_permission,
    ROLES,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def base_txn() -> Transaction:
    return Transaction(
        txn_id="TXN001",
        user_id="USR_A",
        merchant_id="MER_001",
        device_id="DEV_X",
        amount=1500.0,
        timestamp=datetime(2024, 6, 15, 14, 30, 0),
        lat=12.9716,
        lon=77.5946,
        upi_app="gpay",
        txn_type="P2M",
    )


@pytest.fixture
def pipeline() -> FeaturePipeline:
    return FeaturePipeline()


# ─────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────

class TestVelocityFeatures:

    def test_single_transaction_velocity(self, base_txn):
        eng = VelocityFeatureEngineer()
        feats = eng.update_and_compute(base_txn)
        assert feats["amt_1h"] == pytest.approx(1500.0)
        assert feats["count_1h"] == 1
        assert feats["count_7d"] == 1

    def test_multiple_transactions_accumulate(self, base_txn):
        eng = VelocityFeatureEngineer()
        eng.update_and_compute(base_txn)
        txn2 = Transaction(**{**base_txn.__dict__, "txn_id": "TXN002", "amount": 500.0,
                               "timestamp": base_txn.timestamp + timedelta(minutes=30)})
        feats = eng.update_and_compute(txn2)
        assert feats["amt_1h"] == pytest.approx(2000.0)
        assert feats["count_1h"] == 2

    def test_window_cutoff_excludes_old_transactions(self, base_txn):
        eng = VelocityFeatureEngineer()
        eng.update_and_compute(base_txn)
        # transaction 2h later should not see base_txn in 1h window
        txn_later = Transaction(**{**base_txn.__dict__, "txn_id": "TXN003", "amount": 200.0,
                                    "timestamp": base_txn.timestamp + timedelta(hours=2)})
        feats = eng.update_and_compute(txn_later)
        assert feats["amt_1h"] == pytest.approx(200.0)
        assert feats["count_1h"] == 1

    def test_7day_window_accumulates_across_days(self, base_txn):
        eng = VelocityFeatureEngineer()
        for i in range(5):
            t = Transaction(**{**base_txn.__dict__,
                               "txn_id": f"TXN{i:03}", "amount": 100.0,
                               "timestamp": base_txn.timestamp + timedelta(days=i)})
            feats = eng.update_and_compute(t)
        assert feats["count_7d"] == 5
        assert feats["amt_7d"] == pytest.approx(500.0)


class TestAmountStatFeatures:

    def test_zscore_first_transaction(self, base_txn):
        eng = AmountStatFeatureEngineer()
        feats = eng.compute(base_txn)
        assert feats["amount_zscore"] == pytest.approx(0.0, abs=1e-3)

    def test_log_transform(self, base_txn):
        eng = AmountStatFeatureEngineer()
        feats = eng.compute(base_txn)
        assert feats["amount_log"] == pytest.approx(math.log1p(1500.0), rel=1e-4)

    def test_zscore_outlier(self, base_txn):
        eng = AmountStatFeatureEngineer()
        # Build history of small amounts
        for i in range(10):
            t = Transaction(**{**base_txn.__dict__, "txn_id": f"TXNS{i}", "amount": 100.0,
                               "timestamp": base_txn.timestamp + timedelta(minutes=i)})
            eng.compute(t)
        # Now send a large outlier
        big = Transaction(**{**base_txn.__dict__, "txn_id": "TXNBIG", "amount": 50000.0,
                              "timestamp": base_txn.timestamp + timedelta(minutes=11)})
        feats = eng.compute(big)
        assert feats["amount_zscore"] > 5.0   # Should be a clear outlier


class TestTemporalFeatures:

    def test_cyclical_encoding_range(self, base_txn):
        eng = TemporalFeatureEngineer()
        feats = eng.compute(base_txn)
        assert -1.0 <= feats["hour_sin"] <= 1.0
        assert -1.0 <= feats["hour_cos"] <= 1.0
        assert -1.0 <= feats["day_sin"] <= 1.0
        assert -1.0 <= feats["day_cos"] <= 1.0

    def test_weekend_flag(self, base_txn):
        eng = TemporalFeatureEngineer()
        # Saturday
        sat_txn = Transaction(**{**base_txn.__dict__, "timestamp": datetime(2024, 6, 15, 14, 0)})  # Saturday
        feats = eng.compute(sat_txn)
        assert feats["is_weekend"] == 1

    def test_night_flag(self, base_txn):
        eng = TemporalFeatureEngineer()
        night_txn = Transaction(**{**base_txn.__dict__, "timestamp": datetime(2024, 6, 15, 2, 0)})
        feats = eng.compute(night_txn)
        assert feats["is_night"] == 1

    def test_day_flag(self, base_txn):
        eng = TemporalFeatureEngineer()
        day_txn = Transaction(**{**base_txn.__dict__, "timestamp": datetime(2024, 6, 15, 12, 0)})
        feats = eng.compute(day_txn)
        assert feats["is_night"] == 0


class TestGeoFeatures:

    def test_first_transaction_no_velocity(self, base_txn):
        eng = GeoFeatureEngineer()
        feats = eng.compute(base_txn)
        assert feats["geo_velocity_kmh"] == 0.0

    def test_geo_velocity_computed(self, base_txn):
        eng = GeoFeatureEngineer()
        eng.compute(base_txn)
        # Move ~800km away 1 hour later (Delhi from Bangalore)
        txn2 = Transaction(**{**base_txn.__dict__, "txn_id": "TXN_DEL",
                               "lat": 28.6139, "lon": 77.2090,
                               "timestamp": base_txn.timestamp + timedelta(hours=1)})
        feats = eng.compute(txn2)
        assert feats["geo_velocity_kmh"] > 100   # implausibly fast

    def test_new_geo_region_flag(self, base_txn):
        eng = GeoFeatureEngineer()
        feats = eng.compute(base_txn)
        assert feats["is_new_geo_region"] == 1  # First visit

    def test_known_geo_region_not_flagged(self, base_txn):
        eng = GeoFeatureEngineer()
        eng.compute(base_txn)
        txn2 = Transaction(**{**base_txn.__dict__, "txn_id": "TXN2",
                               "timestamp": base_txn.timestamp + timedelta(hours=1)})
        feats = eng.compute(txn2)
        assert feats["is_new_geo_region"] == 0


class TestFullPipeline:

    def test_transform_returns_feature_vector(self, pipeline, base_txn):
        fv = pipeline.transform(base_txn)
        assert fv.txn_id == "TXN001"
        assert fv.amount == 1500.0
        assert len(fv.user_embed) == 16

    def test_to_dataframe_shape(self, pipeline, base_txn):
        txns = [
            Transaction(**{**base_txn.__dict__, "txn_id": f"T{i}",
                           "timestamp": base_txn.timestamp + timedelta(minutes=i * 10)})
            for i in range(10)
        ]
        df = pipeline.to_dataframe(txns)
        assert len(df) == 10
        assert "amount_zscore" in df.columns
        assert "user_emb_0" in df.columns


# ─────────────────────────────────────────────
# Security Tests
# ─────────────────────────────────────────────

class TestPIITokenizer:

    def test_deterministic_tokenization(self):
        tok = PIITokenizer(secret_key="test-key")
        assert tok.tokenize("9876543210") == tok.tokenize("9876543210")

    def test_different_values_give_different_tokens(self):
        tok = PIITokenizer(secret_key="test-key")
        assert tok.tokenize("9876543210") != tok.tokenize("9876543211")

    def test_token_prefix(self):
        tok = PIITokenizer(secret_key="test-key")
        assert tok.tokenize("abc").startswith("tok_")

    def test_scrub_phone(self):
        tok = PIITokenizer()
        result = tok.scrub_string("call 9876543210 now")
        assert "9876543210" not in result
        assert "[PHONE]" in result

    def test_scrub_upi_id(self):
        tok = PIITokenizer()
        result = tok.scrub_string("send to john@paytm please")
        assert "john@paytm" not in result

    def test_batch_tokenize_length(self):
        tok = PIITokenizer()
        tokens = tok.tokenize_batch(["a", "b", "c"])
        assert len(tokens) == 3


class TestFieldEncryptor:

    def test_encrypt_decrypt_roundtrip(self):
        enc = FieldEncryptor(passphrase="test-pass")
        original = "sensitive_device_id_12345"
        encrypted = enc.encrypt(original)
        assert encrypted != original
        assert enc.decrypt(encrypted) == original

    def test_encrypt_dict_fields(self):
        enc = FieldEncryptor(passphrase="test-pass")
        data = {"device_id": "DEV123", "amount": 500}
        result = enc.encrypt_dict_fields(data, ["device_id"])
        assert result["device_id"] != "DEV123"
        assert result["amount"] == 500  # unchanged


class TestRateLimiter:

    def test_allows_within_burst(self):
        rl = RateLimiter(rate_per_sec=10, burst=5)
        for _ in range(5):
            assert rl.allow("client_1") is True

    def test_blocks_after_burst_exhausted(self):
        rl = RateLimiter(rate_per_sec=1, burst=3)
        for _ in range(3):
            rl.allow("client_x")
        assert rl.allow("client_x") is False

    def test_different_clients_independent(self):
        rl = RateLimiter(rate_per_sec=1, burst=2)
        for _ in range(2):
            rl.allow("c1")
        rl.allow("c1")  # exhausted
        assert rl.allow("c2") is True   # c2 unaffected


class TestAdversarialDetector:

    def test_boundary_amount_flagged(self):
        det = AdversarialDetector()
        result = det.inspect("user1", 9999.99, datetime.now())
        assert "boundary_amount" in result["adversarial_flags"]

    def test_repeated_amount_probe(self):
        det = AdversarialDetector()
        ts = datetime.now()
        for _ in range(3):
            det.inspect("user1", 500.0, ts)
        result = det.inspect("user1", 500.0, ts)
        assert "repeated_amount_probe" in result["adversarial_flags"]

    def test_normal_transaction_no_flags(self):
        det = AdversarialDetector()
        result = det.inspect("user1", 347.50, datetime.now())
        assert result["adversarial_flags"] == []
        assert result["is_suspicious"] is False


class TestRBAC:

    def test_admin_has_all_permissions(self):
        assert has_permission("admin", "write:models") is True
        assert has_permission("admin", "anything") is True

    def test_analyst_limited_permissions(self):
        assert has_permission("analyst", "read:scores") is True
        assert has_permission("analyst", "write:models") is False

    def test_unknown_role_no_permissions(self):
        assert has_permission("unknown_role", "read:scores") is False


class TestSecurityLayer:

    def test_sanitize_transaction(self):
        sl = SecurityLayer()
        txn = {"user_id": "9876543210", "device_id": "DEV123", "amount": 1000}
        result = sl.sanitize_transaction(txn)
        assert result["user_id"].startswith("tok_")
        assert result["device_id"] != "DEV123"

    def test_check_request_passes(self):
        sl = SecurityLayer()
        result = sl.check_request("app_client", "user1", 500.0, datetime.now())
        assert result["rate_limit_passed"] is True

    def test_check_request_adversarial(self):
        sl = SecurityLayer()
        for _ in range(4):
            sl.check_request("app_client", "user2", 9999.99, datetime.now())
        result = sl.check_request("app_client", "user2", 9999.99, datetime.now())
        assert result["is_suspicious"] is True
