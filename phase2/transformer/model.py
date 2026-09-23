"""
UPI-Shield · Phase 2
Transformer-Based Sequential Anomaly Detection
BERT-style encoder over the last N transactions per user.
Outputs an anomaly score (reconstruction error + classification head).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class TransformerConfig:
    feature_dim: int   = 64      # input feature vector size
    d_model: int       = 128     # transformer hidden dim
    nhead: int         = 4       # attention heads
    num_layers: int    = 4       # encoder layers
    dim_feedforward: int = 256
    dropout: float     = 0.1
    max_seq_len: int   = 50      # max transaction history length
    num_classes: int   = 2       # fraud / legit


# ─────────────────────────────────────────────
# Positional Encoding with Temporal Gaps
# ─────────────────────────────────────────────

class TemporalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal PE augmented with a learnable temporal gap embedding.
    Gap = seconds since previous transaction, log-scaled.
    This lets the model distinguish rapid-fire probe transactions.
    """

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal base
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

        # Learnable gap MLP
        self.gap_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, x: Tensor, time_gaps: Tensor | None = None) -> Tensor:
        """
        x:         (B, L, D)
        time_gaps: (B, L) log-seconds since prev txn; 0 for first token
        """
        seq_len = x.size(1)
        out = x + self.pe[:, :seq_len, :]
        if time_gaps is not None:
            gap_emb = self.gap_proj(time_gaps.unsqueeze(-1).float())   # (B, L, D)
            out = out + gap_emb
        return self.dropout(out)


# ─────────────────────────────────────────────
# Transformer Encoder
# ─────────────────────────────────────────────

class TransactionEncoder(nn.Module):
    """
    Projects raw feature vectors → d_model, then passes through
    a standard Transformer encoder.  The CLS-token representation
    is used for both classification and reconstruction scoring.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model))

        # Temporal PE
        self.pos_enc = TemporalPositionalEncoding(cfg.d_model, cfg.max_seq_len + 1, cfg.dropout)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        # Reconstruction head (auto-encoder objective for anomaly scoring)
        self.recon_head = nn.Linear(cfg.d_model, cfg.feature_dim)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

    def forward(
        self,
        x: Tensor,
        time_gaps: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        x:      (B, L, feature_dim)  — sequence of feature vectors
        Returns dict with:
            logits        (B, 2)       fraud classification
            anomaly_score (B,)         reconstruction error per sample
            embeddings    (B, d_model) CLS representation
        """
        B, L, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)                    # (B, L, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x_full = torch.cat([cls, x_proj], dim=1)       # (B, L+1, D)

        # Add temporal PE
        gaps_full = None
        if time_gaps is not None:
            cls_gap = torch.zeros(B, 1, device=time_gaps.device)
            gaps_full = torch.cat([cls_gap, time_gaps], dim=1)
        x_full = self.pos_enc(x_full, gaps_full)

        # Extend padding mask for CLS
        mask = None
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=src_key_padding_mask.device)
            mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # Transformer encoding
        enc = self.encoder(x_full, src_key_padding_mask=mask)   # (B, L+1, D)

        cls_repr = enc[:, 0, :]          # (B, D)
        seq_repr = enc[:, 1:, :]         # (B, L, D)

        # Outputs
        logits = self.cls_head(cls_repr)
        recon  = self.recon_head(seq_repr)                      # (B, L, feature_dim)
        anomaly_score = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))

        return {
            "logits": logits,
            "anomaly_score": anomaly_score,
            "embeddings": cls_repr,
            "reconstruction": recon,
        }

    @torch.no_grad()
    def score(self, x: Tensor, time_gaps: Tensor | None = None) -> Tensor:
        """
        Combined anomaly score:
            0.4 * fraud_prob  +  0.6 * normalised_recon_error
        Returns (B,) tensor in [0, 1].
        """
        self.eval()
        out = self.forward(x, time_gaps)
        fraud_prob   = torch.softmax(out["logits"], dim=-1)[:, 1]    # P(fraud)
        recon_norm   = torch.sigmoid(out["anomaly_score"] - 1.0)     # centre around 1.0 MSE
        return 0.4 * fraud_prob + 0.6 * recon_norm


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

class TransformerTrainer:
    """
    Supervised + self-supervised training.
    Loss = CrossEntropy (labelled) + MSE reconstruction (all samples).
    Handles class imbalance via focal loss weighting.
    """

    def __init__(
        self,
        model: TransactionEncoder,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        recon_weight: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.recon_weight = recon_weight
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)

    def _focal_loss(self, logits: Tensor, labels: Tensor, gamma: float = 2.0) -> Tensor:
        """Focal loss to handle heavy class imbalance (fraud <<< legit)."""
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()

    def train_step(
        self,
        x: Tensor,
        labels: Tensor,
        time_gaps: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> dict[str, float]:
        self.model.train()
        x, labels = x.to(self.device), labels.to(self.device)
        if time_gaps is not None:
            time_gaps = time_gaps.to(self.device)

        out = self.model(x, time_gaps, mask)

        cls_loss   = self._focal_loss(out["logits"], labels)
        recon_loss = F.mse_loss(out["reconstruction"], x)
        loss = cls_loss + self.recon_weight * recon_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        with torch.no_grad():
            preds = out["logits"].argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        return {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "recon_loss": recon_loss.item(),
            "accuracy": acc,
        }

    @torch.no_grad()
    def evaluate(self, x: Tensor, labels: Tensor) -> dict[str, float]:
        self.model.eval()
        x, labels = x.to(self.device), labels.to(self.device)
        out = self.model(x)
        preds = out["logits"].argmax(dim=-1)
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    def save(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict(), "config": self.model.cfg}, path)
        print(f"[Transformer] Saved → {path}")

    @staticmethod
    def load(path: str, device: str = "cpu") -> TransactionEncoder:
        ckpt = torch.load(path, map_location=device)
        model = TransactionEncoder(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        return model.to(device)


# ─────────────────────────────────────────────
# ONNX Export for Low-Latency Inference
# ─────────────────────────────────────────────

def export_to_onnx(model: TransactionEncoder, path: str = "transformer.onnx") -> None:
    """Export model to ONNX for sub-10ms serving via ONNXRuntime."""
    model.eval()
    cfg = model.cfg
    dummy_x    = torch.randn(1, cfg.max_seq_len, cfg.feature_dim)
    dummy_gaps = torch.zeros(1, cfg.max_seq_len)
    torch.onnx.export(
        model,
        (dummy_x, dummy_gaps),
        path,
        input_names=["features", "time_gaps"],
        output_names=["logits", "anomaly_score", "embeddings", "reconstruction"],
        dynamic_axes={
            "features":    {0: "batch", 1: "seq_len"},
            "time_gaps":   {0: "batch", 1: "seq_len"},
            "logits":      {0: "batch"},
            "embeddings":  {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[ONNX] Exported → {path}")
