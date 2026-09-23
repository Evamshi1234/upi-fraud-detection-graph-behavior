"""
UPI-Shield · Phase 2
Graph Neural Network — Fraud Ring & Coordinated Attack Detection
Uses GraphSAGE / GAT on a heterogeneous user–merchant–device graph.
Detects fraud rings via node embedding similarity and community detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class GNNConfig:
    node_feat_dim: int    = 32     # raw node feature size
    edge_feat_dim: int    = 8      # edge feature size (amount_norm, time_diff, etc.)
    hidden_dim: int       = 64
    embed_dim: int        = 32     # final node embedding size
    num_layers: int       = 3
    dropout: float        = 0.2
    num_heads: int        = 4      # for GAT layers
    use_edge_features: bool = True


# ─────────────────────────────────────────────
# In-Memory Graph (no torch_geometric dependency for portability)
# ─────────────────────────────────────────────

@dataclass
class GraphData:
    """
    Lightweight adjacency representation.
    In production, backed by a property graph DB (Neo4j/TigerGraph)
    with streaming edge updates via Kafka.
    """
    node_features: dict[str, Tensor]      # node_id -> feature tensor
    edge_index: Tensor                    # (2, E) — [src; dst] node indices
    edge_features: Tensor                 # (E, edge_feat_dim)
    node_id_map: dict[str, int]           # node_id str -> int index
    node_types: dict[int, str]            # node_idx -> "user"|"merchant"|"device"
    labels: dict[int, int] = field(default_factory=dict)  # node_idx -> fraud label (optional)


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

class TransactionGraphBuilder:
    """
    Maintains the live transaction graph.
    Nodes: users, merchants, devices.
    Edges: transactions (with amount, timestamp features).
    Supports incremental updates for streaming scenarios.
    """

    def __init__(self, node_feat_dim: int = 32, edge_feat_dim: int = 8) -> None:
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self._node_id_map: dict[str, int] = {}
        self._node_types: dict[int, str] = {}
        self._node_feats: dict[int, list[float]] = {}
        self._edges: list[tuple[int, int, list[float]]] = []
        self._labels: dict[int, int] = {}

    def _get_or_add_node(self, nid: str, ntype: str, features: list[float] | None = None) -> int:
        if nid not in self._node_id_map:
            idx = len(self._node_id_map)
            self._node_id_map[nid] = idx
            self._node_types[idx] = ntype
            self._node_feats[idx] = features or [0.0] * self.node_feat_dim
        return self._node_id_map[nid]

    def add_transaction(
        self,
        user_id: str,
        merchant_id: str,
        device_id: str,
        amount_norm: float,
        time_diff: float,
        is_fraud: int | None = None,
        user_feats: list[float] | None = None,
        merchant_feats: list[float] | None = None,
    ) -> None:
        u_idx = self._get_or_add_node(user_id, "user", user_feats)
        m_idx = self._get_or_add_node(merchant_id, "merchant", merchant_feats)
        d_idx = self._get_or_add_node(device_id, "device")

        # Edge features: [amount_norm, time_diff, 0-pad...]
        ef = [amount_norm, time_diff] + [0.0] * (self.edge_feat_dim - 2)

        self._edges.extend([
            (u_idx, m_idx, ef),
            (u_idx, d_idx, ef),
            (d_idx, m_idx, ef),
        ])

        if is_fraud is not None:
            self._labels[u_idx] = is_fraud

    def build(self) -> GraphData:
        if not self._edges:
            raise ValueError("No edges in graph. Call add_transaction first.")

        n = len(self._node_id_map)

        # Node feature matrix
        feat_matrix = torch.zeros(n, self.node_feat_dim)
        for idx, feats in self._node_feats.items():
            feat_matrix[idx] = torch.tensor(feats[:self.node_feat_dim] +
                                             [0.0] * max(0, self.node_feat_dim - len(feats)))

        # Edge tensors
        src  = torch.tensor([e[0] for e in self._edges], dtype=torch.long)
        dst  = torch.tensor([e[1] for e in self._edges], dtype=torch.long)
        edge_feats = torch.tensor([e[2] for e in self._edges], dtype=torch.float)

        # Add reverse edges (undirected)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ])
        edge_feats = torch.cat([edge_feats, edge_feats])

        return GraphData(
            node_features={"all": feat_matrix},
            edge_index=edge_index,
            edge_features=edge_feats,
            node_id_map=self._node_id_map,
            node_types=self._node_types,
            labels=self._labels,
        )


# ─────────────────────────────────────────────
# GraphSAGE Layer (manual, no PyG dependency)
# ─────────────────────────────────────────────

class SAGEConv(nn.Module):
    """
    GraphSAGE aggregation: h_v = MLP(concat(h_v, mean(h_neighbours))).
    Efficient for inductive learning on unseen nodes.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_feat_dim: int = 0) -> None:
        super().__init__()
        concat_dim = in_dim * 2 + edge_feat_dim
        self.lin = nn.Linear(concat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """x: (N, D), edge_index: (2, E)"""
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbor features (mean pooling)
        agg = torch.zeros_like(x)
        count = torch.zeros(N, 1, device=x.device)
        msg = x[src]
        if edge_attr is not None:
            # Simple linear projection of edge features into node space
            msg = msg + edge_attr[:, :msg.size(-1)]   # additive edge contribution
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, device=x.device))
        agg = agg / (count.clamp(min=1))

        # Concat self + aggregated
        concat = torch.cat([x, agg], dim=-1)
        if edge_attr is not None and edge_attr.size(-1) <= concat.size(-1):
            # Global edge summary (mean) appended
            edge_summary = edge_attr.mean(0, keepdim=True).expand(N, -1)
            concat = torch.cat([concat, edge_summary], dim=-1)

        out = self.lin(concat)
        return F.relu(self.norm(out))


# ─────────────────────────────────────────────
# GAT Layer
# ─────────────────────────────────────────────

class GATConv(nn.Module):
    """
    Single-head Graph Attention layer.
    Attention: alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, **_) -> Tensor:
        N = x.size(0)
        h = self.W(x)                                  # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]

        # Attention coefficients
        h_cat = torch.cat([h[src], h[dst]], dim=-1)    # (E, 2*out_dim)
        e = F.leaky_relu(self.a(h_cat), 0.2).squeeze(-1)  # (E,)

        # Sparse softmax via scatter
        alpha = torch.zeros(N, device=x.device)
        alpha_exp = torch.exp(e - e.max())
        denom = torch.zeros(N, device=x.device)
        denom.scatter_add_(0, dst, alpha_exp)
        alpha_norm = alpha_exp / (denom[dst] + 1e-9)
        alpha_norm = self.dropout(alpha_norm)

        # Aggregate
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(h[src]), alpha_norm.unsqueeze(1) * h[src])
        return F.elu(self.norm(out))


# ─────────────────────────────────────────────
# Full GNN Model
# ─────────────────────────────────────────────

class FraudGNN(nn.Module):
    """
    3-layer alternating SAGE + GAT encoder with residual connections.
    Outputs:
      - node embeddings (for feature store ingestion)
      - fraud logits (for node-level classification)
      - ring scores (community-level anomaly via embedding similarity)
    """

    def __init__(self, cfg: GNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        dims = [cfg.node_feat_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [cfg.embed_dim]

        self.layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            if i % 2 == 0:
                self.layers.append(SAGEConv(dims[i], dims[i + 1], cfg.edge_feat_dim if cfg.use_edge_features else 0))
            else:
                self.layers.append(GATConv(dims[i], dims[i + 1], cfg.dropout))

        # Residual projection when dims differ
        self.res_projs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) if dims[i] != dims[i + 1] else nn.Identity()
            for i in range(cfg.num_layers)
        ])

        self.dropout = nn.Dropout(cfg.dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim // 2, 2),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> dict[str, Tensor]:
        h = x
        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_index, edge_attr if isinstance(layer, SAGEConv) else None)
            h = h_new + self.res_projs[i](h)
            h = self.dropout(h)

        logits = self.classifier(h)
        return {"embeddings": h, "logits": logits}

    @torch.no_grad()
    def get_node_embeddings(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None
    ) -> Tensor:
        self.eval()
        return self.forward(x, edge_index, edge_attr)["embeddings"]

    @torch.no_grad()
    def detect_fraud_rings(
        self, embeddings: Tensor, threshold: float = 0.85
    ) -> list[list[int]]:
        """
        Simple cosine-similarity community detection.
        Two nodes are in the same ring if cos_sim > threshold.
        Returns list of suspected fraud rings (groups of node indices).
        """
        normed = F.normalize(embeddings, dim=-1)
        sim = normed @ normed.T                   # (N, N)
        sim.fill_diagonal_(0.0)

        visited = set()
        rings: list[list[int]] = []

        for i in range(len(embeddings)):
            if i in visited:
                continue
            neighbours = (sim[i] > threshold).nonzero(as_tuple=True)[0].tolist()
            if len(neighbours) >= 2:
                ring = [i] + neighbours
                rings.append(ring)
                visited.update(ring)

        return rings


# ─────────────────────────────────────────────
# GNN Trainer
# ─────────────────────────────────────────────

class GNNTrainer:

    def __init__(self, model: FraudGNN, lr: float = 5e-4, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(
        self,
        graph: GraphData,
        labelled_mask: Tensor,
    ) -> dict[str, float]:
        self.model.train()
        x = graph.node_features["all"].to(self.device)
        ei = graph.edge_index.to(self.device)
        ea = graph.edge_features.to(self.device)

        label_indices = list(graph.labels.keys())
        labels = torch.tensor([graph.labels[i] for i in label_indices], dtype=torch.long).to(self.device)

        out = self.model(x, ei, ea)
        logits_labelled = out["logits"][label_indices]

        # Focal loss for imbalanced fraud labels
        ce = F.cross_entropy(logits_labelled, labels, weight=torch.tensor([1.0, 10.0]).to(self.device))
        ce.backward()
        self.opt.step()
        self.opt.zero_grad()

        preds = logits_labelled.argmax(-1)
        acc = (preds == labels).float().mean().item()
        return {"loss": ce.item(), "accuracy": acc}

    def save(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict(), "config": self.model.cfg}, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> FraudGNN:
        ckpt = torch.load(path, map_location=device)
        model = FraudGNN(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        return model.to(device)
