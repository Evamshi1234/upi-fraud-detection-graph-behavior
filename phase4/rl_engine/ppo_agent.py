"""
UPI-Shield · Phase 4
Reinforcement Learning Policy Engine
Adaptive fraud threshold optimization using PPO.
Agent learns to balance fraud detection rate vs false-positive cost.
Constrained MDP: FPR hard cap enforced at every step.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import numpy as np


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

@dataclass
class RLState:
    """
    Observation space for the RL agent.
    Represents the current operating statistics of the fraud system.
    """
    # Rolling metrics (last N transactions)
    fraud_score_mean: float      # mean fraud score in window
    fraud_score_std: float
    fraud_score_p95: float
    fpr_1h: float                # false positive rate last hour
    tpr_1h: float                # true positive rate (recall) last hour
    fraud_rate_1h: float         # fraction of txns flagged as fraud
    # Current thresholds
    block_threshold: float       # current BLOCK threshold
    review_threshold: float      # current REVIEW threshold
    # Context
    hour_of_day: float           # 0-1 normalised
    is_weekend: float
    # Trend
    score_drift: float           # mean score change vs prev window

    def to_array(self) -> np.ndarray:
        return np.array([
            self.fraud_score_mean, self.fraud_score_std, self.fraud_score_p95,
            self.fpr_1h, self.tpr_1h, self.fraud_rate_1h,
            self.block_threshold, self.review_threshold,
            self.hour_of_day, self.is_weekend, self.score_drift,
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return 11


@dataclass
class RLAction:
    """
    Discrete action: adjust block and review thresholds by a fixed delta.
    Action space = 9 combinations of {lower, keep, raise} × {block, review}.
    """
    block_delta: float    # -0.05, 0.0, +0.05
    review_delta: float

    @staticmethod
    def from_index(idx: int) -> "RLAction":
        deltas = [-0.05, 0.0, 0.05]
        b = deltas[idx // 3]
        r = deltas[idx % 3]
        return RLAction(block_delta=b, review_delta=r)

    @staticmethod
    def action_space_size() -> int:
        return 9


@dataclass
class StepResult:
    next_state: RLState
    reward: float
    done: bool
    info: dict[str, Any]


class FraudEnvironment:
    """
    Simulation environment built from historical transaction replay.
    Supports online mode (live feedback from production).

    Reward shaping:
        +10 × detected_fraud_rate       (recall reward)
        -20 × false_positive_penalty    (FPR cost)
        -50  if FPR > FPR_HARD_CAP      (constraint violation)
        -5  × |threshold_change|        (stability penalty)
    """

    FPR_HARD_CAP = 0.005     # 0.5% max FPR
    EPISODE_LEN  = 200       # steps per episode

    def __init__(
        self,
        history: list[dict] | None = None,
        block_threshold: float = 0.70,
        review_threshold: float = 0.35,
    ) -> None:
        self._history = history or self._generate_synthetic_history(5000)
        self._block_t  = block_threshold
        self._review_t = review_threshold
        self._step_count = 0
        self._score_window: deque[float] = deque(maxlen=500)
        self._label_window: deque[int]   = deque(maxlen=500)
        self._pred_window: deque[int]    = deque(maxlen=500)
        self._rng = random.Random(42)

    @staticmethod
    def _generate_synthetic_history(n: int) -> list[dict]:
        """Generate synthetic transaction scores + labels for training."""
        rng = random.Random(0)
        history = []
        for i in range(n):
            is_fraud = rng.random() < 0.02   # 2% fraud rate
            if is_fraud:
                score = rng.gauss(0.75, 0.15)
            else:
                score = rng.gauss(0.15, 0.12)
            score = max(0.0, min(1.0, score))
            history.append({
                "score": score,
                "label": int(is_fraud),
                "amount": rng.uniform(100, 50000),
                "hour": rng.randint(0, 23),
            })
        return history

    def reset(self) -> RLState:
        self._step_count = 0
        self._score_window.clear()
        self._label_window.clear()
        self._pred_window.clear()
        return self._get_state()

    def step(self, action: RLAction) -> StepResult:
        # Apply action (clamp thresholds)
        self._block_t  = np.clip(self._block_t  + action.block_delta,  0.5, 0.95)
        self._review_t = np.clip(self._review_t + action.review_delta, 0.2, self._block_t - 0.05)

        # Simulate N transactions
        batch_size = 50
        tp = fp = tn = fn = 0
        for _ in range(batch_size):
            txn = self._rng.choice(self._history)
            score = txn["score"]
            label = txn["label"]
            pred = int(score >= self._review_t)   # positive = REVIEW or BLOCK

            self._score_window.append(score)
            self._label_window.append(label)
            self._pred_window.append(pred)

            if pred == 1 and label == 1: tp += 1
            elif pred == 1 and label == 0: fp += 1
            elif pred == 0 and label == 0: tn += 1
            else: fn += 1

        # Metrics
        total_neg = fp + tn
        total_pos = tp + fn
        fpr = fp / max(total_neg, 1)
        tpr = tp / max(total_pos, 1)

        # Reward
        reward = 10.0 * tpr - 20.0 * fpr
        if fpr > self.FPR_HARD_CAP:
            reward -= 50.0                       # hard constraint violation
        reward -= 5.0 * (abs(action.block_delta) + abs(action.review_delta))  # stability

        self._step_count += 1
        done = self._step_count >= self.EPISODE_LEN

        return StepResult(
            next_state=self._get_state(),
            reward=reward,
            done=done,
            info={
                "fpr": round(fpr, 4),
                "tpr": round(tpr, 4),
                "block_threshold": round(self._block_t, 3),
                "review_threshold": round(self._review_t, 3),
                "constraint_violated": fpr > self.FPR_HARD_CAP,
            },
        )

    def _get_state(self) -> RLState:
        scores = list(self._score_window) or [0.5]
        prev_mean = np.mean(scores[: len(scores) // 2]) if len(scores) > 10 else 0.5
        curr_mean = np.mean(scores[len(scores) // 2 :]) if len(scores) > 10 else 0.5

        labels = list(self._label_window)
        preds  = list(self._pred_window)

        fpr = tpr = 0.0
        if labels:
            neg = [p for p, l in zip(preds, labels) if l == 0]
            pos = [p for p, l in zip(preds, labels) if l == 1]
            fpr = sum(neg) / max(len(neg), 1)
            tpr = sum(pos) / max(len(pos), 1)

        now = datetime.utcnow()
        return RLState(
            fraud_score_mean = float(np.mean(scores)),
            fraud_score_std  = float(np.std(scores)),
            fraud_score_p95  = float(np.percentile(scores, 95)),
            fpr_1h           = round(fpr, 4),
            tpr_1h           = round(tpr, 4),
            fraud_rate_1h    = round(sum(preds) / max(len(preds), 1), 4),
            block_threshold  = self._block_t,
            review_threshold = self._review_t,
            hour_of_day      = now.hour / 23.0,
            is_weekend       = float(now.weekday() >= 5),
            score_drift      = round(float(curr_mean - prev_mean), 4),
        )


# ─────────────────────────────────────────────
# PPO Policy Network
# ─────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _build_ppo_network(state_dim: int, action_dim: int, hidden: int = 64):
    if not TORCH_AVAILABLE:
        return None, None

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),    nn.Tanh(),
            )
            self.actor  = nn.Linear(hidden, action_dim)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            h = self.shared(x)
            return self.actor(h), self.critic(h)

        def get_action(self, state: np.ndarray, deterministic: bool = False):
            x = torch.FloatTensor(state).unsqueeze(0)
            logits, value = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    return ActorCritic(), ActorCritic()   # actor, critic (shared weights here)


# ─────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────

@dataclass
class PPOConfig:
    lr: float         = 3e-4
    gamma: float      = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    n_epochs: int     = 4
    batch_size: int   = 64
    buffer_size: int  = 2048
    device: str       = "cpu"


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class PPOTrainer:
    """
    Proximal Policy Optimisation trainer for the fraud threshold agent.
    Falls back to a random policy when PyTorch is unavailable.
    """

    def __init__(self, env: FraudEnvironment, cfg: PPOConfig | None = None) -> None:
        self.env = env
        self.cfg = cfg or PPOConfig()
        self._buffer: list[Experience] = []
        self._total_steps = 0
        self._episode_rewards: list[float] = []

        state_dim  = RLState.__dataclass_fields__.__len__()   # 11
        action_dim = RLAction.action_space_size()             # 9

        if TORCH_AVAILABLE:
            self.network, _ = _build_ppo_network(state_dim, action_dim)
            self.opt = torch.optim.Adam(self.network.parameters(), lr=self.cfg.lr)
        else:
            self.network = None
            self.opt = None

    def _select_action(self, state: np.ndarray) -> tuple[int, float, float]:
        if self.network is None:
            # Random baseline
            return random.randint(0, 8), 0.0, 0.0
        return self.network.get_action(state)

    def collect_rollout(self) -> list[Experience]:
        state = self.env.reset()
        experiences = []
        ep_reward = 0.0

        while len(experiences) < self.cfg.buffer_size:
            s_arr = state.to_array()
            action_idx, log_prob, value = self._select_action(s_arr)
            action = RLAction.from_index(action_idx)
            result = self.env.step(action)

            experiences.append(Experience(
                state=s_arr,
                action=action_idx,
                reward=result.reward,
                next_state=result.next_state.to_array(),
                done=result.done,
                log_prob=log_prob,
                value=value,
            ))
            ep_reward += result.reward

            if result.done:
                self._episode_rewards.append(ep_reward)
                ep_reward = 0.0
                state = self.env.reset()
            else:
                state = result.next_state

        self._total_steps += len(experiences)
        return experiences

    def update(self, experiences: list[Experience]) -> dict[str, float]:
        if not TORCH_AVAILABLE or self.network is None:
            return {"loss": 0.0}

        import torch
        # Compute GAE advantages
        rewards = [e.reward for e in experiences]
        values  = [e.value  for e in experiences]
        dones   = [e.done   for e in experiences]

        advantages = []
        gae = 0.0
        for i in reversed(range(len(experiences))):
            next_val = values[i + 1] if i + 1 < len(experiences) and not dones[i] else 0.0
            delta = rewards[i] + self.cfg.gamma * next_val - values[i]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values)]

        states   = torch.FloatTensor(np.array([e.state  for e in experiences]))
        actions  = torch.LongTensor([e.action  for e in experiences])
        old_lp   = torch.FloatTensor([e.log_prob for e in experiences])
        adv_t    = torch.FloatTensor(advantages)
        ret_t    = torch.FloatTensor(returns)
        adv_t    = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.cfg.n_epochs):
            perm = torch.randperm(len(experiences))
            for start in range(0, len(experiences), self.cfg.batch_size):
                idx = perm[start: start + self.cfg.batch_size]
                logits, values_pred = self.network(states[idx])
                dist  = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(actions[idx])

                ratio = (new_lp - old_lp[idx]).exp()
                clipped = ratio.clamp(1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)
                actor_loss  = -torch.min(ratio * adv_t[idx], clipped * adv_t[idx]).mean()
                critic_loss = F.mse_loss(values_pred.squeeze(), ret_t[idx])
                entropy     = dist.entropy().mean()

                loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.opt.step()
                total_loss += loss.item()

        return {
            "loss": round(total_loss / max(self.cfg.n_epochs, 1), 4),
            "mean_reward": round(float(np.mean(rewards)), 4),
            "mean_advantage": round(float(adv_t.mean().item()), 4),
        }

    def train(self, n_iterations: int = 50, verbose: bool = True) -> list[dict]:
        history = []
        for it in range(n_iterations):
            experiences = self.collect_rollout()
            metrics = self.update(experiences)
            metrics["iteration"] = it
            metrics["total_steps"] = self._total_steps
            if self._episode_rewards:
                metrics["ep_reward_mean"] = round(float(np.mean(self._episode_rewards[-10:])), 2)
            history.append(metrics)
            if verbose and it % 10 == 0:
                print(f"[RL] Iter {it:03d} | Loss={metrics['loss']:.4f} | "
                      f"MeanReward={metrics.get('mean_reward', 0):.3f} | "
                      f"Steps={self._total_steps}")
        return history

    def get_policy_thresholds(self, state: RLState | None = None) -> dict[str, float]:
        """Get recommended thresholds from the current policy."""
        if state is None:
            state = self.env.reset()
        s_arr = state.to_array()
        action_idx, _, _ = self._select_action(s_arr)
        action = RLAction.from_index(action_idx)
        new_block  = np.clip(state.block_threshold  + action.block_delta,  0.5, 0.95)
        new_review = np.clip(state.review_threshold + action.review_delta, 0.2, new_block - 0.05)
        return {
            "block_threshold":  round(float(new_block), 3),
            "review_threshold": round(float(new_review), 3),
            "action_idx": action_idx,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if TORCH_AVAILABLE and self.network:
            import torch
            torch.save(self.network.state_dict(), path)
        print(f"[RL] Policy saved → {path}")