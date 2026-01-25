# vrpp/rl_models/env_ppo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import json
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Paths (ALWAYS from project root)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../vrpp
DATA_DIR = PROJECT_ROOT / "data"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_PPO_DIR = OUTPUTS_DIR / "ppo"


@dataclass(frozen=True)
class PPOEnvConfig:
    parquet_path: Path = DATA_DIR / "rl_state_base.parquet"
    reward_stats_path: Path = DATA_DIR / "reward_stats.json"

    # Episode mechanics
    episode_len: int = 256
    seed: int = 42

    # Reward shaping fallback (only used if dataset doesn't have per-action delay columns)
    action_delay_multipliers: Tuple[float, float, float] = (1.0, 0.85, 0.70)


class VRPPPPOEnv(gym.Env):
    """
    VRPP PPO environment.

    State (expected columns):
      - p_major
      - p_minor
      - season_* one-hot (auto-detect)
      - weight_norm (or similar)

    Actions:
      0 Normal, 1 Priority, 2 Expedited

    Reward:
      R = -clip(delay_hours(action), 0, p99_delay_hours)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[PPOEnvConfig] = None):
        super().__init__()
        self.cfg = config or PPOEnvConfig()

        self._rng = np.random.default_rng(self.cfg.seed)
        self._df = self._load_df(self.cfg.parquet_path)
        self._p99 = self._load_p99(self.cfg.reward_stats_path)

        self._state_cols = self._infer_state_cols(self._df)
        self._delay_cols = self._infer_delay_cols(self._df)

        obs_dim = len(self._state_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._ep_indices: np.ndarray = np.array([], dtype=np.int64)
        self._t: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._ep_indices = self._sample_episode_indices(self.cfg.episode_len)

        obs = self._get_obs(int(self._ep_indices[self._t]))
        info = {"row_idx": int(self._ep_indices[self._t])}
        return obs, info

    def step(self, action: int):
        action = int(action)
        row_idx = int(self._ep_indices[self._t])

        raw_delay = self._get_delay_hours(row_idx, action)
        clipped_delay = float(np.clip(raw_delay, 0.0, self._p99))
        reward = -clipped_delay

        self._t += 1
        terminated = self._t >= len(self._ep_indices)
        truncated = False

        if not terminated:
            next_obs = self._get_obs(int(self._ep_indices[self._t]))
        else:
            next_obs = self._get_obs(row_idx)

        info = {
            "row_idx": row_idx,
            "action": action,
            "raw_delay_hours": float(raw_delay),
            "clipped_delay_hours": clipped_delay,
            "p99_delay_hours": float(self._p99),
        }
        return next_obs, float(reward), terminated, truncated, info

    # -----------------------------
    # Internals
    # -----------------------------
    @staticmethod
    def _load_df(parquet_path: Path) -> pd.DataFrame:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to read parquet. Install an engine in your venv, e.g.:\n"
                "  pip install pyarrow\n"
                f"Original error: {e}"
            )

    @staticmethod
    def _load_p99(reward_stats_path: Path) -> float:
        if not reward_stats_path.exists():
            raise FileNotFoundError(f"reward_stats.json not found: {reward_stats_path}")
        with reward_stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        if "p99_delay_hours" not in stats:
            raise KeyError("reward_stats.json must contain key: 'p99_delay_hours'")
        return float(stats["p99_delay_hours"])

    @staticmethod
    def _infer_state_cols(df: pd.DataFrame) -> list[str]:
        cols = df.columns.tolist()

        base_candidates = ["p_major", "p_minor"]
        weight_candidates = ["weight_norm", "weight_normalized", "weight_scaled", "weight"]

        season_cols = sorted([c for c in cols if c.lower().startswith("season")])
        state_cols: list[str] = []

        for c in base_candidates:
            if c in cols:
                state_cols.append(c)

        state_cols.extend(season_cols)

        for wc in weight_candidates:
            if wc in cols:
                state_cols.append(wc)
                break

        if len(state_cols) == 0:
            raise ValueError(
                "Could not infer state columns. Expected at least some of: "
                "p_major, p_minor, season*, weight_norm (or similar)."
            )

        for c in state_cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                raise TypeError(f"State column must be numeric: {c} (dtype={df[c].dtype})")

        return state_cols

    @staticmethod
    def _infer_delay_cols(df: pd.DataFrame) -> tuple[str, ...]:
        cols = set(df.columns)

        triplets = [
            ("delay_hours_a0", "delay_hours_a1", "delay_hours_a2"),
            ("delay_hours_normal", "delay_hours_priority", "delay_hours_expedited"),
            ("Delay_Hours_Normal", "Delay_Hours_Priority", "Delay_Hours_Expedited"),
        ]
        for a0, a1, a2 in triplets:
            if a0 in cols and a1 in cols and a2 in cols:
                return (a0, a1, a2)

        singles = ["delay_hours", "Delay_Hours", "delay", "Delay"]
        for s in singles:
            if s in cols:
                return (s,)

        raise ValueError(
            "Could not infer delay column(s). Provide either:\n"
            "  - one column: delay_hours / Delay_Hours\n"
            "  - OR three columns: action-specific delays (normal/priority/expedited)."
        )

    def _sample_episode_indices(self, episode_len: int) -> np.ndarray:
        n = len(self._df)
        if n == 0:
            raise ValueError("rl_state_base.parquet is empty.")
        idx = self._rng.integers(low=0, high=n, size=episode_len, endpoint=False)
        return idx.astype(np.int64)

    def _get_obs(self, row_idx: int) -> np.ndarray:
        row = self._df.iloc[row_idx]
        return row[self._state_cols].to_numpy(dtype=np.float32, copy=False)

    def _get_delay_hours(self, row_idx: int, action: int) -> float:
        row = self._df.iloc[row_idx]

        if len(self._delay_cols) == 3:
            return float(row[self._delay_cols[action]])

        base_delay = float(row[self._delay_cols[0]])
        mult = float(self.cfg.action_delay_multipliers[action])
        return base_delay * mult
