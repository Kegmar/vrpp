# rl_models/env_dqn.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import json
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces


# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OUTPUTS_DQN_DIR = PROJECT_ROOT / "outputs" / "dqn"


@dataclass(frozen=True)
class DQNEnvConfig:
    parquet_path: Path = DATA_DIR / "rl_state_base.parquet"
    reward_stats_path: Path = DATA_DIR / "reward_stats.json"

    episode_len: int = 256
    seed: int = 42

    action_delay_multipliers: Tuple[float, float, float] = (1.0, 0.85, 0.70)


class VRPPDQNEnv(gym.Env):
    """
    DQN environment for VRPP.

    State: same as PPO
    Action: Discrete(3)
    Reward: -clip(delay_hours, 0, p99)
    """

    def __init__(self, config: Optional[DQNEnvConfig] = None):
        super().__init__()
        self.cfg = config or DQNEnvConfig()

        self._rng = np.random.default_rng(self.cfg.seed)
        self._df = self._load_df(self.cfg.parquet_path)
        self._p99 = self._load_p99(self.cfg.reward_stats_path)

        self._state_cols = self._infer_state_cols(self._df)
        self._delay_cols = self._infer_delay_cols(self._df)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._state_cols),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._ep_idx = None
        self._t = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._ep_idx = self._sample_indices(self.cfg.episode_len)

        obs = self._get_obs(self._ep_idx[self._t])
        return obs, {}

    def step(self, action: int):
        row_idx = self._ep_idx[self._t]

        delay = self._get_delay(row_idx, int(action))
        delay_clipped = float(np.clip(delay, 0.0, self._p99))
        reward = -delay_clipped

        self._t += 1
        terminated = self._t >= len(self._ep_idx)
        truncated = False

        obs = (
            self._get_obs(self._ep_idx[self._t])
            if not terminated
            else self._get_obs(row_idx)
        )

        info = {
            "action": int(action),
            "raw_delay": float(delay),
            "clipped_delay": delay_clipped,
        }

        return obs, reward, terminated, truncated, info

    # =============================
    # Helpers
    # =============================
    @staticmethod
    def _load_df(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def _load_p99(path: Path) -> float:
        with open(path, "r") as f:
            return float(json.load(f)["p99_delay_hours"])

    @staticmethod
    def _infer_state_cols(df: pd.DataFrame) -> list[str]:
        cols = df.columns
        base = [c for c in ["p_major", "p_minor"] if c in cols]
        season = sorted([c for c in cols if c.lower().startswith("season")])
        weight = next(
            (c for c in ["weight_norm", "weight"] if c in cols),
            None,
        )
        state = base + season + ([weight] if weight else [])
        if not state:
            raise ValueError("No valid state columns found")
        return state

    @staticmethod
    def _infer_delay_cols(df: pd.DataFrame) -> tuple[str, ...]:
        triplets = [
            ("delay_hours_normal", "delay_hours_priority", "delay_hours_expedited"),
            ("Delay_Hours_Normal", "Delay_Hours_Priority", "Delay_Hours_Expedited"),
        ]
        for t in triplets:
            if all(c in df.columns for c in t):
                return t

        for c in ["delay_hours", "Delay_Hours"]:
            if c in df.columns:
                return (c,)

        raise ValueError("No delay columns found")

    def _sample_indices(self, n: int):
        return self._rng.integers(0, len(self._df), size=n)

    def _get_obs(self, idx: int):
        return self._df.iloc[idx][self._state_cols].to_numpy(dtype=np.float32)

    def _get_delay(self, idx: int, action: int):
        row = self._df.iloc[idx]
        if len(self._delay_cols) == 3:
            return row[self._delay_cols[action]]
        return row[self._delay_cols[0]] * self.cfg.action_delay_multipliers[action]
