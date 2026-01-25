# rl_models/baseline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl_models.env_dqn import VRPPDQNEnv, DQNEnvConfig

# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_BASELINE_DIR = PROJECT_ROOT / "outputs" / "baseline"
OUTPUTS_BASELINE_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# Baseline policies
# =============================
def always_normal(obs) -> int:
    return 0


def always_priority(obs) -> int:
    return 1


def always_expedited(obs) -> int:
    return 2


BASELINE_POLICIES: Dict[str, Callable] = {
    "always_normal": always_normal,
    "always_priority": always_priority,
    "always_expedited": always_expedited,
}


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 42
    n_episodes: int = 300
    episode_len: int = 256


def run_baseline(
    name: str,
    policy_fn: Callable,
    cfg: BaselineConfig,
) -> pd.DataFrame:
    """
    Runs a baseline policy and returns episode-level results.
    """
    env_cfg = DQNEnvConfig(seed=cfg.seed, episode_len=cfg.episode_len)
    env = VRPPDQNEnv(env_cfg)

    rng = np.random.default_rng(cfg.seed)
    results: List[Dict] = []

    for ep in range(cfg.n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0

        a0 = a1 = a2 = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_return += reward
            ep_len += 1

            if action == 0:
                a0 += 1
            elif action == 1:
                a1 += 1
            elif action == 2:
                a2 += 1

        results.append(
            {
                "baseline": name,
                "episode": ep,
                "episode_return": ep_return,
                "episode_length": ep_len,
                "action_0_count": a0,
                "action_1_count": a1,
                "action_2_count": a2,
            }
        )

    return pd.DataFrame(results)


def main():
    cfg = BaselineConfig()
    all_results: List[pd.DataFrame] = []

    for name, policy_fn in BASELINE_POLICIES.items():
        print(f"Running baseline: {name}")
        df = run_baseline(name, policy_fn, cfg)
        all_results.append(df)

        # Save per-baseline log
        df.to_csv(OUTPUTS_BASELINE_DIR / f"{name}_episode_log.csv", index=False)

        # Plot reward curve
        plt.figure(figsize=(8, 4))
        plt.plot(df["episode"], df["episode_return"])
        plt.xlabel("Episode")
        plt.ylabel("Episode return")
        plt.title(f"Baseline: {name}")
        plt.tight_layout()
        plt.savefig(OUTPUTS_BASELINE_DIR / f"{name}_reward_curve.png", dpi=160)
        plt.close()

    # Combined summary
    df_all = pd.concat(all_results, ignore_index=True)

    summary = (
        df_all.groupby("baseline")["episode_return"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.to_csv(OUTPUTS_BASELINE_DIR / "summary.csv", index=False)

    print("\n=== Baseline summary ===")
    print(summary)


if __name__ == "__main__":
    main()
