# vrpp/rl_models/train_dqn.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rl_models.env_dqn import VRPPDQNEnv, DQNEnvConfig, OUTPUTS_DQN_DIR


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 300_000

    # DQN hyperparams (reasonable defaults)
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 5_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1_000

    # Exploration schedule (tunable)
    exploration_fraction: float = 0.20
    exploration_final_eps: float = 0.05

    # Env
    episode_len: int = 256


class EpisodeReturnCallback(BaseCallback):
    """
    Logs episode-level returns using info["episode"] provided by Monitor.

    Also tracks action distribution per episode for interpretability.
    """

    def __init__(self):
        super().__init__()
        self.rows: List[Dict] = []
        self._ep_actions: List[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True

        info = infos[0]  # single env

        # Track actions (we add "action" into env info)
        if isinstance(info, dict) and "action" in info:
            self._ep_actions.append(int(info["action"]))

        # Monitor adds this dict ONLY when an episode ends:
        # info["episode"] = {"r": episode_return, "l": episode_length, "t": elapsed_time}
        ep = info.get("episode", None)
        if ep is not None:
            a0 = self._ep_actions.count(0)
            a1 = self._ep_actions.count(1)
            a2 = self._ep_actions.count(2)

            total = max(a0 + a1 + a2, 1)

            self.rows.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "episode_return": float(ep["r"]),
                    "episode_length": int(ep["l"]),
                    "action_0_count": int(a0),
                    "action_1_count": int(a1),
                    "action_2_count": int(a2),
                    "action_0_share": float(a0 / total),
                    "action_1_share": float(a1 / total),
                    "action_2_share": float(a2 / total),
                }
            )

            # reset per-episode tracker
            self._ep_actions.clear()

        return True


def ensure_outputs_dir() -> Path:
    OUTPUTS_DQN_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DQN_DIR


def save_plots(df_ep: pd.DataFrame, out_dir: Path) -> None:
    # 1) Episode return curve (raw)
    plt.figure(figsize=(8, 4))
    plt.plot(df_ep["timesteps"], df_ep["episode_return"])
    plt.xlabel("Timesteps")
    plt.ylabel("Episode return")
    plt.title("DQN learning curve (episode return)")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png", dpi=160)
    plt.close()

    # 2) Smoothed episode return
    df_ep = df_ep.copy()
    df_ep["return_smooth"] = df_ep["episode_return"].rolling(window=10, min_periods=1).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(df_ep["timesteps"], df_ep["return_smooth"])
    plt.xlabel("Timesteps")
    plt.ylabel("Smoothed episode return (rolling=10)")
    plt.title("Smoothed DQN learning curve (episode return)")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve_smooth.png", dpi=160)
    plt.close()

    # 3) Overall action distribution (sum counts across episodes)
    a0 = int(df_ep["action_0_count"].sum())
    a1 = int(df_ep["action_1_count"].sum())
    a2 = int(df_ep["action_2_count"].sum())

    plt.figure(figsize=(5, 4))
    plt.bar(["Normal(0)", "Priority(1)", "Expedited(2)"], [a0, a1, a2])
    plt.xlabel("Action")
    plt.ylabel("Total count")
    plt.title("Overall DQN action distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "action_hist.png", dpi=160)
    plt.close()

    # 4) Action shares over time (episode-level)
    plt.figure(figsize=(8, 4))
    plt.plot(df_ep["timesteps"], df_ep["action_0_share"], label="Normal (0)")
    plt.plot(df_ep["timesteps"], df_ep["action_1_share"], label="Priority (1)")
    plt.plot(df_ep["timesteps"], df_ep["action_2_share"], label="Expedited (2)")
    plt.xlabel("Timesteps")
    plt.ylabel("Action share (per episode)")
    plt.title("DQN action selection over time (episode shares)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "action_over_time.png", dpi=160)
    plt.close()


def main():
    cfg = TrainConfig()
    set_random_seed(cfg.seed)

    out_dir = ensure_outputs_dir()

    # Build env + wrap with Monitor to get episode returns
    env_cfg = DQNEnvConfig(seed=cfg.seed, episode_len=cfg.episode_len)
    env = VRPPDQNEnv(env_cfg)

    # Monitor writes optional monitor.csv if you pass filename; we don't need it,
    # we just need info["episode"] so keep it simple:
    env = Monitor(env)

    cb = EpisodeReturnCallback()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        verbose=1,
        seed=cfg.seed,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=True)

    # Save model
    model_path = out_dir / "dqn_v1.zip"
    model.save(model_path.as_posix())

    # Save episode log
    df_ep = pd.DataFrame(cb.rows)
    log_path = out_dir / "episode_log.csv"
    df_ep.to_csv(log_path, index=False)

    # Save plots
    if len(df_ep) > 0:
        save_plots(df_ep, out_dir)

    print("\n=== DQN outputs saved ===")
    print(f"Model:   {model_path}")
    print(f"Log:     {log_path}")
    print(f"Plots:   {out_dir / 'reward_curve.png'}")
    print(f"         {out_dir / 'reward_curve_smooth.png'}")
    print(f"         {out_dir / 'action_hist.png'}")
    print(f"         {out_dir / 'action_over_time.png'}")


if __name__ == "__main__":
    main()

