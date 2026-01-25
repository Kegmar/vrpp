# vrpp/rl_models/train_ppo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from rl_models.env_ppo import PPOEnvConfig, VRPPPPOEnv, PROJECT_ROOT, OUTPUTS_PPO_DIR


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    n_envs: int = 8

    total_timesteps: int = 300_000

    # PPO hyperparams
    n_steps: int = 2048
    batch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 10

    episode_len: int = 256


class RolloutStatsCallback(BaseCallback):
    """
    Collects simple training stats so we can dump CSV + plots into outputs/ppo.
    """

    def __init__(self):
        super().__init__()
        self.rows: List[Dict] = []

        # Rolling counters
        self._step_rewards: List[float] = []
        self._actions: List[int] = []

    def _on_step(self) -> bool:
        # rewards for vectorized envs come in self.locals["rewards"]
        rewards = self.locals.get("rewards", None)
        infos = self.locals.get("infos", None)

        if rewards is not None:
            # rewards is np.array shape (n_envs,)
            self._step_rewards.extend([float(r) for r in rewards])

        if infos is not None:
            for inf in infos:
                if isinstance(inf, dict) and "action" in inf:
                    self._actions.append(int(inf["action"]))

        # Every rollout end, SB3 calls _on_rollout_end
        return True

    def _on_rollout_end(self) -> None:
        # Summarize what we collected since last rollout end
        if len(self._step_rewards) == 0:
            return

        mean_r = float(np.mean(self._step_rewards))
        std_r = float(np.std(self._step_rewards))
        min_r = float(np.min(self._step_rewards))
        max_r = float(np.max(self._step_rewards))

        action_counts = {0: 0, 1: 0, 2: 0}
        for a in self._actions:
            if a in action_counts:
                action_counts[a] += 1

        self.rows.append(
            {
                "timesteps": int(self.num_timesteps),
                "mean_step_reward": mean_r,
                "std_step_reward": std_r,
                "min_step_reward": min_r,
                "max_step_reward": max_r,
                "action_0_count": int(action_counts[0]),
                "action_1_count": int(action_counts[1]),
                "action_2_count": int(action_counts[2]),
            }
        )

        # Reset buffers
        self._step_rewards.clear()
        self._actions.clear()


def ensure_outputs_dir() -> Path:
    OUTPUTS_PPO_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_PPO_DIR


def save_plots(df_log: pd.DataFrame, out_dir: Path) -> None:
    # Reward curve
    plt.figure()
    plt.plot(df_log["timesteps"], df_log["mean_step_reward"])
    plt.xlabel("Timesteps")
    plt.ylabel("Mean step reward")
    plt.title("PPO training: mean step reward")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png", dpi=160)
    plt.close()

    # Action histogram (aggregate across rollouts)
    a0 = int(df_log["action_0_count"].sum())
    a1 = int(df_log["action_1_count"].sum())
    a2 = int(df_log["action_2_count"].sum())

    plt.figure()
    plt.bar(["Normal(0)", "Priority(1)", "Expedited(2)"], [a0, a1, a2])
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("PPO training: action distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "action_hist.png", dpi=160)
    plt.close()


def main():
    cfg = TrainConfig()
    set_random_seed(cfg.seed)

    out_dir = ensure_outputs_dir()

    # Env config
    env_cfg = PPOEnvConfig(
        episode_len=cfg.episode_len,
        seed=cfg.seed,
    )

    # Vectorized env
    vec_env = make_vec_env(
        env_id=lambda: VRPPPPOEnv(env_cfg),
        n_envs=cfg.n_envs,
        seed=cfg.seed,
    )

    # Callback for logging
    cb = RolloutStatsCallback()

    # Model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=cfg.seed,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gae_lambda=cfg.gae_lambda,
        gamma=cfg.gamma,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        clip_range=cfg.clip_range,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # Train
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=True)

    # Save model
    model_path = out_dir / "ppo_v1.zip"
    model.save(model_path.as_posix())

    # Save logs
    df_log = pd.DataFrame(cb.rows)
    log_path = out_dir / "train_log.csv"
    df_log.to_csv(log_path, index=False)

    # Save plots
    if len(df_log) > 0:
        save_plots(df_log, out_dir)

    print("\n=== PPO outputs saved ===")
    print(f"Model:   {model_path}")
    print(f"Log:     {log_path}")
    print(f"Plots:   {out_dir / 'reward_curve.png'}")
    print(f"         {out_dir / 'action_hist.png'}")


if __name__ == "__main__":
    main()
