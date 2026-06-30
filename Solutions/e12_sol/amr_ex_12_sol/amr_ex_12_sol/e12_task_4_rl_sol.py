from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Dict, Literal, Optional, Type

import gymnasium as gym
import tyro
from stable_baselines3 import DQN, PPO, SAC

from amr_ex_12_sol.e12_task_3_dqn_sol import DQNConfig


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    """Learning rate for the optimizer"""
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class SACConfig:
    learning_rate: float = 3e-4
    """Learning rate for the optimizer"""
    learning_starts: int = 100
    buffer_size: int = 256
    tau: float = 0.005


@dataclass
class Config:
    """Configuration for RL training."""

    algo: (
        Annotated[PPOConfig, tyro.conf.subcommand(name="ppo")]
        | Annotated[SACConfig, tyro.conf.subcommand(name="sac")]
        | Annotated[DQNConfig, tyro.conf.subcommand(name="dqn")]
    ) = field(default_factory=DQNConfig)
    """What specific RL algorithm to run"""
    env_id: str = "Walker2d-v5"
    """The Gym environment"""
    total_timesteps: int = 500_000
    """Total number of timesteps to train"""
    device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    """The device to run on"""
    save_path: Optional[str] = "my_policy"
    """Path to save the trained trainer"""
    eval_episodes: int = 5
    """How many evaluation episodes to run after trainig"""


# Algorithm registry so we can associate an RL algorithm with its config class.
ALGO_REGISTRY: Dict[Type, Any] = {
    DQNConfig: DQN,
    PPOConfig: PPO,
    SACConfig: SAC,
}


def main(cfg: Config):
    print(f"🚀 Initializing training for: {cfg.env_id}")

    algo_type = type(cfg.algo)
    SB3Class = ALGO_REGISTRY[algo_type]

    print(f"🛠️ Instantiating {SB3Class.__name__} for {cfg.env_id}")

    env = gym.make(cfg.env_id)

    # NOTE: We use asdict() to unpack the dataclass into keyword arguments
    trainer = SB3Class(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        **asdict(cfg.algo),
    )

    print(f"📈 Training for {cfg.total_timesteps} steps...")
    trainer.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)

    if cfg.save_path:
        trainer.save(cfg.save_path)
        print(f"💾 trainer saved to {cfg.save_path}.zip")

    if cfg.eval_episodes > 0:
        print(f"👀 Evaluating for {cfg.eval_episodes} episodes...")
        # NOTE: Re-create env so we can actually see what is happening
        eval_env = gym.make(cfg.env_id, render_mode="human")

        for episode in range(cfg.eval_episodes):
            obs, info = eval_env.reset()
            terminated = truncated = False
            score = 0

            while not (terminated or truncated):
                action, _states = trainer.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                score += float(reward)

            print(f"Episode {episode + 1}: Score = {score}")

        eval_env.close()


if __name__ == "__main__":
    tyro.cli(
        main,
        config=(
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitArgPrefixes,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
