import random
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from stable_baselines3 import DQN as SB3DQN

from amr_ex_12_sol.utils import QNetwork, ReplayBuffer


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 10000
    exploration_initial_eps: float = 0.5
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.1
    target_update_interval: int = 10
    learning_starts: int = 500
    tau: float = 0.005


@dataclass
class Config:
    trainer_cfg: DQNConfig = field(default_factory=DQNConfig)
    use_stablebaselines: bool = False
    total_timesteps: int = 500_000
    device: str = "cpu"


class DQNTrainer:
    def __init__(self, env: gym.Env, cfg: DQNConfig, device: torch.device | str):
        self.env = env
        self.device = device

        state_dim = env.observation_space.shape[0]  # ty: ignore
        action_dim = env.action_space.n  # ty: ignore

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.memory: ReplayBuffer = ReplayBuffer(cfg.buffer_size)

        # Hyperparameters
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.learning_starts = cfg.learning_starts
        self.tau = cfg.tau
        self.target_update_interval = cfg.target_update_interval

        # Epsilon params
        self.eps_start = cfg.exploration_initial_eps
        self.eps_end = cfg.exploration_final_eps
        self.eps_fraction = cfg.exploration_fraction
        self.curr_epsilon = cfg.exploration_initial_eps

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        obs_t = torch.FloatTensor(observation).to(self.device)
        with torch.no_grad():
            action = self.q_net(obs_t).argmax().item()
        return np.array(action), state

    def _update_epsilon(self, step: int, total_steps: int):
        # Linear decay over the specified fraction of steps
        decay_duration = int(total_steps * self.eps_fraction)
        if step < decay_duration:
            self.curr_epsilon = self.eps_start - (step / decay_duration) * (
                self.eps_start - self.eps_end
            )
        else:
            self.curr_epsilon = self.eps_end

    def learn(self, total_timesteps: int) -> "DQNTrainer":
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_count = 0

        for global_step in range(total_timesteps):
            self._update_epsilon(global_step, total_timesteps)

            if random.random() < self.curr_epsilon:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict(obs, deterministic=True)

            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.memory.push(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += float(reward)

            if len(self.memory) > self.learning_starts:
                b_s, b_a, b_r, b_ns, b_d = self.memory.sample(
                    self.batch_size, self.device
                )

                current_q = self.q_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = self.target_net(b_ns).max(1)[0]
                    expected_q = b_r + (self.gamma * max_next_q * ~b_d)

                loss = nn.MSELoss()(current_q, expected_q)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optimizer.step()

                # Soft update target network
                for target_param, local_param in zip(
                    self.target_net.parameters(), self.q_net.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * local_param.data
                        + (1.0 - self.tau) * target_param.data
                    )

            if done:
                if episode_count % 10 == 0:
                    print(
                        f"Step: {global_step}/{total_timesteps} | Last Reward: {episode_reward:.2f} | Eps: {self.curr_epsilon:.2f}"
                    )
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_count += 1

        return self


def main(cfg: Config):
    env = gym.make("LunarLander-v3")

    if cfg.use_stablebaselines:
        trainer = SB3DQN("MlpPolicy", env, verbose=1, **asdict(cfg.trainer_cfg))
    else:
        trainer = DQNTrainer(env, cfg.trainer_cfg, cfg.device)

    print("learning")
    trainer.learn(total_timesteps=cfg.total_timesteps)

    print("evaluating")
    env = gym.make("LunarLander-v3", render_mode="human")
    for i in range(5):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = trainer.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            episode_reward += float(reward)
            if term or trunc:
                done = True
                print(f"accumulated reward for episode {i + 1}: {episode_reward:.1f}")


if __name__ == "__main__":
    tyro.cli(main)
