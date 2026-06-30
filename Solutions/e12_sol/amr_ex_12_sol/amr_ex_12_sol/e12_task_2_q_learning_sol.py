import random
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import tyro

from amr_ex_12_sol.utils import play_human_vs_agent, plot_q_learning_policy


# 2. Hyperparameters
# TODO: Play around with these
@dataclass
class QLearningConfig:
    learning_rate: float = 0.01
    start_epsilon: float = 0.75
    epsilon_decay: float = 2e-6
    final_epsilon: float = 0.1
    discount_factor: float = 0.99


@dataclass
class Config:
    """BlackJack Q Learning environment config"""

    n_episodes: int = 500_000
    """Total number of episodes the learning agent can play"""

    q_learner: QLearningConfig = field(default_factory=QLearningConfig)
    """Q-Learning agent configuration"""

    play: bool = True
    """Whether the interactive playings session should start"""
    num_play_iter: int = 10
    """How many episodes you want to play blackjack"""


class QLearner:
    def __init__(self, config: QLearningConfig, action_space: gym.Space):
        self.action_space = action_space
        """The action space of the environment"""

        self.q_values = defaultdict(lambda: np.zeros(action_space.n))  # ty: ignore
        """Q-value table. you can index this with `self.q_values[obs][action]` to get the Q(s,a)"""

        self.config = config
        """Config structure"""

        self.epsilon = self.config.start_epsilon
        """Exploration probability"""

        self.is_training = True

    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False

    def get_action(self, obs):
        """Epsilon-greedy policy"""
        if not self.is_training:
            return int(np.argmax(self.q_values[obs]))

        if random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            return int(np.argmax(self.q_values[obs]))  # Exploit

    def step(
        self,
        current_obs: int,
        current_action: int,
        next_obs: int,
        reward: float,
        terminated: bool,
    ):
        """Implements the Q-Learning update step.

        Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        td_update = self.compute_temporal_difference(
            current_obs, current_action, next_obs, reward, terminated
        )
        self.q_values[current_obs][current_action] += (
            self.config.learning_rate * td_update
        )

    def compute_temporal_difference(
        self,
        current_obs: int,
        current_action: int,
        next_obs: int,
        reward: float,
        terminated: bool,
    ) -> float:
        """Compute the TD residual"""

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward
            + self.config.discount_factor * future_q_value
            - self.q_values[current_obs][current_action]
        )

        return temporal_difference

    def update_epsilon(self, iteration: int) -> None:
        """Decay epsilon along the training to switch from exploration to exploitation"""
        self.epsilon = max(
            self.config.final_epsilon, self.epsilon - self.config.epsilon_decay
        )


def main(config: Config):
    env = gym.make("Blackjack-v1")

    q_learner = QLearner(config.q_learner, env.action_space)

    q_learner.train()

    # 4. Training Loop
    for episode in range(config.n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = q_learner.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            # Update Q-Value using the Bellman Equation
            q_learner.step(obs, action, next_obs, reward, terminated)

            done = terminated or truncated
            obs = next_obs

        q_learner.update_epsilon(episode)

        if episode % 10000 == 0:
            print(f"Episode {episode} | Epsilon: {q_learner.epsilon:.2f}")

    print("Training finished.")

    # Visualize both scenarios
    plot_q_learning_policy(q_learner, usable_ace=True)
    plot_q_learning_policy(q_learner, usable_ace=False)

    if not config.play:
        return

    play_human_vs_agent(q_learner)  # ty: ignore


if __name__ == "__main__":
    tyro.cli(main)
