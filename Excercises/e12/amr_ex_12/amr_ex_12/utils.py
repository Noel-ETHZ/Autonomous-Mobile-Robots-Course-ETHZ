from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from amr_ex_12.e12_task_2_q_learning import QLearner, QLearningConfig


def plot_mdp_policy(V: np.ndarray, policy: np.ndarray, type: str):
    """
    V: optimal value function
    policy: optimal policy
    type: policy type to display in the title
    """
    V_grid = V.reshape(4, 4)
    policy_grid = policy.reshape(4, 4)

    arrows = {0: (-0.4, 0), 1: (0, -0.4), 2: (0.4, 0), 3: (0, 0.4)}

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(V_grid, cmap="YlGnBu")

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    for y in range(4):
        for x in range(4):
            action = policy_grid[y, x]
            dx, dy = arrows[action]
            ax.arrow(
                x,
                y,
                dx * 0.5,
                -dy * 0.5,
                head_width=0.1,
                head_length=0.1,
                fc="red",
                ec="red",
            )
            ax.text(
                x,
                y,
                f"{V_grid[y, x]:.2f}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=10,
            )
    ax.text(
        3,
        3,
        "GOAL",
        ha="center",
        va="center",
        weight="bold",
        color="white",
        backgroundcolor="green",
    )

    plt.title(f"{type}: Values and Actions")
    plt.colorbar(im, label="State Value")
    plt.show()


class RandomAgent:
    def __init__(self, config: QLearningConfig, action_space: gym.Space):
        self.action_space = action_space
        self.config = config

    def get_action(self, obs):
        return self.action_space.sample()


def play_human_vs_agent(q_learner: QLearner, num_human_games=10):
    env = gym.make("Blackjack-v1", render_mode="human")
    print("\n--- HUMAN CHALLENGE PHASE ---")
    human_wins = 0

    for i in range(num_human_games):
        obs, info = env.reset()
        done = False
        print(
            f"\nGame {i + 1}: Your Hand Sum: {obs[0]}, Dealer Shows: {obs[1]}, Usable Ace: {obs[2]}"
        )

        while not done:
            env.render()
            while True:
                action = input("Enter 1 to HIT or 0 to STICK: ")
                try:
                    action = int(action)
                    if action in [0, 1]:
                        break

                except ValueError:
                    print(f"invalid action: {action}")

            obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)  # stop annyoing warnings
            done = terminated or truncated

            if not done:
                print(
                    f"New Hand Sum: {obs[0]}, Dealer Shows: {obs[1]}, Usable Ace: {obs[2]}"
                )

        if reward > 0:
            print("Outcome: YOU WIN! 🎉")
            human_wins += 1
        elif reward < 0:
            print(f"Outcome: {obs[0]}, YOU BUST/LOSE. 💀")
        else:
            print("Outcome: PUSH (Tie). 🤝")

    # --- Agent Evaluation Phase ---
    print("\n--- AGENT EVALUATION PHASE ---")
    env = gym.make("Blackjack-v1")
    agent_wins = 0
    eval_episodes = 100_000

    q_learner.eval()

    for _ in range(eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            # Agent always picks the best action from Q-table
            action = q_learner.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if float(reward) > 0:
            agent_wins += 1

    print("\n--- RANDOM AGENT EVALUATION PHASE ---")
    ragent_wins = 0
    random_agent = RandomAgent(q_learner.config, env.action_space)

    for _ in range(eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            # Agent always picks the best action from Q-table
            action = random_agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if float(reward) > 0:
            ragent_wins += 1

    # Final Comparison
    print("\n" + "=" * 30)
    print(
        f"Human Win Rate: {(human_wins / num_human_games) * 100:.1f}% ({num_human_games} games)"
    )
    print(
        f"Agent Win Rate: {(agent_wins / eval_episodes) * 100:.1f}% ({eval_episodes} games)"
    )
    print(
        f"Random Agent Win Rate: {(ragent_wins / eval_episodes) * 100:.1f}% ({eval_episodes} games)"
    )
    print("=" * 30)


def plot_q_learning_policy(self, usable_ace=True):
    """Create a grid for the heatmap (Player Sum 12-21 vs Dealer Card 1-10)"""
    min_val = 12 if usable_ace else 4
    player_range = np.arange(min_val, 22)
    dealer_range = np.arange(1, 11)
    policy_grid = np.zeros((len(player_range), len(dealer_range)))

    for i, p_sum in enumerate(player_range):
        for j, d_card in enumerate(dealer_range):
            # Query the Q-table for the best action at this state
            state = (p_sum, d_card, usable_ace)
            policy_grid[i, j] = np.argmax(self.q_values[state])

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        policy_grid,
        annot=True,
        xticklabels=dealer_range,
        yticklabels=player_range,
        cmap="YlGnBu",
        cbar_kws={"label": "0: Stick, 1: Hit"},
    )
    plt.title(f"Blackjack Policy (Usable Ace: {usable_ace})")
    plt.xlabel("Dealer Showing Card")
    plt.ylabel("Player Current Sum")
    plt.show()


class QNetwork(nn.Module):
    """Q-Network class

    Implemented as an MLP that takes in a state and predicts a vector of Q values where
    entry i is associated with action i.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        # NOTE: This is really slow. you can speed this up by using a fixed lenght array (np or torch)
        # and an index to keep track of where you are in the buffer.
        self.buffer: deque[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of SARSA sequences from the buffer

        Args:
            batch_size: how many samples to return
            device: on which device the data should be returned

        Returns:
            State: current state (as torch Tensor)
            Action: the action that was taken (as torch Tensor)
            Reward: the reward that was observed (as torch Tensor)
            NextState: the state that was realized (as torch Tensor)
            Done: whehter the sequence terminated (as torch Tensor)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.LongTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.LongTensor(np.array(done)).to(device),
        )

    def __len__(self):
        return len(self.buffer)
