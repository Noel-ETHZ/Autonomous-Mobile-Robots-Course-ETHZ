from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro

from amr_ex_12_sol.utils import plot_mdp_policy


@dataclass
class Config:
    env_id: str = "FrozenLake-v1"
    """Gzm env id"""
    env_slip_prob: float = 0.4
    """Probability that the agent will execute the commanded action"""
    gamma: float = 0.99
    """Discount factor"""
    num_value_iters: int = 1000
    value_iter_threshold: float = 1e-8

    num_policy_iter: int = 100


def value_iteration(cfg: Config):
    env = gym.make(cfg.env_id, is_slippery=True, success_rate=cfg.env_slip_prob)
    n_states = env.observation_space.n  # ty: ignore
    n_actions = env.action_space.n  # ty: ignore
    gamma = cfg.gamma

    P = np.zeros((n_actions, n_states, n_states))
    """Reward tensor with shape (actions, states, states)"""
    R = np.zeros((n_actions, n_states, n_states))
    """Reward tensor with shape (actions, states, states)"""

    for s in range(n_states):
        for a in range(n_actions):
            # transitions is a list of (prob, next_s, reward, done)
            transitions = env.unwrapped.P[s][a]  # ty: ignore
            for prob, next_s, reward, done in transitions:
                P[a, s, next_s] += prob
                R[a, s, next_s] += reward

    V = np.zeros(n_states)

    for i in range(cfg.num_value_iters):
        # Calculate Q-values using matrix form
        Q = np.einsum("ijk,ijk->ij", P, R) + gamma * np.einsum("ijk,k->ij", P, V)

        # Extract new optimal value function using Bellman optimality
        new_V = np.max(Q, axis=0)

        if np.max(np.abs(new_V - V)) < cfg.value_iter_threshold:
            print(f"Converged in {i} iterations.")
            break
        V = new_V

    # Extract the optimal policy
    optimal_policy = np.argmax(Q, axis=0)

    return V, optimal_policy


def policy_iteration(cfg: Config):
    env = gym.make(cfg.env_id, is_slippery=True, success_rate=cfg.env_slip_prob)
    n_states = env.observation_space.n  # ty: ignore
    n_actions = env.action_space.n  # ty: ignore
    gamma = cfg.gamma

    P = np.zeros((n_actions, n_states, n_states))
    """Reward tensor with shape (actions, states, states)"""
    R = np.zeros((n_actions, n_states, n_states))
    """Reward tensor with shape (actions, states, states)"""

    for s in range(n_states):
        for a in range(n_actions):
            # transitions is a list of (prob, next_s, reward, done)
            transitions = env.unwrapped.P[s][a]  # ty: ignore
            for prob, next_s, reward, done in transitions:
                P[a, s, next_s] += prob
                R[a, s, next_s] += reward

    # Bootstrap policy and value function
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)

    for i in range(cfg.num_policy_iter):
        # Extract policy conditioned transition probabilities and rewards
        P_pi = P[policy, np.arange(n_states), :]  # should be (n_states, n_states)
        R_pi = R[policy, np.arange(n_states), :]  # should be (n_states, n_states)

        # Compute the expected reward in state i
        R_expected = np.einsum("ij,ij->i", P_pi, R_pi)

        # Solve (I - gamma * P_pi)V = R_pi for V
        A = np.eye(n_states) - gamma * P_pi
        V = np.linalg.solve(A, R_expected)

        old_policy = policy.copy()

        # Calculate Q-values for all actions using the new V
        Q = np.einsum("ijk,ijk->ij", P, R) + gamma * (P @ V)

        # New policy is the greedy action
        policy = np.argmax(Q, axis=0)

        # Check for convergence: if policy hasn't changed, we're done
        if np.array_equal(policy, old_policy):
            print(f"Policy Iteration converged in {i} iterations.")
            break

    return V, policy


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    V_vi, policy = value_iteration(cfg)
    plot_mdp_policy(V_vi, policy, "Value iteration")
    V_pi, policy = policy_iteration(cfg)
    plot_mdp_policy(V_pi, policy, "Policy iteration")

    assert np.allclose(V_vi, V_pi), "Your Value functions differ between VI and PI."
