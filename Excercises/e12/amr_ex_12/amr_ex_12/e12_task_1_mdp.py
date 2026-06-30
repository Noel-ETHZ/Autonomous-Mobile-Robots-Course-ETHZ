from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro

from amr_ex_12.utils import plot_mdp_policy


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
                # TODO: Fill in the correct terms
                P[a, s, next_s] += ...
                R[a, s, next_s] += ...

    V = np.zeros(n_states)

    for i in range(cfg.num_value_iters):
        # Calculate Q-values using matrix form
        # TODO: Compute the Q function like you derived in task a.
        Q = np.einsum(..., P, R) + gamma * np.einsum(..., P, V)

        # Extract new optimal value function using Bellman optimality
        # TODO: Compute the new value function.
        new_V = ...

        if np.max(np.abs(new_V - V)) < cfg.value_iter_threshold:
            print(f"Converged in {i} iterations.")
            break
        V = new_V

    # TODO: Extract the optimal policy from the value function.
    optimal_policy = ...

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
                # TODO: Fill in the correct terms
                P[a, s, next_s] += ...
                R[a, s, next_s] += ...

    # Bootstrap policy and value function
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)

    for i in range(cfg.num_policy_iter):
        # TODO: Extract the policy conditioned transition probabilities and reward tensor
        P_pi = P[...]  # should be (n_states, n_states)
        R_pi = R[...]  # should be (n_states, n_states)

        # Compute the expected reward in state i
        # TODO: Complete the einsum operation.
        R_expected = np.einsum(..., P_pi, R_pi)

        # TODO: Solve the system of equations you derived in b.
        A = ...
        V = np.linalg.solve(A, R_expected)

        old_policy = policy.copy()

        # Calculate Q-values for all actions using the new V
        # TODO: Compute the Q function like you derived in task a using the value function.
        Q = np.einsum(..., P, R) + gamma * np.einsum(..., P, V)

        # TODO: Extect the optimal policy from the value function.
        optimal_policy = ...

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
