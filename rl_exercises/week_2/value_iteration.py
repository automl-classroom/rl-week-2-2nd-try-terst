from __future__ import annotations

from typing import Any, Tuple

import gymnasium
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class ValueIteration(AbstractAgent):
    """Agent that computes an optimal policy via Value Iteration.

    Parameters
    ----------
    env : MarsRover or gymnasium.Env
        The target environment, must expose `.states`, `.actions`,
        `.transition_matrix` and `.get_reward_per_action()`.
    gamma : float, default=0.9
        Discount factor for future rewards.
    seed : int, default=333
        Random seed for tie‐breaking among equally‐good actions.

    Attributes
    ----------
    V : np.ndarray, shape (n_states,)
        The computed optimal value function.
    pi : np.ndarray, shape (n_states,)
        The greedy policy derived from V.
    policy_fitted : bool
        Whether value iteration has been run yet.
    """

    def __init__(
        self,
        env: MarsRover | gymnasium.Env,
        gamma: float = 0.9,
        seed: int = 333,
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore
        super().__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.seed = seed

        # TODO: Extract MDP components from the environment
        self.S = None
        self.A = None
        self.T = None
        self.R_sa = None
        self.n_states = None
        self.n_actions = None

        self.n_obs = self.env.observation_space.n  # Number of observations/states
        self.n_actions = self.env.action_space.n  # Number of actions

        # placeholders
        self.V = np.zeros(self.n_states, dtype=float)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.policy_fitted = False

    def update_agent(self, *args: tuple, **kwargs: dict) -> None:
        """Run value iteration to compute the optimal policy and state-action values."""
        if not self.policy_fitted:
            # Initialize MDP components
            self.S = np.arange(self.n_obs)
            self.A = np.arange(self.n_actions)
            self.T = self.env.get_transition_matrix()
            self.R_sa = self.env.get_reward_per_action()

            # Run value iteration
            self.V, self.pi = value_iteration(
                T=self.T,
                R_sa=self.R_sa,
                gamma=self.gamma,
                seed=self.seed,
            )

            self.policy_fitted = True

    def predict_action(
        self,
        observation: int,
        info: dict | None = None,
        evaluate: bool = False,
    ) -> tuple[int, dict]:
        """Choose action = π(observation). Runs update if needed."""
        if not self.policy_fitted:
            self.update_agent()

        # Select action based on the learned policy
        action = self.pi[observation]
        return action, {}


def value_iteration(
    *,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    seed: int | None = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Value Iteration on a finite MDP.

    Solves for
        V*(s) = max_a [ R_sa[s,a] + γ ∑_{s'} T[s,a,s'] V*(s') ]
    and then
        π*(s) = argmax_a [ R_sa[s,a] + γ ∑_{s'} T[s,a,s'] V*(s') ].

    Parameters
    ----------
    T : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R_sa : np.ndarray, shape (n_states, n_actions)
        Rewards for each (state, action).
    gamma : float
        Discount factor (0 ≤ γ < 1).
    seed : int or None
        RNG seed for tie‐breaking among equal actions.
    epsilon : float
        Stopping threshold on max value‐update difference.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal state‐value function.
    pi : np.ndarray, shape (n_states,)
        Greedy policy w.r.t. V, with random tie‐breaking.
    """
    n_states, n_actions = R_sa.shape
    V = np.zeros(n_states, dtype=float)
    rng = np.random.default_rng(seed)
    pi = np.zeros(n_states, dtype=int)

    while True:
        delta = 0
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = R_sa[s, a] + gamma * np.sum(T[s, a, :] * V)
            max_q = np.max(q_values)
            delta = max(delta, abs(max_q - V[s]))
            V[s] = max_q
            pi[s] = rng.choice(np.flatnonzero(q_values == max_q))

        if delta < epsilon:
            break

    return V, pi
