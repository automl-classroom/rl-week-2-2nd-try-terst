from __future__ import annotations

import gymnasium as gym
import numpy as np


class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(2)  # Two states: 0 and 1
        self.action_space = gym.spaces.Discrete(2)  # Two actions: 0 and 1
        self.state = 0  # Start state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0  # Always start in state 0
        return int(self.state), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise RuntimeError(f"Invalid action: {action}")
        # Reward is equal to the action taken, cast to float
        reward = float(action)
        self.state = action  # Transition to the state equal to the action
        terminated = False  # No terminal state in this simple example
        truncated = False
        return int(self.state), reward, terminated, truncated, {}

    def get_reward_per_action(self):
        return np.array([[0, 1], [0, 1]])  # Rewards for (state, action)

    def get_transition_matrix(self):
        T = np.zeros((2, 2, 2))  # (n_states, n_actions, n_states)
        T[0, 0, 0] = 1  # From state 0, action 0 -> state 0
        T[0, 1, 1] = 1  # From state 0, action 1 -> state 1
        T[1, 0, 0] = 1  # From state 1, action 0 -> state 0
        T[1, 1, 1] = 1  # From state 1, action 1 -> state 1
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.rng.random() < self.noise:
            obs = self.rng.integers(self.env.observation_space.n)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.rng.random() < self.noise:
            obs = self.rng.integers(self.env.observation_space.n)
        return obs, reward, terminated, truncated, info
