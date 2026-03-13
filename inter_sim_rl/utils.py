"""Utility helpers for the RL agent."""

import numpy as np


def get_random_action(action_space):
    """Randomly select an action from the given action space."""
    return np.random.choice(action_space)
