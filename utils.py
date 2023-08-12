import random
import numpy as np

def get_random_action(action_space):
    """
    Randomly selects an action from the given action space.

    Parameters:
    - action_space (list or array-like): List of available actions.

    Returns:
    Any: A randomly selected action from the action space.
    """
    return np.random.choice(action_space)