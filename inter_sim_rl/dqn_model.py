"""
DQN model and target network for reinforcement learning.
"""

from typing import Any, List

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential


class DQNModel:
    """
    Deep Q-Network (DQN) model with target network for reinforcement learning.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the DQN model and target model.

        Parameters:
            input_dim: Dimension of the input state vector.
            output_dim: Number of possible actions in the action space.
        """
        self.model = self._build_model(input_dim, output_dim)
        self.target_model = self._build_model(input_dim, output_dim)
        self.update_target_model()

    def _build_model(self, input_dim: int, output_dim: int) -> Model:
        """Build the neural network architecture."""
        model = Sequential(
            [
                Dense(128, activation="relu", input_dim=input_dim),
                Dense(64, activation="relu"),
                Dense(output_dim, activation="linear"),
            ]
        )
        model.compile(loss="mse", optimizer="adam")
        return model

    def update_target_model(self) -> None:
        """Copy weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())


def choose_action(
    action_space: List[str],
    current_state: Any,
    dqn_model: DQNModel,
    epsilon: float,
) -> str:
    """Choose an action using epsilon-greedy policy."""
    state_vector = current_state.get_state_vector()
    if np.random.rand() < epsilon:
        return str(np.random.choice(action_space))
    q_values = dqn_model.model.predict(np.array([state_vector]), verbose=0)[0]
    return action_space[int(np.argmax(q_values))]
