import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the DQN model architecture
class DQNModel:
    """
    Deep Q-Network (DQN) model for reinforcement learning.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the DQN model.

        Parameters:
            input_dim (int): Dimension of the input state vector.
            output_dim (int): Number of possible actions in the action space.
        """
        self.model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dense(64, activation='relu'),
            Dense(output_dim, activation='linear')  # Linear activation for Q-values
        ])
        self.model.compile(loss='mse', optimizer='adam')


def choose_action(action_space, current_state, dqn_model, epsilon):
    state_vector = current_state.get_state_vector()
    if np.random.rand() < epsilon:
        chosen_action = np.random.choice(action_space)
    else:
        q_values = dqn_model.model.predict(np.array([state_vector]))[0]
        chosen_action = action_space[np.argmax(q_values)]
    return chosen_action


def update_dqn_model(dqn_model, data_collection, action_space, gamma, batch_size, num_batches):
    for _ in range(num_batches):
        batch_indices = random.sample(range(len(data_collection)), batch_size)
        batch_data = [data_collection[i] for i in batch_indices]
        states = np.array([episode[0].get_state_vector() for episode in batch_data])
        actions = np.array([episode[1] for episode in batch_data])
        rewards = np.array([episode[2] for episode in batch_data])
        next_states = np.array([episode[3].get_state_vector() for episode in batch_data])
        q_values_current = dqn_model.model.predict(states)
        q_values_next = dqn_model.target_model.predict(next_states)
        targets = rewards + gamma * np.max(q_values_next, axis=1)

        for i in range(len(batch_data)):
            q_values_current[i][action_space.index(actions[i])] = targets[i]

        dqn_model.model.fit(states, q_values_current, batch_size=batch_size, epochs=1, verbose=0, learning_rate=0.0001)

    if num_simulations % target_update_frequency == 0:
        dqn_model.update_target_model()

