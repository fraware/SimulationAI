from state_representation import StateRepresentation
from rl_environment import RLEnvironment
from utils import get_random_action
import numpy as np
import tensorflow as tf
import mlflow
from dqn_model import DQNModel
from google_maps_api import get_coordinates, get_random_nearby_place, get_directions
import googlemaps
import matplotlib.pyplot as plt
from datetime import datetime
import random
from google_maps_api import get_coordinates, get_random_nearby_place, get_directions
from driving_directions import plot_map, turn_right, go_straight, turn_back, turn_left ,simulate_driving_right, simulate_driving_left, simulate_driving_alternate, parse_rule_choice, prepare_state_for_model, encode_instruction


# Initialize the DQN model

action_space = ['turn_left', 'turn_right', 'go_straight', 'turn_back']
input_dim = len(StateRepresentation([], [], []).get_state_vector())  # Initialize with empty state for dimension
output_dim = len(action_space)
dqn_model = DQNModel(input_dim, output_dim)
env = RLEnvironment(starting_address, max_intersections, rule_choice)
gamma = 0.95 # 0.99
target_update_frequency = 1000 # 5000
episodes_to_collect = 10  # Define the number of episodes to collect data before training

# Data Collection

data_collection = []
num_simulations = 100
epsilon = 0.1  # Epsilon value for epsilon-greedy exploration
batch_size = 32
num_batches = len(data_collection) // batch_size
num_test_simulations = 50
success_count = 0
total_time_to_destination = 0
path = []               # Store visited locations
rewards = []            # Store received rewards
actions = []            # Store taken actions
correct_actions = []    # Store correctness of actions

# Start the main experiment run
mlflow.set_experiment("Navigation Experiment")
mlflow.start_run()

# Log hyperparameters
mlflow.log_param("num_simulations", num_simulations)
mlflow.log_param("epsilon", epsilon)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("gamma", gamma)
mlflow.log_param("target_update_frequency", target_update_frequency)
mlflow.log_param("episodes_to_collect", episodes_to_collect)

for _ in range(num_simulations):

    # Reset the environment for a new simulation
    current_state = env.initialize_environment(starting_address)
    time_to_destination = 0

    while not env.is_episode_terminated():
        # Use Google Maps API to retrieve nearby streets and intersections
        nearby_streets = gmaps.places_nearby(location=current_state['current_address'], radius=100, type='route')
        current_state.nearby_streets = nearby_streets
        
        # Choose an action using the DQN model with epsilon-greedy exploration
        state_vector = current_state.get_state_vector()
        if np.random.rand() < epsilon:
            chosen_action = np.random.choice(action_space)  # Explore by choosing a random action
        else:
            q_values = dqn_model.model.predict(np.array([state_vector]))[0]
            chosen_action = action_space[np.argmax(q_values)]  # Exploit by choosing the action with highest Q-value
        
        # Calculate reward based on chosen action and previous instruction
        reward = env.calculate_reward(current_state['previous_instruction'], chosen_action)
        
        # Update the environment and get the next state
        next_state, reward, done = env.step(chosen_action)

        # Store data for training
        data_collection.append((current_state, chosen_action, reward, next_state))
        
        # Store the current state, chosen action, reward, and correctness for visualization
        path.append(current_state['current_location'])
        rewards.append(reward)
        actions.append(chosen_action)
        
        # Determine correctness of the action
        correct_action = chosen_action == current_state['previous_instruction']
        correct_actions.append(correct_action)

        # Update current state
        current_state = next_state
    
        # Increment time to destination
        time_to_destination += 1
        
        if done:
            if env.intersection_count >= env.max_intersections:
                success_count += 1
                total_time_to_destination += time_to_destination
            break

    # Train the DQN model using the collected data (if needed)
    if len(data_collection) >= episodes_to_collect:
        # Update the DQN model using online learning
        for _ in range(num_batches):
            batch_indices = random.sample(range(len(data_collection)), batch_size)
            batch_data = [data_collection[i] for i in batch_indices]
            
            states = np.array([episode[0].get_state_vector() for episode in batch_data])
            actions = np.array([episode[1] for episode in batch_data])
            rewards = np.array([episode[2] for episode in batch_data])
            next_states = np.array([episode[3].get_state_vector() for episode in batch_data])
            
            # Calculate Q-values for current and next states
            q_values_current = dqn_model.model.predict(states)
            q_values_next = dqn_model.target_model.predict(next_states)
            
            # Calculate target Q-values using Bellman equation
            targets = rewards + gamma * np.max(q_values_next, axis=1)
            
            # Update the Q-values of chosen actions
            for i in range(len(batch_data)):
                q_values_current[i][action_space.index(actions[i])] = targets[i]
            
            # Fine-tuning: Train the DQN model on the batch with a lower learning rate
            dqn_model.model.fit(states, q_values_current, batch_size=batch_size, epochs=1, verbose=0, learning_rate=0.0001)

        # Update the target network periodically
        if num_simulations % target_update_frequency == 0:
            dqn_model.update_target_model()

    # Visualize the agent's path, actions, rewards, and correctness after each simulation
    env.visualize_agent_path(path, rewards, actions, correct_actions)

    # Clear the lists for the next simulation
    path.clear()
    rewards.clear()
    actions.clear()
    correct_actions.clear()

# Calculate success rate and average time to destination
success_rate = success_count / num_test_simulations * 100
average_time_to_destination = total_time_to_destination / success_count if success_count > 0 else 0

# Print testing evaluation metrics
print("Testing Evaluation:")
print(f"Success Rate: {test_success_rate:.2f}%")
print(f"Average Time to Destination: {test_average_time_to_destination:.2f} time steps")

# Log metrics
mlflow.log_metric("success_rate", success_rate)
mlflow.log_metric("average_time_to_destination", average_time_to_destination)

# Save the trained model's weights
model_weights_file = r'C:\Users\mpetel\Documents\Simulation Project\trained_dqn_model.h5'
dqn_model.model.save_weights(model_weights_file)

# Save the trained model's architecture and weights using model.save()
model_save_path = r'C:\Users\mpetel\Documents\Simulation Project\trained_dqn_model'
dqn_model.model.save(model_save_path)

# Log the trained model's architecture and weights as an artifact
mlflow.log_artifact(model_save_path + '.h5')
mlflow.log_artifact(model_save_path + '.json')

# End the main experiment run
mlflow.end_run()


# Testing

# Load the trained model's architecture and weights using tf.keras.models.load_model()
loaded_model = tf.keras.models.load_model(model_save_path)

# Define a new instance of RLEnvironment for testing
test_env = RLEnvironment(test_starting_address, test_max_intersections, test_rule_choice)
num_test_simulations = 50
success_count = 0
total_time_to_destination = 0
path = []               # Store visited locations
rewards = []            # Store received rewards
actions = []            # Store taken actions
correct_actions = []    # Store correctness of actions

for _ in range(num_test_simulations):

    # Start a new experiment run for testing
    mlflow.start_run(run_name="Testing Run")

    # Reset the environment for a new test simulation
    current_state = test_env.initialize_environment(test_starting_address)
    time_to_destination = 0

    while not test_env.is_episode_terminated():
        # Use Google Maps API to retrieve nearby streets and intersections
        nearby_streets = gmaps.places_nearby(location=current_state['current_address'], radius=100, type='route')
        current_state.nearby_streets = nearby_streets
        
        # Choose an action using the trained DQN model
        state_vector = current_state.get_state_vector()
        q_values = dqn_model.model.predict(np.array([state_vector]))[0]
        chosen_action = action_space[np.argmax(q_values)]
        
        # Update the environment and get the next state
        next_state, reward, done = test_env.step(chosen_action)

        # Store the current state, chosen action, reward, and correctness for visualization
        path.append(current_state['current_location'])
        rewards.append(reward)
        actions.append(chosen_action)
        
        # Determine correctness of the action
        correct_action = chosen_action == current_state['previous_instruction']
        correct_actions.append(correct_action)

        # Update current state
        current_state = next_state
    
        # Increment time to destination
        time_to_destination += 1
        
        if done:
            if test_env.intersection_count >= test_env.max_intersections:
                success_count += 1
                total_time_to_destination += time_to_destination
            break

    # End the testing experiment run
    mlflow.end_run()

    # Visualize the agent's path, actions, rewards, and correctness after each simulation
    env.visualize_agent_path(path, rewards, actions, correct_actions)

    # Clear the lists for the next simulation
    path.clear()
    rewards.clear()
    actions.clear()
    correct_actions.clear()

# Calculate success rate and average time to destination for testing
test_success_rate = success_count / num_test_simulations * 100
test_average_time_to_destination = total_time_to_destination / success_count if success_count > 0 else 0

# Print testing evaluation metrics
print("Testing Evaluation:")
print(f"Success Rate: {test_success_rate:.2f}%")
print(f"Average Time to Destination: {test_average_time_to_destination:.2f} time steps")

# Log testing metrics
mlflow.log_metric("test_success_rate", test_success_rate)
mlflow.log_metric("test_average_time_to_destination", test_average_time_to_destination)


if __name__ == "__main__":
    # Replace 'starting_address' with your desired starting address
    starting_address = "1600 Amphitheatre Parkway, Mountain View, CA, 94043"

    # Get the user's choice of rules
    rule_choice = parse_rule_choice()

    # Load or initialize your machine learning model
    model = tf.keras.models.load_model(model_save_path)

    # Simulate the driving based on the chosen rule
    if rule_choice == 'right':
        simulate_driving_right(starting_address, max_intersections=100, model=model)
    elif rule_choice == 'left':
        simulate_driving_left(starting_address, max_intersections=100, model=model)
    elif rule_choice == 'alternate':
        simulate_driving_alternate(starting_address, max_intersections=100, model=model)
    else:
        print("Invalid rule choice.")