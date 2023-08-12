import numpy as np
import matplotlib.pyplot as plt

class StateRepresentation:
    """
    Represents the current state in the reinforcement learning environment.
    """
    def __init__(self, current_location, current_direction, nearby_streets):
        """
        Initialize the state representation.

        Parameters:
            current_location (tuple): Current GPS coordinates.
            current_direction (tuple): Current direction vector.
            nearby_streets (list): List of nearby street information.
        """
        self.current_location = current_location
        self.current_direction = current_direction
        self.nearby_streets = nearby_streets

    def get_state_vector(self):
        """
        Convert the state into a feature vector.

        Returns:
            np.ndarray: Feature vector representing the state.
        """
        location_vector = np.array([self.current_location[0], self.current_location[1]])
        direction_vector = np.array([self.current_direction[0], self.current_direction[1]])
        nearby_streets_vector = np.array(self.nearby_streets)

        state_vector = np.concatenate((location_vector, direction_vector, nearby_streets_vector))
        return state_vector


class RLEnvironment:
    """
    Reinforcement learning environment for navigation.
    """

    def __init__(self, starting_address, max_intersections, rule_choice):
        """
        Initialize the RL environment.

        Parameters:
            starting_address (str): Starting address for navigation.
            max_intersections (int): Maximum number of intersections to navigate.
            rule_choice (str): Choice of navigation rule ('right', 'left', 'alternate').
        """
        self.starting_address = starting_address
        self.max_intersections = max_intersections
        self.rule_choice = rule_choice
        self.intersection_count = 0
        self.current_state = self.initialize_environment(starting_address)
        self.action_space = ['turn_left', 'turn_right', 'go_straight', 'turn_back']

    def initialize_environment(self, current_address):
        """
        Initialize the environment state.

        Parameters:
            current_address (str): Current address for navigation.

        Returns:
            dict: Initialized state of the environment.
        """
        current_state = {
            'current_address': current_address,
            'previous_instruction': None  # Initialize as None
            # Add other relevant state information here
        }
        return current_state
    
    def is_episode_terminated(self):
        """
        Check if the episode is terminated.

        Returns:
            bool: True if episode is terminated, False otherwise.
        """
        return self.intersection_count >= self.max_intersections
    
    def step(self, chosen_action):
        """
        Perform a step in the environment based on the chosen action.

        Parameters:
            chosen_action (str): Chosen action to take.

        Returns:
            tuple: Next state, reward, and episode termination flag.
        """
        # Calculate reward based on chosen action and previous instruction
        reward = self.calculate_reward(self.current_state['previous_instruction'], chosen_action)
        
        # Update the environment and get the next state
        next_state = self.update_environment(chosen_action)
        
        # Update intersection count
        if 'turn' in chosen_action:
            self.intersection_count += 1
        
        # Check if the episode is terminated
        done = self.is_episode_terminated()
        
        return next_state, reward, done
    
    def get_observation(self):
        """
        Get the current observation (state) of the environment.

        Returns:
            dict: Current state of the environment.
        """
        return self.current_state

    def visualize_agent_path(self, path, rewards, actions, correct_actions):
        """
        Visualize the agent's path, actions, rewards, and correctness of actions.

        Parameters:
            path (list): List of visited locations.
            rewards (list): List of rewards received at each step.
            actions (list): List of actions taken at each step.
            correct_actions (list): List of boolean values indicating correctness of actions.
        """
        plt.figure(figsize=(12, 8))
        
        plt.plot(*zip(*path), marker='o', color='b', label='Visited Locations')
        
        for i, txt in enumerate(path):
            action = actions[i]
            reward = rewards[i]
            correct = correct_actions[i]

            # Determine color based on correctness of action
            color = 'g' if correct else 'r'

            plt.annotate(f"Step {i+1}\nAction: {action}\nReward: {reward:.2f}", (path[i]), textcoords="offset points", xytext=(-15, 10), ha='center', color=color)

        plt.title("Agent's Path, Actions, Rewards, and Correctness")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.legend()
        plt.grid()
        plt.show()
    
    def calculate_reward(self, previous_instruction, chosen_action):
        """
        Calculate the reward for a chosen action based on the previous instruction.

        Parameters:
            previous_instruction (str): Previous navigation instruction.
            chosen_action (str): Chosen action to take.

        Returns:
            float: Calculated reward value.
        """
        # Reward for taking the right turn at intersections
        if self.rule_choice == 'right':
            if 'right' in previous_instruction.lower() and chosen_action == 'turn_right':
                return 1.0  # Positive reward for correct turn
            elif 'right' in previous_instruction.lower():
                return -0.5  # Negative reward for incorrect action after right turn
            else:
                return 0.0  # Neutral reward for other actions
        
        # Reward for taking the left turn at intersections
        elif self.rule_choice == 'left':
            if 'left' in previous_instruction.lower() and chosen_action == 'turn_left':
                return 1.0  # Positive reward for correct turn
            elif 'left' in previous_instruction.lower():
                return -0.5  # Negative reward for incorrect action after left turn
            else:
                return 0.0  # Neutral reward for other actions
        
        # Reward for alternating turns at intersections
        elif self.rule_choice == 'alternate':
            if 'left' in previous_instruction.lower() and chosen_action == 'turn_right':
                return 1.0  # Positive reward for alternating turn
            elif 'right' in previous_instruction.lower() and chosen_action == 'turn_left':
                return 1.0  # Positive reward for alternating turn
            elif 'left' in previous_instruction.lower() or 'right' in previous_instruction.lower():
                return -0.5  # Negative reward for incorrect action after alternate turn
            else:
                return 0.0  # Neutral reward for other actions

        def update_environment(self, chosen_action):
            # Implement how the environment is updated based on chosen action
            # For example, update the current state based on chosen_action
            # Update current state before returning the next state
            self.current_state = next_state
            return next_state
