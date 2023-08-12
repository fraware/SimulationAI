import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import googlemaps
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow

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
