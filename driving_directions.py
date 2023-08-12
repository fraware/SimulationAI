from dqn_model import DQNModel
from state_representation import StateRepresentation
from rl_environment import RLEnvironment
import mlflow
import random
import numpy as np
import random
import matplotlib.pyplot as plt
import mlflow
from dqn_model import DQNModel
from rl_environment import RLEnvironment
from google_maps_api import get_coordinates, get_random_nearby_place, get_directions
import googlemaps
import matplotlib.pyplot as plt
from datetime import datetime
import random
from google_maps_api import get_coordinates, get_random_nearby_place, get_directions

def plot_map(latitudes, longitudes, starting_coords):
    """
    Plot the car's movement on a map using matplotlib.

    Parameters:
    - latitudes (list): List of latitude values representing the car's movement.
    - longitudes (list): List of longitude values representing the car's movement.
    - starting_coords (tuple): Tuple containing latitude and longitude of the starting point.

    Returns:
    None
    """
    # Plot the car's movement on a map using matplotlib
    plt.figure(figsize=(10, 8))
    plt.plot(longitudes, latitudes, 'bo-', markersize=8, label='Car Movement')
    plt.plot(starting_coords[1], starting_coords[0], 'ro', markersize=10, label='Starting Point')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Driving Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

def turn_right(instruction):
    """
    Modify the instruction to simulate turning right at an intersection.

    Parameters:
    - instruction (str): Original instruction containing directions.

    Returns:
    str: Modified instruction indicating turning right.
    """
    # Function to modify the instruction to simulate turning right at an intersection
    if 'right' in instruction.lower():
        return instruction
    else:
        return f"Turn right to {instruction.split('on')[1]}"

def turn_left(instruction):
    """
    Modify the instruction to simulate turning left at an intersection.

    Parameters:
    - instruction (str): Original instruction containing directions.

    Returns:
    str: Modified instruction indicating turning left.
    """
    # Function to modify the instruction to simulate turning left at an intersection
    if 'left' in instruction.lower():
        return instruction
    else:
        return f"Turn left to {instruction.split('on')[1]}"

def go_straight(instruction):
    """
    Modify the instruction to simulate going straight at an intersection.

    Parameters:
    - instruction (str): Original instruction containing directions.

    Returns:
    str: Modified instruction indicating going straight.
    """
    # Function to modify the instruction to simulate going straight at an intersection
    if 'straight' in instruction.lower():
        return instruction
    else:
        return f"Go straight on {instruction.split('on')[1]}"

def turn_back(instruction):
    """
    Modify the instruction to simulate turning back at an intersection.

    Parameters:
    - instruction (str): Original instruction containing directions.

    Returns:
    str: Modified instruction indicating turning back.
    """
    # Function to modify the instruction to simulate turning back at an intersection
    if 'turn around' in instruction.lower():
        return instruction
    else:
        return f"Turn around on {instruction.split('on')[1]}"


def encode_instruction(instruction):
    """
    Encode an instruction into a numerical representation.

    Parameters:
    - instruction (str): The instruction to be encoded.

    Returns:
    - encoded_instruction (np.ndarray): The encoded instruction as a NumPy array.
    """
    # Define a dictionary that maps instruction types to numerical values
    instruction_mapping = {
        'turn_left': 0,
        'turn_right': 1,
        'go_straight': 2,
        'turn_back': 3
    }

    # Convert the instruction to lowercase and get its numerical value from the mapping
    encoded_value = instruction_mapping.get(instruction.lower(), -1)  # -1 for unknown instructions

    # Create a one-hot encoded array based on the numerical value
    encoded_instruction = np.zeros(len(instruction_mapping))
    if encoded_value >= 0:
        encoded_instruction[encoded_value] = 1

    return encoded_instruction

def prepare_state_for_model(current_address, next_address, instruction):
    """
    Prepare the current state for the machine learning model.

    Parameters:
    - current_address (str): The current address.
    - next_address (str): The next address.
    - instruction (str): The instruction for the next step.

    Returns:
    - state (np.ndarray): The prepared state as a NumPy array.
    """
    # Convert addresses to coordinates
    current_coords = get_coordinates(current_address)
    next_coords = get_coordinates(next_address)

    # Encode instruction (e.g., one-hot encoding)
    encoded_instruction = encode_instruction(instruction)

    # Combine all components into the state vector
    state = np.concatenate((current_coords, next_coords, encoded_instruction))

    return state

def simulate_driving_right(starting_address, max_intersections=100, model=None):
    """
    Simulate driving with the right-turning rule.

    Parameters:
    - starting_address (str): Starting address for the driving simulation.
    - max_intersections (int): Maximum number of intersections to simulate (default: 100).

    Returns:
    None
    """
    # Simulate driving with the right-turning rule
    intersection_count = 0
    car_positions = []
    current_address = starting_address

    while intersection_count < max_intersections:
        # Get the nearby places or streets from the current address
        next_address = get_random_nearby_place(current_address)

        if next_address:
            # Get driving directions between the current address and the next address
            driving_steps = get_directions(current_address, next_address)

            if driving_steps:
                # Retrieve step instructions and duration for the first step
                instruction = driving_steps[0]['html_instructions']
                duration = driving_steps[0]['duration']['text']

                # Call the machine learning model to get an action
                if model:
                    current_state = prepare_state_for_model(current_address, next_address, instruction)
                    action = model.predict(current_state)

                    # Decide based on the model's prediction
                    if action == 'right':
                        print(f"Model predicts: Turn right at intersection.")
                    elif action == 'go_straight':
                        updated_instruction = go_straight(instruction)
                        print(f"Model predicts: {updated_instruction}")
                    elif action == 'turn_back':
                        updated_instruction = turn_back(instruction)
                        print(f"Model predicts: {updated_instruction}")

                # Check if the instruction contains the word 'right' to detect an intersection
                if 'right' in instruction.lower():
                    intersection_count += 1
                    print(f"Intersection {intersection_count}: {instruction} ({duration})")
                else:
                    updated_instruction = turn_right(instruction)
                    print(f"Step: {updated_instruction} ({duration})")

                # Get latitude and longitude of the next address for plotting
                next_address_coords = get_coordinates(next_address)
                if next_address_coords:
                    car_positions.append(next_address_coords)

                # Update the current address for the next iteration
                current_address = next_address

            else:
                print(f"Failed to get driving directions from {current_address} to {next_address}.")
                break

        else:
            print("No nearby places found. Ending simulation.")
            break

    # Plot the car's movement on a map
    latitudes, longitudes = zip(*car_positions)
    plot_map(latitudes, longitudes, get_coordinates(starting_address))

def simulate_driving_left(starting_address, max_intersections=100, model=None):
    """
    Simulate driving with the left-turning rule.

    Parameters:
    - starting_address (str): Starting address for the driving simulation.
    - max_intersections (int): Maximum number of intersections to simulate (default: 100).

    Returns:
    None
    """
    # Simulate driving with the left-turning rule
    intersection_count = 0
    car_positions = []
    current_address = starting_address

    while intersection_count < max_intersections:
        # Get the nearby places or streets from the current address
        next_address = get_random_nearby_place(current_address)

        if next_address:
            # Get driving directions between the current address and the next address
            driving_steps = get_directions(current_address, next_address)

            if driving_steps:
                # Retrieve step instructions and duration for the first step
                instruction = driving_steps[0]['html_instructions']
                duration = driving_steps[0]['duration']['text']

                # Call the machine learning model to get an action
                if model:
                    current_state = prepare_state_for_model(current_address, next_address, instruction)
                    action = model.predict(current_state)

                    # Decide based on the model's prediction
                    if action == 'left':
                        print(f"Model predicts: Turn left at intersection.")
                    elif action == 'go_straight':
                        updated_instruction = go_straight(instruction)
                        print(f"Model predicts: {updated_instruction}")
                    elif action == 'turn_back':
                        updated_instruction = turn_back(instruction)
                        print(f"Model predicts: {updated_instruction}")

                # Check if the instruction contains the word 'left' to detect an intersection
                if 'left' in instruction.lower():
                    intersection_count += 1
                    print(f"Intersection {intersection_count}: {instruction} ({duration})")
                else:
                    updated_instruction = turn_left(instruction)
                    print(f"Step: {updated_instruction} ({duration})")

                # Get latitude and longitude of the next address for plotting
                next_address_coords = get_coordinates(next_address)
                if next_address_coords:
                    car_positions.append(next_address_coords)

                # Update the current address for the next iteration
                current_address = next_address

            else:
                print(f"Failed to get driving directions from {current_address} to {next_address}.")
                break

        else:
            print("No nearby places found. Ending simulation.")
            break

    # Plot the car's movement on a map
    latitudes, longitudes = zip(*car_positions)
    plot_map(latitudes, longitudes, get_coordinates(starting_address))

def simulate_driving_alternate(starting_address, max_intersections=100, model=None):
    """
    Simulate driving with the alternate turning rule (turn right twice, then turn left once, repeat).

    Parameters:
    - starting_address (str): Starting address for the driving simulation.
    - max_intersections (int): Maximum number of intersections to simulate (default: 100).

    Returns:
    None
    """
    # Simulate driving with the alternate turning rule (turn right twice, then turn left once, repeat)
    intersection_count = 0
    car_positions = []
    current_address = starting_address

    while intersection_count < max_intersections:
        # Get the nearby places or streets from the current address
        next_address = get_random_nearby_place(current_address)

        if next_address:
            # Get driving directions between the current address and the next address
            driving_steps = get_directions(current_address, next_address)

            if driving_steps:
                # Retrieve step instructions and duration for the first step
                instruction = driving_steps[0]['html_instructions']
                duration = driving_steps[0]['duration']['text']

                # Call the machine learning model to get an action
                if model:
                    current_state = prepare_state_for_model(current_address, next_address, instruction)
                    action = model.predict(current_state)

                    # Decide based on the model's prediction
                    if action == 'right':
                        print(f"Model predicts: Turn right at intersection.")
                    elif action == 'left':
                        print(f"Model predicts: Turn left at intersection.")
                    elif action == 'go_straight':
                        updated_instruction = go_straight(instruction)
                        print(f"Model predicts: {updated_instruction}")
                    elif action == 'turn_back':
                        updated_instruction = turn_back(instruction)
                        print(f"Model predicts: {updated_instruction}")

                # Define the turn sequence: right, right, left
                turn_sequence = ['right', 'right', 'left']

                # Check if the instruction contains the word from the turn sequence
                if any(turn in instruction.lower() for turn in turn_sequence):
                    intersection_count += 1
                    if turn_sequence[turn_count % 3] in instruction.lower():
                        print(f"Intersection {intersection_count}: {instruction} ({duration})")
                        turn_count += 1
                    else:
                        updated_instruction = f"Turn {turn_sequence[turn_count % 3]} to {instruction.split('on')[1]}"
                        print(f"Step: {updated_instruction} ({duration})")
                        turn_count += 1
                else:
                    updated_instruction = turn_right(instruction)
                    print(f"Step: {updated_instruction} ({duration})")

                # Get latitude and longitude of the next address for plotting
                next_address_coords = get_coordinates(next_address)
                if next_address_coords:
                    car_positions.append(next_address_coords)

                # Update the current address for the next iteration
                current_address = next_address

            else:
                print(f"Failed to get driving directions from {current_address} to {next_address}.")
                break

        else:
            print("No nearby places found. Ending simulation.")
            break

    # Plot the car's movement on a map
    latitudes, longitudes = zip(*car_positions)
    plot_map(latitudes, longitudes, get_coordinates(starting_address))

def parse_rule_choice():
    """
    Parser function to get the user's choice of rules.

    Returns:
    str: Choice of rules ('right', 'left', or 'alternate').
    """
    # Parser function to get the user's choice of rules
    while True:
        print("Choose a set of rules:")
        print("1. Right-turning at intersections (Default)")
        print("2. Left-turning at intersections")
        print("3. Turn right twice, then turn left once, repeat")
        choice = input("Enter the number of your choice (1, 2, or 3): ")

        if choice == '1':
            return 'right'
        elif choice == '2':
            return 'left'
        elif choice == '3':
            return 'alternate'
        else:
            print("Invalid choice. Please enter '1', '2', or '3'.")
