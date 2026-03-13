"""
Driving simulation and rule-based turn helpers for the demo.
"""

import matplotlib.pyplot as plt
import numpy as np

from inter_sim_rl.google_maps_api import (
    get_coordinates,
    get_directions,
    get_random_nearby_place,
)
from inter_sim_rl.state_representation import StateRepresentation

# Action space aligned with main and rl_environment
DEFAULT_ACTION_SPACE = ["turn_left", "turn_right", "go_straight", "turn_back"]


def plot_map(latitudes, longitudes, starting_coords):
    """Plot the car's movement on a map using matplotlib."""
    plt.figure(figsize=(10, 8))
    plt.plot(longitudes, latitudes, "bo-", markersize=8, label="Car Movement")
    plt.plot(
        starting_coords[1],
        starting_coords[0],
        "ro",
        markersize=10,
        label="Starting Point",
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Driving Simulation")
    plt.legend()
    plt.grid(True)
    plt.show()


def turn_right(instruction):
    """Modify the instruction to simulate turning right at an intersection."""
    if "right" in instruction.lower():
        return instruction
    return f"Turn right to {instruction.split('on')[1]}"


def turn_left(instruction):
    """Modify the instruction to simulate turning left at an intersection."""
    if "left" in instruction.lower():
        return instruction
    return f"Turn left to {instruction.split('on')[1]}"


def go_straight(instruction):
    """Modify the instruction to simulate going straight."""
    if "straight" in instruction.lower():
        return instruction
    return f"Go straight on {instruction.split('on')[1]}"


def turn_back(instruction):
    """Modify the instruction to simulate turning back."""
    if "turn around" in instruction.lower():
        return instruction
    return f"Turn around on {instruction.split('on')[1]}"


def encode_instruction(instruction):
    """Encode an instruction into a numerical representation (one-hot)."""
    instruction_mapping = {
        "turn_left": 0,
        "turn_right": 1,
        "go_straight": 2,
        "turn_back": 3,
    }
    encoded_value = instruction_mapping.get(instruction.lower(), -1)
    encoded_instruction = np.zeros(len(instruction_mapping))
    if encoded_value >= 0:
        encoded_instruction[encoded_value] = 1
    return encoded_instruction


def prepare_state_for_model(current_address, next_address, instruction):
    """
    Prepare state for the model (legacy vector format).
    For demo with trained DQN, use state_vector_for_model() so input dim matches.
    """
    current_coords = get_coordinates(current_address)
    next_coords = get_coordinates(next_address)
    encoded_instruction = encode_instruction(instruction)
    state = np.concatenate(
        (current_coords or (0, 0), next_coords or (0, 0), encoded_instruction)
    )
    return state


def state_vector_for_model(current_address, instruction, current_coords=None):
    """Build state vector matching StateRepresentation for the trained DQN."""
    if current_coords is None:
        current_coords = get_coordinates(current_address) or (0.0, 0.0)
    state = StateRepresentation(
        current_location=current_coords,
        current_direction=(0.0, 0.0),
        nearby_streets=[],
        current_address=current_address or "",
        previous_instruction=instruction,
    )
    return state.get_state_vector()


def simulate_driving_right(
    starting_address, max_intersections=100, model=None, action_space=None
):
    """Simulate driving with the right-turning rule."""
    actions = action_space or DEFAULT_ACTION_SPACE
    intersection_count = 0
    car_positions = []
    current_address = starting_address

    while intersection_count < max_intersections:
        next_address = get_random_nearby_place(current_address)
        if not next_address:
            print("No nearby places found. Ending simulation.")
            break
        driving_steps = get_directions(current_address, next_address)
        if not driving_steps:
            print(
                f"Failed to get driving directions from {current_address} "
                f"to {next_address}."
            )
            break
        instruction = driving_steps[0]["html_instructions"]
        duration = driving_steps[0]["duration"]["text"]

        if model:
            state_vec = state_vector_for_model(current_address, instruction)
            batch = np.array([state_vec])
            q_values = model.predict(batch, verbose=0)[0]
            action = actions[int(np.argmax(q_values))]
            if action == "turn_right":
                print("Model predicts: Turn right at intersection.")
            elif action == "go_straight":
                print(f"Model predicts: {go_straight(instruction)}")
            elif action == "turn_back":
                print(f"Model predicts: {turn_back(instruction)}")
            else:
                print(f"Model predicts: {action}")

        if "right" in instruction.lower():
            intersection_count += 1
            print(f"Intersection {intersection_count}: {instruction} ({duration})")
        else:
            print(f"Step: {turn_right(instruction)} ({duration})")

        next_address_coords = get_coordinates(next_address)
        if next_address_coords:
            car_positions.append(next_address_coords)
        current_address = next_address

    if car_positions:
        start_coords = get_coordinates(starting_address)
        if start_coords:
            latitudes, longitudes = zip(*car_positions)
            plot_map(latitudes, longitudes, start_coords)


def simulate_driving_left(
    starting_address, max_intersections=100, model=None, action_space=None
):
    """Simulate driving with the left-turning rule."""
    actions = action_space or DEFAULT_ACTION_SPACE
    intersection_count = 0
    car_positions = []
    current_address = starting_address

    while intersection_count < max_intersections:
        next_address = get_random_nearby_place(current_address)
        if not next_address:
            print("No nearby places found. Ending simulation.")
            break
        driving_steps = get_directions(current_address, next_address)
        if not driving_steps:
            print(
                f"Failed to get driving directions from {current_address} "
                f"to {next_address}."
            )
            break
        instruction = driving_steps[0]["html_instructions"]
        duration = driving_steps[0]["duration"]["text"]

        if model:
            state_vec = state_vector_for_model(current_address, instruction)
            batch = np.array([state_vec])
            q_values = model.predict(batch, verbose=0)[0]
            action = actions[int(np.argmax(q_values))]
            if action == "turn_left":
                print("Model predicts: Turn left at intersection.")
            elif action == "go_straight":
                print(f"Model predicts: {go_straight(instruction)}")
            elif action == "turn_back":
                print(f"Model predicts: {turn_back(instruction)}")
            else:
                print(f"Model predicts: {action}")

        if "left" in instruction.lower():
            intersection_count += 1
            print(f"Intersection {intersection_count}: {instruction} ({duration})")
        else:
            print(f"Step: {turn_left(instruction)} ({duration})")

        next_address_coords = get_coordinates(next_address)
        if next_address_coords:
            car_positions.append(next_address_coords)
        current_address = next_address

    if car_positions:
        start_coords = get_coordinates(starting_address)
        if start_coords:
            latitudes, longitudes = zip(*car_positions)
            plot_map(latitudes, longitudes, start_coords)


def simulate_driving_alternate(
    starting_address, max_intersections=100, model=None, action_space=None
):
    """Simulate driving with the alternate rule (right, right, left, repeat)."""
    actions = action_space or DEFAULT_ACTION_SPACE
    intersection_count = 0
    car_positions = []
    current_address = starting_address
    turn_count = 0
    turn_sequence = ["right", "right", "left"]

    while intersection_count < max_intersections:
        next_address = get_random_nearby_place(current_address)
        if not next_address:
            print("No nearby places found. Ending simulation.")
            break
        driving_steps = get_directions(current_address, next_address)
        if not driving_steps:
            print(
                f"Failed to get driving directions from {current_address} "
                f"to {next_address}."
            )
            break
        instruction = driving_steps[0]["html_instructions"]
        duration = driving_steps[0]["duration"]["text"]

        if model:
            state_vec = state_vector_for_model(current_address, instruction)
            batch = np.array([state_vec])
            q_values = model.predict(batch, verbose=0)[0]
            action = actions[int(np.argmax(q_values))]
            if action == "turn_right":
                print("Model predicts: Turn right at intersection.")
            elif action == "turn_left":
                print("Model predicts: Turn left at intersection.")
            elif action == "go_straight":
                print(f"Model predicts: {go_straight(instruction)}")
            elif action == "turn_back":
                print(f"Model predicts: {turn_back(instruction)}")
            else:
                print(f"Model predicts: {action}")

        if any(turn in instruction.lower() for turn in turn_sequence):
            intersection_count += 1
            expected = turn_sequence[turn_count % 3]
            if expected in instruction.lower():
                print(f"Intersection {intersection_count}: {instruction} ({duration})")
            else:
                part = (
                    instruction.split("on")[1] if "on" in instruction else instruction
                )
                print(f"Step: Turn {expected} to {part} ({duration})")
            turn_count += 1
        else:
            print(f"Step: {turn_right(instruction)} ({duration})")

        next_address_coords = get_coordinates(next_address)
        if next_address_coords:
            car_positions.append(next_address_coords)
        current_address = next_address

    if car_positions:
        start_coords = get_coordinates(starting_address)
        if start_coords:
            latitudes, longitudes = zip(*car_positions)
            plot_map(latitudes, longitudes, start_coords)


def parse_rule_choice():
    """Get the user's choice of rules ('right', 'left', or 'alternate')."""
    while True:
        print("Choose a set of rules:")
        print("1. Right-turning at intersections (Default)")
        print("2. Left-turning at intersections")
        print("3. Turn right twice, then turn left once, repeat")
        choice = input("Enter the number of your choice (1, 2, or 3): ")
        if choice == "1":
            return "right"
        if choice == "2":
            return "left"
        if choice == "3":
            return "alternate"
        print("Invalid choice. Please enter '1', '2', or '3'.")
