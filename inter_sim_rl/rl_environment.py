"""
RL environment for intersection navigation: state, rewards, step, viz.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from inter_sim_rl.google_maps_api import (
    get_coordinates,
    get_random_nearby_place,
)
from inter_sim_rl.state_representation import StateRepresentation

ACTION_SPACE = ["turn_left", "turn_right", "go_straight", "turn_back"]


class RLEnvironment:
    """
    Reinforcement learning environment for navigation.
    """

    def __init__(
        self,
        starting_address: str,
        max_intersections: int,
        rule_choice: str,
    ) -> None:
        """
        Initialize the RL environment.

        Parameters:
            starting_address: Starting address for navigation.
            max_intersections: Maximum number of intersections to navigate.
            rule_choice: Navigation rule ('right', 'left', 'alternate').
        """
        self.starting_address = starting_address
        self.max_intersections = max_intersections
        self.rule_choice = rule_choice
        self.intersection_count = 0
        self.current_state = self.initialize_environment(starting_address)
        self.action_space = ACTION_SPACE.copy()

    def initialize_environment(self, current_address: str) -> StateRepresentation:
        """
        Initialize the environment state at the given address.

        Parameters:
            current_address: Current address for navigation.

        Returns:
            Initialized state as StateRepresentation.
        """
        coords = get_coordinates(current_address)
        if coords is None:
            coords = (0.0, 0.0)
        return StateRepresentation(
            current_location=coords,
            current_direction=(0.0, 0.0),
            nearby_streets=[],
            current_address=current_address,
            previous_instruction=None,
        )

    def is_episode_terminated(self) -> bool:
        """Return True if the episode is terminated."""
        return self.intersection_count >= self.max_intersections

    def step(self, chosen_action: str) -> Tuple[StateRepresentation, float, bool]:
        """
        Perform a step in the environment based on the chosen action.

        Parameters:
            chosen_action: Chosen action to take.

        Returns:
            Tuple of (next state, reward, done).
        """
        reward = self.calculate_reward(
            self.current_state.previous_instruction, chosen_action
        )
        next_state = self.update_environment(chosen_action)
        if "turn" in chosen_action:
            self.intersection_count += 1
        done = self.is_episode_terminated()
        self.current_state = next_state
        return next_state, reward, done

    def update_environment(self, chosen_action: str) -> StateRepresentation:
        """
        Update the environment given the chosen action; return the new state.
        """
        next_address = get_random_nearby_place(self.current_state.current_address)
        if next_address is None:
            # Stay at same place if no nearby place found
            return StateRepresentation(
                current_location=self.current_state.current_location,
                current_direction=self.current_state.current_direction,
                nearby_streets=self.current_state.nearby_streets,
                current_address=self.current_state.current_address,
                previous_instruction=chosen_action,
            )
        coords = get_coordinates(next_address)
        if coords is None:
            coords = self.current_state.current_location
        return StateRepresentation(
            current_location=coords,
            current_direction=(0.0, 0.0),
            nearby_streets=[],
            current_address=next_address,
            previous_instruction=chosen_action,
        )

    def get_observation(self) -> StateRepresentation:
        """Return the current observation (state)."""
        return self.current_state

    def visualize_agent_path(
        self,
        path: List[Tuple[float, float]],
        rewards: List[float],
        actions: List[str],
        correct_actions: List[bool],
    ) -> None:
        """
        Visualize the agent's path, actions, rewards, and correctness.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(*zip(*path), marker="o", color="b", label="Visited Locations")
        for i in range(len(path)):
            action = actions[i]
            reward = rewards[i]
            correct = correct_actions[i]
            color = "g" if correct else "r"
            plt.annotate(
                f"Step {i + 1}\nAction: {action}\nReward: {reward:.2f}",
                path[i],
                textcoords="offset points",
                xytext=(-15, 10),
                ha="center",
                color=color,
            )
        plt.title("Agent's Path, Actions, Rewards, and Correctness")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid()
        plt.show()

    def calculate_reward(
        self,
        previous_instruction: Optional[str],
        chosen_action: str,
    ) -> float:
        """
        Calculate reward for chosen action from previous instruction.
        """
        if previous_instruction is None:
            return 0.0
        prev = previous_instruction.lower()
        if self.rule_choice == "right":
            if "right" in prev and chosen_action == "turn_right":
                return 1.0
            if "right" in prev:
                return -0.5
            return 0.0
        if self.rule_choice == "left":
            if "left" in prev and chosen_action == "turn_left":
                return 1.0
            if "left" in prev:
                return -0.5
            return 0.0
        if self.rule_choice == "alternate":
            if "left" in prev and chosen_action == "turn_right":
                return 1.0
            if "right" in prev and chosen_action == "turn_left":
                return 1.0
            if "left" in prev or "right" in prev:
                return -0.5
            return 0.0
        return 0.0
