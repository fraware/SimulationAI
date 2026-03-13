"""
Main entry point: training, evaluation, and demo for intersection navigation DQN.
"""

import logging
import random
from typing import List, Tuple

import mlflow
import numpy as np
import tensorflow as tf

from inter_sim_rl.config import Config
from inter_sim_rl.dqn_model import DQNModel
from inter_sim_rl.driving_directions import (
    parse_rule_choice,
    simulate_driving_alternate,
    simulate_driving_left,
    simulate_driving_right,
)
from inter_sim_rl.google_maps_api import gmaps
from inter_sim_rl.rl_environment import ACTION_SPACE, RLEnvironment
from inter_sim_rl.state_representation import StateRepresentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_input_dim() -> int:
    """Compute state vector dimension from StateRepresentation."""
    empty_state = StateRepresentation(
        current_location=(0.0, 0.0),
        current_direction=(0.0, 0.0),
        nearby_streets=[],
    )
    return len(empty_state.get_state_vector())


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_training(config: Config) -> Tuple[DQNModel, float, float]:
    """
    Run training loop; return the trained model and final success rate and avg time.
    """
    if config.seed is not None:
        _set_seed(config.seed)
        logger.info("Random seed set to %s", config.seed)

    input_dim = _get_input_dim()
    output_dim = len(ACTION_SPACE)
    dqn_model = DQNModel(input_dim, output_dim)
    env = RLEnvironment(
        config.starting_address,
        config.max_intersections,
        config.rule_choice,
    )

    data_collection: List[Tuple] = []
    success_count = 0
    total_time_to_destination = 0.0
    path: List[Tuple[float, float]] = []
    rewards_list: List[float] = []
    actions_list: List[str] = []
    correct_actions: List[bool] = []

    mlflow.set_experiment("Navigation Experiment")
    mlflow.start_run()
    mlflow.log_param("num_simulations", config.num_simulations)
    mlflow.log_param("epsilon", config.epsilon)
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("gamma", config.gamma)
    mlflow.log_param("target_update_frequency", config.target_update_frequency)
    mlflow.log_param("episodes_to_collect", config.episodes_to_collect)

    global_step = 0
    for episode_idx in range(config.num_simulations):
        current_state = env.initialize_environment(config.starting_address)
        env.intersection_count = 0
        time_to_destination = 0

        while not env.is_episode_terminated():
            try:
                nearby = gmaps.places_nearby(
                    location=current_state.current_address, radius=100, type="route"
                )
            except Exception:
                nearby = {"results": []}
            current_state.nearby_streets = nearby

            state_vector = current_state.get_state_vector()
            if np.random.rand() < config.epsilon:
                chosen_action = str(np.random.choice(ACTION_SPACE))
            else:
                q_values = dqn_model.model.predict(np.array([state_vector]), verbose=0)[
                    0
                ]
                chosen_action = ACTION_SPACE[int(np.argmax(q_values))]

            reward = env.calculate_reward(
                current_state.previous_instruction, chosen_action
            )
            next_state, reward, done = env.step(chosen_action)

            data_collection.append((current_state, chosen_action, reward, next_state))
            path.append(current_state.current_location)
            rewards_list.append(reward)
            actions_list.append(chosen_action)
            correct_actions.append(
                chosen_action == (current_state.previous_instruction or "")
            )

            current_state = next_state
            time_to_destination += 1
            global_step += 1

            if done:
                if env.intersection_count >= config.max_intersections:
                    success_count += 1
                    total_time_to_destination += time_to_destination
                break

        if len(data_collection) >= config.episodes_to_collect:
            n_batches = len(data_collection) // config.batch_size
            for _ in range(min(n_batches, 10)):
                batch_indices = random.sample(
                    range(len(data_collection)), config.batch_size
                )
                batch_data = [data_collection[i] for i in batch_indices]
                states = np.array(
                    [episode[0].get_state_vector() for episode in batch_data]
                )
                actions = np.array([episode[1] for episode in batch_data])
                rewards = np.array([episode[2] for episode in batch_data])
                next_states = np.array(
                    [episode[3].get_state_vector() for episode in batch_data]
                )
                q_values_current = dqn_model.model.predict(states, verbose=0)
                q_values_next = dqn_model.target_model.predict(next_states, verbose=0)
                targets = rewards + config.gamma * np.max(q_values_next, axis=1)
                for i in range(len(batch_data)):
                    idx = ACTION_SPACE.index(actions[i])
                    q_values_current[i][idx] = targets[i]
                dqn_model.model.fit(
                    states,
                    q_values_current,
                    batch_size=config.batch_size,
                    epochs=1,
                    verbose=0,
                )

            if (episode_idx + 1) % config.target_update_frequency == 0:
                dqn_model.update_target_model()

        env.visualize_agent_path(path, rewards_list, actions_list, correct_actions)
        path.clear()
        rewards_list.clear()
        actions_list.clear()
        correct_actions.clear()

    success_rate = (
        (success_count / config.num_simulations * 100)
        if config.num_simulations > 0
        else 0.0
    )
    average_time = (
        total_time_to_destination / success_count if success_count > 0 else 0.0
    )
    logger.info(
        "Training finished: success_rate=%.2f%%, avg_time=%.2f",
        success_rate,
        average_time,
    )
    mlflow.log_metric("success_rate", success_rate)
    mlflow.log_metric("average_time_to_destination", average_time)

    dqn_model.model.save(config.model_save_path)
    mlflow.log_artifact(config.model_save_path)
    mlflow.end_run()

    return dqn_model, success_rate, average_time


def run_evaluation(config: Config, dqn_model: DQNModel) -> Tuple[float, float]:
    """Run evaluation loop; return test success rate and average time."""
    test_env = RLEnvironment(
        config.test_starting_address or config.starting_address,
        config.test_max_intersections or config.max_intersections,
        config.test_rule_choice or config.rule_choice,
    )
    success_count = 0
    total_time_to_destination = 0.0
    path = []
    rewards_list = []
    actions_list = []
    correct_actions = []

    for _ in range(config.num_test_simulations):
        mlflow.start_run(run_name="Testing Run")
        current_state = test_env.initialize_environment(
            config.test_starting_address or config.starting_address
        )
        test_env.intersection_count = 0
        time_to_destination = 0

        while not test_env.is_episode_terminated():
            try:
                nearby = gmaps.places_nearby(
                    location=current_state.current_address,
                    radius=100,
                    type="route",
                )
            except Exception:
                nearby = {"results": []}
            current_state.nearby_streets = nearby

            state_vector = current_state.get_state_vector()
            q_values = dqn_model.model.predict(np.array([state_vector]), verbose=0)[0]
            chosen_action = ACTION_SPACE[int(np.argmax(q_values))]
            next_state, reward, done = test_env.step(chosen_action)

            path.append(current_state.current_location)
            rewards_list.append(reward)
            actions_list.append(chosen_action)
            correct_actions.append(
                chosen_action == (current_state.previous_instruction or "")
            )
            current_state = next_state
            time_to_destination += 1

            if done:
                if test_env.intersection_count >= (
                    config.test_max_intersections or config.max_intersections
                ):
                    success_count += 1
                    total_time_to_destination += time_to_destination
                break

        test_env.visualize_agent_path(path, rewards_list, actions_list, correct_actions)
        path.clear()
        rewards_list.clear()
        actions_list.clear()
        correct_actions.clear()
        mlflow.end_run()

    test_success_rate = (
        success_count / config.num_test_simulations * 100
        if config.num_test_simulations > 0
        else 0.0
    )
    test_avg_time = (
        total_time_to_destination / success_count if success_count > 0 else 0.0
    )
    return test_success_rate, test_avg_time


def main() -> None:
    """Parse options and run training, evaluation, or demo."""
    config = Config()
    logger.info("Starting training...")
    dqn_model, success_rate, avg_time = run_training(config)
    logger.info(
        "Training evaluation: success_rate=%.2f%%, avg_time=%.2f",
        success_rate,
        avg_time,
    )

    logger.info("Running test evaluation...")
    test_success_rate, test_avg_time = run_evaluation(config, dqn_model)
    logger.info(
        "Test evaluation: success_rate=%.2f%%, avg_time=%.2f",
        test_success_rate,
        test_avg_time,
    )

    mlflow.start_run()
    mlflow.log_metric("test_success_rate", test_success_rate)
    mlflow.log_metric("test_average_time_to_destination", test_avg_time)
    mlflow.end_run()


if __name__ == "__main__":
    import sys

    from inter_sim_rl.config import Config

    config = Config()
    config.starting_address = "1600 Amphitheatre Parkway, Mountain View, CA, 94043"

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        main()
    else:
        rule_choice = parse_rule_choice()
        config.rule_choice = rule_choice
        model_path = config.model_save_path
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception:
            print(
                "No trained model found at",
                model_path,
                "- run with --train first to train a model.",
            )
            sys.exit(1)
        if rule_choice == "right":
            simulate_driving_right(
                config.starting_address,
                max_intersections=100,
                model=model,
                action_space=ACTION_SPACE,
            )
        elif rule_choice == "left":
            simulate_driving_left(
                config.starting_address,
                max_intersections=100,
                model=model,
                action_space=ACTION_SPACE,
            )
        elif rule_choice == "alternate":
            simulate_driving_alternate(
                config.starting_address,
                max_intersections=100,
                model=model,
                action_space=ACTION_SPACE,
            )
        else:
            print("Invalid rule choice.")
