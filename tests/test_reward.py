"""Tests for reward calculation in RLEnvironment."""

import pytest

try:
    from inter_sim_rl.rl_environment import RLEnvironment

    RL_AVAILABLE = True
except Exception:
    RL_AVAILABLE = False


@pytest.mark.skipif(not RL_AVAILABLE, reason="rl_environment deps not installed")
def test_reward_right_rule():
    """Right rule: correct turn_right gives 1.0, wrong turn gives -0.5."""
    env = RLEnvironment(
        starting_address="1600 Amphitheatre Parkway, Mountain View, CA",
        max_intersections=2,
        rule_choice="right",
    )
    # Mock so we don't call get_coordinates in initialize_environment.
    # After init, env.current_state has previous_instruction set by step().
    # We only test calculate_reward(previous_instruction, chosen_action).
    assert env.calculate_reward("Turn right onto Main St", "turn_right") == 1.0
    assert env.calculate_reward("Turn right onto Main St", "turn_left") == -0.5
    assert env.calculate_reward("Turn right onto Main St", "go_straight") == -0.5
    assert env.calculate_reward("Go straight", "turn_right") == 0.0
    assert env.calculate_reward(None, "turn_right") == 0.0


@pytest.mark.skipif(not RL_AVAILABLE, reason="rl_environment deps not installed")
def test_reward_left_rule():
    """Left rule: correct turn_left gives 1.0."""
    env = RLEnvironment(
        starting_address="1600 Amphitheatre Parkway, Mountain View, CA",
        max_intersections=2,
        rule_choice="left",
    )
    assert env.calculate_reward("Turn left onto Oak St", "turn_left") == 1.0
    assert env.calculate_reward("Turn left onto Oak St", "turn_right") == -0.5
    assert env.calculate_reward("Go straight", "turn_left") == 0.0


@pytest.mark.skipif(not RL_AVAILABLE, reason="rl_environment deps not installed")
def test_reward_alternate_rule():
    """Alternate rule: left->right and right->left give 1.0."""
    env = RLEnvironment(
        starting_address="1600 Amphitheatre Parkway, Mountain View, CA",
        max_intersections=2,
        rule_choice="alternate",
    )
    assert env.calculate_reward("Turn left", "turn_right") == 1.0
    assert env.calculate_reward("Turn right", "turn_left") == 1.0
    assert env.calculate_reward("Turn left", "turn_left") == -0.5
    assert env.calculate_reward("Turn right", "turn_right") == -0.5
