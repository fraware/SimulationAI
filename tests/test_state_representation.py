"""Tests for StateRepresentation and state vector encoding."""

import numpy as np

from inter_sim_rl.state_representation import (
    NEARBY_PLACES_FEATURE_SIZE,
    StateRepresentation,
)


def test_state_vector_shape():
    """State vector has fixed size: 2 + 2 + NEARBY_PLACES_FEATURE_SIZE."""
    state = StateRepresentation(
        current_location=(1.0, 2.0),
        current_direction=(0.0, 1.0),
        nearby_streets=[],
    )
    vec = state.get_state_vector()
    expected_len = 2 + 2 + NEARBY_PLACES_FEATURE_SIZE
    assert vec.shape == (expected_len,)
    assert vec.dtype == np.float64


def test_state_vector_location_direction():
    """Location and direction are correctly encoded."""
    state = StateRepresentation(
        current_location=(10.5, -20.3),
        current_direction=(1.0, 0.0),
        nearby_streets=[],
    )
    vec = state.get_state_vector()
    np.testing.assert_array_almost_equal(vec[:2], [10.5, -20.3])
    np.testing.assert_array_almost_equal(vec[2:4], [1.0, 0.0])


def test_state_vector_empty_nearby():
    """Empty or None nearby_streets yields zeros for nearby part."""
    state = StateRepresentation(
        current_location=(0.0, 0.0),
        current_direction=(0.0, 0.0),
        nearby_streets=[],
    )
    vec = state.get_state_vector()
    np.testing.assert_array_equal(vec[4:], np.zeros(NEARBY_PLACES_FEATURE_SIZE))


def test_state_vector_nearby_from_api_results():
    """Nearby places from API-style dict are encoded."""
    nearby = {
        "results": [
            {"geometry": {"location": {"lat": 1.0, "lng": 2.0}}},
            {"geometry": {"location": {"lat": 3.0, "lng": 4.0}}},
        ]
    }
    state = StateRepresentation(
        current_location=(0.0, 0.0),
        current_direction=(0.0, 0.0),
        nearby_streets=nearby,
    )
    vec = state.get_state_vector()
    assert vec[4] == 1.0
    assert vec[5] == 2.0
    assert vec[6] == 3.0
    assert vec[7] == 4.0


def test_state_representation_optional_fields():
    """current_address and previous_instruction are optional."""
    state = StateRepresentation(
        current_location=(0.0, 0.0),
        current_direction=(0.0, 0.0),
        nearby_streets=[],
        current_address="123 Main St",
        previous_instruction="turn_right",
    )
    assert state.current_address == "123 Main St"
    assert state.previous_instruction == "turn_right"
    state2 = StateRepresentation((0, 0), (0, 0), [])
    assert state2.current_address == ""
    assert state2.previous_instruction is None
