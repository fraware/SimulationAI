"""Tests for Config."""

import os

import pytest

from inter_sim_rl.config import Config, get_api_key


def test_config_defaults():
    """Config has expected defaults and model_save_path under model_dir."""
    c = Config()
    assert c.num_simulations == 100
    assert c.epsilon == 0.1
    assert c.batch_size == 32
    assert "trained_dqn_model" in c.model_save_path
    assert c.model_dir in c.model_save_path


def test_config_test_defaults_from_train():
    """Test address/intersections/rule default to training values."""
    c = Config()
    assert c.test_starting_address == c.starting_address
    assert c.test_max_intersections == c.max_intersections
    assert c.test_rule_choice == c.rule_choice


def test_get_api_key_uses_env():
    """get_api_key returns env value when set."""
    os.environ["API_KEY"] = "valid-test-key"
    try:
        assert get_api_key() == "valid-test-key"
    finally:
        os.environ.pop("API_KEY", None)


def test_get_api_key_raises_when_missing():
    """get_api_key raises when API_KEY and GOOGLE_MAPS_API_KEY are unset."""
    prev1 = os.environ.pop("API_KEY", None)
    prev2 = os.environ.pop("GOOGLE_MAPS_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="API key not set"):
            get_api_key()
    finally:
        if prev1 is not None:
            os.environ["API_KEY"] = prev1
        if prev2 is not None:
            os.environ["GOOGLE_MAPS_API_KEY"] = prev2
