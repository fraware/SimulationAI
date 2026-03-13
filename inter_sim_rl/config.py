"""
Central configuration for training, evaluation, and paths.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for training and evaluation."""

    starting_address: str = "1600 Amphitheatre Parkway, Mountain View, CA, 94043"
    max_intersections: int = 10
    rule_choice: str = "right"
    model_dir: str = "models"
    output_dir: str = "output"

    # Training
    num_simulations: int = 100
    epsilon: float = 0.1
    batch_size: int = 32
    gamma: float = 0.95
    target_update_frequency: int = 1000
    episodes_to_collect: int = 10

    # Evaluation
    test_starting_address: Optional[str] = None
    test_max_intersections: Optional[int] = None
    test_rule_choice: Optional[str] = None
    num_test_simulations: int = 50

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.test_starting_address is None:
            self.test_starting_address = self.starting_address
        if self.test_max_intersections is None:
            self.test_max_intersections = self.max_intersections
        if self.test_rule_choice is None:
            self.test_rule_choice = self.rule_choice

    @property
    def model_save_path(self) -> str:
        """Path for saving the full model (directory)."""
        os.makedirs(self.model_dir, exist_ok=True)
        return os.path.join(self.model_dir, "trained_dqn_model")

    @property
    def model_weights_path(self) -> str:
        """Path for saving model weights only."""
        os.makedirs(self.model_dir, exist_ok=True)
        return os.path.join(self.model_dir, "trained_dqn_model.h5")


# Values that indicate the key was never set (rejected at runtime)
_INVALID_API_KEY_VALUES = frozenset({"", "YourAPIKey", "your_api_key_here"})


def get_api_key() -> str:
    """Read Google Maps API key from environment."""
    key = (
        os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("API_KEY") or ""
    ).strip()
    if not key or key in _INVALID_API_KEY_VALUES:
        raise ValueError(
            "Google Maps API key not set. Set GOOGLE_MAPS_API_KEY or "
            "API_KEY environment variable."
        )
    return key
