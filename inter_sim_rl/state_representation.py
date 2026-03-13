"""
State representation for the RL agent: location, direction, nearby streets.
"""

from typing import Any, Optional, Tuple

import numpy as np

# Fixed size for nearby places encoding (e.g. 10 places x lat+lng)
NEARBY_PLACES_FEATURE_SIZE = 20


class StateRepresentation:
    """
    Represents the current state in the reinforcement learning environment.
    """

    def __init__(
        self,
        current_location: Tuple[float, float],
        current_direction: Tuple[float, float],
        nearby_streets: Any,
        current_address: Optional[str] = None,
        previous_instruction: Optional[str] = None,
    ) -> None:
        """
        Initialize the state representation.

        Parameters:
            current_location: Current GPS coordinates (lat, lng).
            current_direction: Current direction vector.
            nearby_streets: Nearby street/place info (list or API response dict).
            current_address: Current address string (for env step and API calls).
            previous_instruction: Previous navigation instruction (for reward).
        """
        self.current_location = current_location
        self.current_direction = current_direction
        self.nearby_streets = nearby_streets
        self.current_address = current_address or ""
        self.previous_instruction = previous_instruction

    def get_state_vector(self) -> np.ndarray:
        """
        Convert the state into a fixed-size feature vector.

        Returns:
            Feature vector: [location (2), direction (2), nearby_encoding (fixed)].
        """
        loc = self.current_location
        direction = self.current_direction
        if not isinstance(loc, (tuple, list)) or len(loc) < 2:
            location_vector = np.zeros(2, dtype=np.float64)
        else:
            location_vector = np.array([float(loc[0]), float(loc[1])], dtype=np.float64)

        if not isinstance(direction, (tuple, list)) or len(direction) < 2:
            direction_vector = np.zeros(2, dtype=np.float64)
        else:
            direction_vector = np.array(
                [float(direction[0]), float(direction[1])], dtype=np.float64
            )

        nearby_vector = _encode_nearby_streets(self.nearby_streets)
        return np.concatenate((location_vector, direction_vector, nearby_vector))


def _encode_nearby_streets(nearby_streets: Any) -> np.ndarray:
    """Encode nearby_streets (API response or list) into a fixed-size vector."""
    out = np.zeros(NEARBY_PLACES_FEATURE_SIZE, dtype=np.float64)
    if nearby_streets is None:
        return out
    if isinstance(nearby_streets, dict):
        results = nearby_streets.get("results", nearby_streets)
    else:
        results = nearby_streets
    if not isinstance(results, list):
        return out
    count = 0
    for i, place in enumerate(results):
        if count * 2 >= NEARBY_PLACES_FEATURE_SIZE:
            break
        try:
            geo = place.get("geometry", {}) if isinstance(place, dict) else {}
            loc = geo.get("location", {})
            if isinstance(loc, dict):
                lat = loc.get("lat", 0.0)
                lng = loc.get("lng", 0.0)
            else:
                continue
            out[count * 2] = float(lat)
            out[count * 2 + 1] = float(lng)
            count += 1
        except (TypeError, KeyError):
            continue
    return out
